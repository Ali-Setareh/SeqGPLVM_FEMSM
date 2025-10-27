from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.priors import Prior,NormalPrior, GammaPrior, SmoothedBoxPrior
from gpytorch.models.gplvm import  VariationalLatentVariable
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from torch.distributions import Distribution
import gpytorch
import torch, math
import torch.nn as nn
from models.GPLVM import GPLVM, SGPRModel
from copy import deepcopy
from tqdm.notebook import trange
from typing import Optional, Literal
from utils.inspectors import get_actuals_via_getters
from torch.quasirandom import SobolEngine 

@torch.no_grad()
def farthest_points(X: torch.Tensor, M: int, *, seed: int = 42) -> torch.Tensor:
    """
    X: [N, d] (no NaNs); returns M rows of X chosen by greedy farthest-point sampling.
    Always keeps 'dist' as [N] to avoid argmax flattening surprises.
    """
    if X.ndim != 2:
        raise ValueError("X must be [N, d].")
    N = X.size(0)
    if N == 0:
        raise ValueError("Empty X.")
    M = min(int(M), N)

    # start index (random or deterministic)
    if seed is not None:
        g = torch.Generator(device=X.device); g.manual_seed(seed)
        start = int(torch.randint(N, (1,), generator=g, device=X.device).item())
    else:
        start = int(torch.randint(N, (1,), device=X.device).item())

    idx = [start]

    # distances to the selected set (init with the first point)
    dist = torch.cdist(X, X[start].unsqueeze(0)).squeeze(1)  # [N]
    dist[start] = -1.0  # mask the chosen point

    for _ in range(1, M):
        nxt = int(torch.argmax(dist).item())  # in [0, N-1]
        idx.append(nxt)
        # update with distance to the newly chosen point (keep it 1-D)
        dist = torch.minimum(dist, torch.cdist(X, X[nxt].unsqueeze(0)).squeeze(1))
        dist[nxt] = -1.0  # mask selected

    return X[idx]  # [M, d]

@torch.no_grad()
def init_inducing_Z(Mz: int,
                    Q: int,
                    *,
                    prior_std: float | None = None,        # s0 for Normal(0, s0^2)
                    uniform_halfwidth: float | None = None, # a for Uniform[-a, a]
                    method: str = "sobol-normal",          # "sobol-normal", "sobol-uniform", "lhs-uniform"
                    include_endpoints: bool = True,         # for Q=1 + uniform
                    device=None, dtype=None, seed: int | None = None) -> torch.Tensor:
    """
    Returns Z_u of shape [Mz, Q] for inducing points along the latent dimension.

    - If method ends with '-normal', prior_std must be set (uses Normal(0, s0^2)).
    - If method ends with '-uniform', uniform_halfwidth must be set (uses [-a, a]^Q).
    """

    device = device or torch.device("cpu")
    dtype = dtype or torch.get_default_dtype()

    if method.endswith("-normal"):
        if prior_std is None:
            raise ValueError("init_inducing_Z: prior_std required for *-normal methods.")
        # Sobol in (0,1), avoid exact 0/1 to keep erfinv finite
        eng = SobolEngine(dimension=Q, scramble=True, seed=seed)
        U = eng.draw(Mz).to(device=device, dtype=dtype).clamp(1e-6, 1-1e-6)  # [Mz,Q]
        # Map U -> N(0,1) via inverse error function, then scale by s0
        Z = math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0) * prior_std
        return Z

    elif method.endswith("-uniform"):
        if uniform_halfwidth is None:
            raise ValueError("init_inducing_Z: uniform_halfwidth required for *-uniform methods.")
        a = float(uniform_halfwidth)

        if method == "sobol-uniform":
            eng = SobolEngine(dimension=Q, scramble=True, seed=seed)
            U = eng.draw(Mz).to(device=device, dtype=dtype)                  # [Mz,Q] in [0,1]
        elif method == "lhs-uniform":
            # Latin hypercube: stratify each dim
            U = torch.empty(Mz, Q, device=device, dtype=dtype)
            base = torch.linspace(0, 1, steps=Mz+1, device=device, dtype=dtype)
            for j in range(Q):
                # pick one point uniformly from each stratum, then permute
                u = base[:-1] + (base[1:] - base[:-1]) * torch.rand(Mz, device=device, dtype=dtype)
                perm = torch.randperm(Mz, device=device)
                U[:, j] = u[perm]
        else:
            raise ValueError(f"Unknown method: {method}")

        Z = (2.0 * U - 1.0) * a  # map [0,1] -> [-a,a]
        # Nice endpoints for Q==1 and very small Mz
        if Q == 1 and include_endpoints and Mz >= 2 and method != "lhs-uniform":
            Z[0, 0] = -a
            Z[-1, 0] = a
        return Z

    else:
        raise ValueError(f"Unknown method: {method}")
    
@torch.no_grad()
def prior_same_type_resized(prior_train: Prior, n_val: int, q: int, device) -> Prior:
    """
    Keep the *type* and parameters of prior_train, but resize batch shape for validation.
    Handles Normal, Independent(Normal,1), and MultivariateNormal.
    """
    p = prior_train

    if isinstance(p, NormalPrior) :
        # Keep a broad Normal with the same std as train (defaulting to 2.0 if not encoded)
        std = 2.0
        Z_prior_mean = torch.zeros(n_val, q, requires_grad=False, device=device)  # shape: N_val x Q
        prior_Z = NormalPrior(
            Z_prior_mean,
            std * torch.ones_like(
                Z_prior_mean, requires_grad=False, device=device
            ),
        )
        return prior_Z
    elif isinstance(p, SmoothedBoxPrior):
        # Recreate the same box bounds for validation. Try to recover train-time bounds
        # and expand them to [n_val, q]. If bounds were tensors of shape [N_train,q],
        # we preserve per-dimension values from the first row.
       
        a = 2.0  # broad, lets data speak
        Z_prior_mean = torch.zeros(n_val, q, requires_grad=False, device=device)  # shape: N x Q
        prior_Z = SmoothedBoxPrior(
            -a * torch.ones_like(Z_prior_mean, requires_grad=False, device=device),
            a * torch.ones_like(Z_prior_mean, requires_grad=False, device=device),
        )
        return prior_Z
    else:
        raise TypeError(f"Unsupported prior type for resizing: {type(p)}")

class SeqGPLVM(nn.Module):
    '''
    dim(Y) = N x T
    dim(X_cov) = N x T x (dim_cov + 1) (dim_cov for covariates and 1 for last treatment)
    dim(Z) = latent_dim
    '''
    def __init__(
        self,
        *,
        Y: torch.Tensor,
        X_cov: torch.Tensor,
        latent_dim: int,
        n_inducing_x: int,
        n_inducing_hidden: int, 
        init_z: torch.Tensor = None, 
        z_prior: Literal['normal', 'uniform'] = 'normal',
        device = None, 
        lik = None, #gpytorch.likelihoods.GaussianLikelihood()
        learn_inducing_locations = False , 
        use_titsias = False, 
        z_initializer: Literal['normal', 'uniform'] = 'normal', # how to initialize Z INDUCING points
        uniform_halfwidth: float | None = None, # a for Uniform[-a, a]
        prior_std: float | None = None,        # s0 for Normal(0, s0^2)
 
    ):
        """
        Y: (N, T) tensor of targets for each of T GPs
        X_cov: (N, T, C) tensor of covariates per data and task
        latent_dim: Q
        n_inducing: number of inducing points per GP
        init_z: (N, Q) init for shared latent; if None, random
        """
        super().__init__()

        if z_initializer not in {"uniform", "normal"}:
            raise ValueError(f"z_initializer must be 'uniform' or 'normal', got {z_initializer!r}")

        self.config = {
            'latent_dim': latent_dim,
            'n_inducing_x': n_inducing_x,
            'n_inducing_hidden': n_inducing_hidden,
            'init_z': init_z,
            'device': device,
            'lik': lik,
            'learn_inducing_locations': learn_inducing_locations,
            'use_titsias': use_titsias,
        }

        if use_titsias and not issubclass(lik, GaussianLikelihood):
            raise TypeError("Titsias bound requires a GaussianLikelihood class")
        
        self.use_titsias = use_titsias
        N, T = Y.shape
        C = X_cov.shape[-1]
        self.N, self.T, self.C, self.Q = N, T, C, latent_dim
        self.Y = Y
        self.X_cov = X_cov
        self.lik = lik

        # LatentVariable
        if init_z is None:
             init_z = torch.nn.Parameter(torch.randn(N, latent_dim))

        # Define prior for Z
        if z_prior == 'normal':
            Z_prior_mean = torch.zeros(N, latent_dim, requires_grad=False, device=device)  # shape: N x Q
            std = 2.0  # broad, lets data speak (std); 
            prior_Z = NormalPrior(
                Z_prior_mean,
                std * torch.ones_like(
                    Z_prior_mean, requires_grad=False, device=device
                ),
            )  # requires_grad=False because we dont want to learn the prior hyperparams
        elif z_prior == 'uniform':
            a = 2.0  # broad, lets data speak
            Z_prior_mean = torch.zeros(N, latent_dim, requires_grad=False, device=device)  # shape: N x Q
            prior_Z = SmoothedBoxPrior(
                -a * torch.ones_like(Z_prior_mean, requires_grad=False, device=device),
                 a * torch.ones_like(Z_prior_mean, requires_grad=False, device=device),
            )

        self.Z = VariationalLatentVariable(N, T, 
                                           latent_dim, init_z, 
                                           prior_Z)
        
        if not callable(lik):
            raise ValueError("lik must be a likelihood class or factory")
        
        self.lik_factory = lik

        # Create T separate GPLVMs
        self.gps = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        self.mlls = []
       
        #Xu_Z  = torch.randn(n_inducing_hidden, latent_dim).to(device)
        Xu_Z = init_inducing_Z(n_inducing_hidden, 
                               latent_dim, 
                               uniform_halfwidth = uniform_halfwidth if z_initializer == "uniform" else None, 
                               prior_std= prior_std if z_initializer == "normal" else None,
                               method="lhs-uniform" if z_initializer == "uniform" else "sobol-normal", 
                               seed=0, device=device)

        for t in range(T):
            ### NEW ###
            mask  = ~torch.isnan(X_cov[:,t,:]).any(dim=1)
            covs = X_cov[:,t,:][mask]
            #perm = torch.randperm(covs.size(0))[:n_inducing_x]
            # choose some of the non nan Xs to be the inducing points. 
            #Xu_X  = covs [perm].to(device)
            Xu_X = farthest_points(covs, n_inducing_x).to(device)
            ### NEW ###

            subset = ~(torch.isnan(Y[:,t]).reshape(-1).detach().cpu())
            n = subset.float().sum()
    
            # likelihood and ELBO
            lik_t = self.lik_factory()
            if self.lik_factory == GaussianLikelihood: 
                lik_t.noise_covar.register_constraint("raw_noise", GreaterThan(1e-3))
                #lik_t.noise_covar.register_prior("noise_prior", GammaPrior(2.0, 0.5),"raw_noise")

            self.likelihoods.append(lik_t)
            if use_titsias: 
                mean_init = Y[subset, t:t+1].mean().item()
                gp = SGPRModel(
                    train_x = torch.cat([X_cov[subset, t, :], Z_prior_mean[subset,:]], dim=1), # the z part will be override in the forward method
                    train_y = Y[subset, t],
                    likelihood=lik_t,
                    x_inducing = Xu_X ,
                    z_inducing = Xu_Z,
                    mean_init=mean_init
                )
                if not learn_inducing_locations:
                    gp.covar_module.inducing_points.requires_grad_(False)
                
                mll = ExactMarginalLogLikelihood(lik_t, gp)
                
            
            else: 
                gp = GPLVM(n, x_inducing = Xu_X, z_inducing = Xu_Z,
                           learn_inducing_locations = learn_inducing_locations)
                mll = VariationalELBO(lik_t, gp, num_data=n)

            self.gps.append(gp)
            self.mlls.append(mll)
    
    def sample_latent_variable(self):
        sample = self.Z()
        return sample
    
    def forward(self):      #, batch_idx: torch.Tensor):
        # Sample shared latent for batch
        Z_sample = self.sample_latent_variable()
        loss = 0.0
        
        for t, (gp, mll) in enumerate(zip(self.gps, self.mlls)):
            Yt = self.Y[:, t]
            Xt = self.X_cov[:,t,:]
            subset = ~torch.isnan(Yt).reshape(-1).detach()
            if subset.sum() > 0:
                yt = Yt[subset]
                zt = Z_sample[subset, :]
                xt = Xt[subset, :]

                Xjoint = torch.cat([xt,zt], dim=1)    # (B, Q+C)
                
                if self.use_titsias: 
                    gp.set_train_data(
                        inputs=(Xjoint,),
                        targets=yt,
                        strict=False,           # allow changing the size of the data
                        )
                
                with gpytorch.settings.cholesky_jitter(1e-3):
                    f_dist = gp(Xjoint)
                    elbo = -mll(f_dist, yt).sum()
                    loss+= elbo 
        
        # --- add the Z-KL correctly ---
        # Pull the added-loss term that VariationalLatentVariable registered on itself
        kl_z = sum(term.loss() for term in self.Z.added_loss_terms())
        loss += self.T * kl_z            # multiply by data_dim (see note above)
        
        return loss

    @torch.no_grad()
    def predict(
        self,
        X_cov_star: torch.Tensor,      # (N*, T, C)
        z_star:     torch.Tensor,      # (z_integral,N*, Q)
        add_lik_var: bool = True,
        z_integral = 100,
        num_mc_samples: int = 64,      # used only when no analytic variance
        predictive_check = False,
        Y_star: Optional[torch.Tensor] = None # (N*, T) or None it is needed only for predictive checks
    
       
    ):
        
        """
        Returns:
            mean : (N*, T)  – predictive mean (or class-prob for Bernoulli)
            sd   : (N*, T)  – posterior s.d.  (Gaussian/Bernoulli) or MC s.d.
        """
        #self.eval()
        #for lik in self.likelihoods:
         #   lik.eval()

        N_star, device = X_cov_star.size(0), z_star.device
        mean = torch.full((N_star, self.T), float("nan"),device=device)
        #var  = torch.empty_like(mean)
        
        a_pred = torch.full((N_star,self.T,z_integral), float("nan"), device = device )

        log_prob_pred = torch.full((N_star,self.T,z_integral), float("nan"), device = device )
        log_prob_true = torch.full((N_star,self.T,z_integral), float("nan"), device = device )

        mask = ~X_cov_star.isnan().any(dim = -1)

        with gpytorch.settings.fast_pred_var():
            for t in range(self.T):
                #print(t)
                # ---------- inputs -------------------------------------------------
                #x_star_t     = X_cov_star[:, t, :]            # (N*, C)
                #X_joint_star = torch.cat([x_star_t, z_star], dim=1)
                #subset = ~X_joint_star.isnan().any(dim = 1)
                mask_t = mask[:,t]
                if mask_t.sum() == 0:
                    continue
                n = mask_t.int().sum().item()
                x_star_t     = X_cov_star[mask_t, t, :].unsqueeze(0)         # (1,N*[valid], C)
                
                x_star_t = x_star_t.expand(z_integral, -1, -1) #[K, N*[valid], C]
                X_joint_star = torch.cat([x_star_t, z_star[:,mask_t,:]], dim=2) #[K, N*[valid], C+Q]
                X_joint_star   = X_joint_star.reshape(z_integral * n, self.Q + self.C)    # → [K*N_val, C+Q]

                
                

                latent_dist  = self.gps[t](X_joint_star)      # f_* | X_*

                lik_cls      = type(self.likelihoods[t])      # concrete class

                

                # ---------- 1. Gaussian likelihood ---------------------------------
                if lik_cls is GaussianLikelihood:
                    pred = ( self.likelihoods[t](latent_dist)
                             if add_lik_var else latent_dist )
                    #mean[subset, t] = pred.mean
                    #var[subset,  t] = pred.variance
                    mean = pred.mean            # [K*N_val[subset]]
                    #var  = pred.variance        # [K*N_val]
                 
                    


                # ---------- 2. Bernoulli likelihood --------------------------------
                elif lik_cls is BernoulliLikelihood:
                    pred        = self.likelihoods[t](latent_dist)   # Bernoulli
                    #mean[subset, t]  = pred.mean           
                    #var[subset,  t]  = pred.variance      
                    mean = pred.mean            # [K*N_val[subset]]  class-1 prob p

                    #var  = pred.variance        # [K*N_val[subset]]  # == p(1-p)

                # ---------- 3. everything else  (Softmax, Poisson, custom …) -------
                else:
                    pred = self.likelihoods[t](latent_dist)

                    # (a) If we still got a Distribution and it has mean/variance,
                    #     use them directly – cheaper than MC.
                    if isinstance(pred, Distribution) \
                       and hasattr(pred, "mean") and hasattr(pred, "variance"):
                        mean = pred.mean.squeeze(-1)
                        #var[subset,  t] = pred.variance.squeeze(-1)

                    # (b) Otherwise draw Monte-Carlo samples and estimate.
                    else:
                        try:
                            samples = pred.rsample(
                                torch.Size([num_mc_samples]))
                        except AttributeError:
                            samples = pred.sample(
                                torch.Size([num_mc_samples]))

                        # samples: (S, N*)   →  estimate mean / var along S
                        #mean[subset, t] = samples.mean(0)
                        #var[subset,  t] = samples.var(0, unbiased=False)
                        mean = samples.mean(0)
                        #var = samples.var(0, unbiased=False)

                mean_t = mean.reshape(z_integral,n)  # [K,N_val[subset]]
                #var_t  = var.reshape(z_integral, N_star)   # [K, N_val[subset]]
                a_pred[mask_t, t, :] = mean_t.transpose(0, 1)

                # predictive check
                if predictive_check: 

                    ####    NEW   ###########################
                    #f_sample   = latent_dist.rsample()
                    #########################################


                    Y_t_valid = Y_star[mask_t, t]          # [N_valid]
                    y_dist = self.likelihoods[t](latent_dist)   # OLD                       # Distribution(batch=[K*N_valid])
                    #y_dist = self.likelihoods[t](f_sample) # NEW
                    

                    if self.use_titsias: 

                        μ_flat = y_dist.mean       # [K * n_valid]
                        σ2_flat= y_dist.variance   # [K * n_valid] (already adds noise var)

                        # Reshape back to [K, n_valid]:
                        μ_t = μ_flat.view(z_integral, n)    # [K, n_valid]
                        σ2_t= σ2_flat.view(z_integral, n)   # [K, n_valid]

                        Y_flat     = Y_t_valid.unsqueeze(0).expand(z_integral, -1)  # [K, n_valid]
                        
                        # log p_N(y | μ, σ²) = -0.5 [ log(2π σ²) + (y - μ)²/σ² ]
                        ll_flat_true = -0.5 * (
                            torch.log(2 * torch.pi * σ2_t)
                            + ((Y_flat - μ_t) ** 2).div(σ2_t)
                        )  # [K, n_valid]

                        eps      = torch.randn_like(μ_t)                # [K, n_valid]
                        A_pred   = μ_t + torch.sqrt(σ2_t) * eps          # [K, n_valid]

                        ll_flat_pred = -0.5 * (
                        torch.log(2 * torch.pi * σ2_t)
                        + ((A_pred - μ_t) ** 2).div(σ2_t)
                            )  # [K, n_valid]
                        
                        # scatter into a_ll[n, t, k] in exactly the same way:
                        log_prob_true[mask_t, t, :] = ll_flat_true.transpose(0, 1)  # [N_valid, K]
                        log_prob_pred[mask_t, t, :] = ll_flat_pred.transpose(0, 1)  # [N_valid, K]



                    else: 
                        A_rep_flat = y_dist.sample()

                        #    Now we need to repeat each valid y value K times, in the SAME order that
                        #    X_flat was flattened. The order is: for k in 0..K-1, for i in 0..N_valid-1.
                        #    One easy way is to do:
                        Y_flat = Y_t_valid.unsqueeze(0).expand(z_integral, -1)  # [K, N_valid]
                        Y_flat = Y_flat.reshape(z_integral * n)               # [K*N_valid] 

                        ll_flat_true = self.likelihoods[t].expected_log_prob(Y_flat, latent_dist) # NEW #OLD : latent_dist)  # [K*N_valid]
                        ll_flat_pred = self.likelihoods[t].expected_log_prob(A_rep_flat, latent_dist) #NEW #OLD: latent_dist)  # [K*N_valid]

                        #  Reshape ll_flat back to [K, N_valid] so we can scatter into a_ll:
                        ll_t_true = ll_flat_true.reshape(z_integral, n)  # [K, N_valid]
                        ll_t_pred = ll_flat_pred.reshape(z_integral, n)  # [K, N_valid]

                        # scatter into a_ll[n, t, k] in exactly the same way:
                        log_prob_true[mask_t, t, :] = ll_t_true.transpose(0, 1)  # [N_valid, K]
                        log_prob_pred[mask_t, t, :] = ll_t_pred.transpose(0, 1)  # [N_valid, K]


                
                
                print(f"\r step: {t}", end = "")
        #sd = torch.sqrt(var.clamp_min(1e-9))
        if predictive_check: 
            return a_pred, log_prob_pred,log_prob_true
        else:
            return a_pred #mean, sd
    

    @torch.no_grad()
    def propensity(
        self,
        X_cov_star: torch.Tensor,      # (N*, T, C)
        z_star:     torch.Tensor,      # (K, N*, Q)
        *,
        A_obs: Optional[torch.Tensor] = None,  # (N*, T) observed treatment for GPS/log-prop; optional
        add_lik_var: bool = True,              # must be True for Gaussian GPS (use obs noise)
        z_integral: int = 100,
        sample_count: int = 0,         # number of A samples to draw
        sample_independent: bool = False,  # for Gaussian, sample factorized N(mu,var) instead of full MVN
    ) -> dict:
        """
        Returns a dict with per-(n,t,k) quantities for treatment models:

        - For BernoulliLikelihood:
            'propensity' : (N*, T, K) with p(A=1 | X, Z^k, D)
            if A_obs is provided:
                'log_gps' : (N*, T, K) with log pmf at the observed A in {0,1}
        - For GaussianLikelihood (continuous treatment):
            if A_obs is provided:
                'log_gps' : (N*, T, K) with element-wise Normal log pdf at observed A
            always (for convenience):
                'mu'  : (N*, T, K)
                'var' : (N*, T, K)  predictive variance (includes noise when add_lik_var=True)

        Notes:
        - The K axis corresponds to z_integral latent draws.
        - Use the returned 'log_gps' to build IPTW weights in log-space across t.
        """
        self.eval()
        for gp, lik in zip(self.gps, self.likelihoods):
            gp.eval(); lik.eval()

        N_star, T = X_cov_star.size(0), self.T
        device = z_star.device
        K = z_integral

        prop = torch.full((N_star, T, K), float("nan"), device=device)  # only filled for Bernoulli
        log_gps = torch.full((N_star, T, K), float("nan"), device=device) if A_obs is not None else None
        mu_out  = torch.full((N_star, T, K), float("nan"), device=device)
        var_out = torch.full((N_star, T, K), float("nan"), device=device)
        A_samples = torch.full((sample_count, N_star, T, K), float("nan"), device=device) if sample_count>0 else None
        log_gps_samples = (
        torch.full((sample_count, N_star, T, K), float("nan"), device=device)
        if sample_count > 0 else None
        )


        mask = ~X_cov_star.isnan().any(dim=-1)  # (N*, T)

        with gpytorch.settings.fast_pred_var():
            for t in range(T):
                mask_t = mask[:, t]
                if mask_t.sum() == 0:
                    continue
                n = int(mask_t.sum().item())

                # build [K, n, C+Q] then flatten to [K*n, C+Q]
                x_t = X_cov_star[mask_t, t, :].unsqueeze(0).expand(K, -1, -1)         # [K, n, C]
                xz  = torch.cat([x_t, z_star[:, mask_t, :]], dim=2).reshape(K * n, -1)  # [K*n, C+Q]

                gp_t  = self.gps[t]
                lik_t = self.likelihoods[t]
                latent = gp_t(xz)  # distribution over f at these inputs

                lik_cls = type(lik_t)

                if lik_cls is gpytorch.likelihoods.GaussianLikelihood:
                    # Predictive for A (continuous) — include obs noise for GPS
                    pred = lik_t(latent) if add_lik_var else latent

                    mu_vec  = pred.mean      # [K*n]
                    var_vec = pred.variance  # [K*n]  (diag; includes noise if add_lik_var)

                    # store params (useful even if you only need log_gps)
                    mu_t  = mu_vec.view(K, n).T   # [n, K]
                    var_t = var_vec.view(K, n).T  # [n, K]
                    mu_out[mask_t, t, :]  = mu_t
                    var_out[mask_t, t, :] = var_t

                    # If observed A given, compute element-wise Normal log pdf (GPS)
                    if A_obs is not None:
                        a_t = A_obs[mask_t, t]                                 # [n]
                        a_flat = a_t.unsqueeze(0).expand(K, -1).reshape(K * n) # [K*n]
                        # log N(a | mu, var)
                        log_pdf = -0.5 * (torch.log(2 * torch.pi * var_vec) + (a_flat - mu_vec) ** 2 / var_vec)
                        log_gps[mask_t, t, :] = log_pdf.view(K, n).T           # [n, K]
                    
                    # Optionally draw samples from the predictive (for IPTW diagnostics)
                    if sample_count > 0:
                        if sample_independent:
                            # factorized Normal: mu + sqrt(var)*eps
                            eps = torch.randn(sample_count, K*n, device=device)
                            samp = mu_vec.unsqueeze(0) + eps * var_vec.sqrt().unsqueeze(0)  # [S, K*n]
                        else:
                            # full MVN sample (captures correlation across points)
                            samp = pred.rsample((sample_count,))  # [S, K*n]
                        A_samples[:, mask_t, t, :] = samp.view(sample_count, K, n).permute(0, 2, 1)
                        
                        diag = torch.distributions.Normal(mu_vec, var_vec.sqrt())
                        logpdf_elem = diag.log_prob(samp)               # [S, K*n] element-wise
                        log_gps_samples[:, mask_t, t, :] = logpdf_elem.view(sample_count, K, n).permute(0, 2, 1)

                elif lik_cls is gpytorch.likelihoods.BernoulliLikelihood:
                    # Predictive Bernoulli (probit) — already integrates over f
                    pred = lik_t(latent)                  # Bernoulli over A
                    p_vec = pred.mean                     # [K*n] propensity for A=1
                    prop_t = p_vec.view(K, n).T           # [n, K]
                    prop[mask_t, t, :] = prop_t

                    # Optional: log pmf at observed A (useful for stabilized weights)
                    if A_obs is not None:
                        a_t = A_obs[mask_t, t]                                 # [n], values 0/1
                        a_flat = a_t.unsqueeze(0).expand(K, -1).reshape(K * n) # [K*n]
                        log_pmf = pred.log_prob(a_flat)                        # [K*n], element-wise
                        log_gps[mask_t, t, :] = log_pmf.view(K, n).T           # [n, K]
                    
                    if sample_count > 0:
                        samp = pred.sample((sample_count,))         # [S, K*n], 0/1 draws
                        A_samples[:, mask_t, t, :] = samp.view(sample_count, K, n).permute(0, 2, 1).to(torch.float32)
                        logpmf = pred.log_prob(samp)                    # [S, K*n]
                        log_gps_samples[:, mask_t, t, :] = logpmf.view(sample_count, K, n).permute(0, 2, 1)
                        

                else:
                    # Generic fallback: try to use returned distribution directly.
                    pred = lik_t(latent)
                    if A_obs is not None and hasattr(pred, "log_prob"):
                        a_t = A_obs[mask_t, t]
                        a_flat = a_t.unsqueeze(0).expand(K, -1).reshape(K * n)
                        lp = pred.log_prob(a_flat)
                        # If lp came back as joint (scalar), skip; otherwise scatter
                        if lp.shape == a_flat.shape:
                            log_gps[mask_t, t, :] = lp.view(K, n).T

                    # If pred has mean/variance (e.g., Poisson gives mean=rate), you can
                    # also record expected value here if desired.

        out = {
            "propensity": prop,  # (N*, T, K) — filled only for Bernoulli
            "mu": mu_out,        # (N*, T, K) — useful for continuous
            "var": var_out,      # (N*, T, K)
        }
        if log_gps is not None:
            out["log_gps"] = log_gps  # (N*, T, K ) 
        if A_samples is not None:
            out["A_samples"] = A_samples  # (S, N*, T, K)
        if log_gps_samples is not None:
            out["log_gps_samples_z_meaned"] = log_gps_samples.mean(3)  # (S, N*, T, ) we average over K which is z_integral and the last axis
        return out

class SeqGPLVMVal(SeqGPLVM):
    """
    Validation-time variant that:
      - copies a trained SeqGPLVM
      - freezes all original params
      - adds a fresh VariationalLatentVariable Z_val for the validation patients
      - re-uses the per-time GP modules & likelihoods
      - provides its own forward() and fit_z_posterior()
    """
    def __init__(self, *args, val_meta=None, **kwargs):
        # Build the training-time pieces first
        super().__init__(*args, **kwargs)
        # If val metadata is present (loading from ckpt), build Z_val & mlls_val now
        if val_meta is not None:
            self._init_validation_components_from_meta(val_meta)

    def _init_validation_components_from_meta(self, val_meta):
        device = next(self.parameters()).device
        N_val, T, Q = int(val_meta["N_val"]), self.T, self.Q

        # Dummy init; weights will be loaded from state_dict
        init_z = torch.zeros(N_val, Q, device=device)
        self.Z_val = VariationalLatentVariable(N_val, T, Q, init_z, self.Z.prior_x)

        # Recreate per-time MLLs using saved counts
        self.mlls_val = []
        for t in range(T):
            n_t = int(val_meta["n_per_t"][t])
            if self.use_titsias:
                mll_t = type(self.mlls[t])(self.likelihoods[t], self.gps[t])
            else:
                mll_t = type(self.mlls[t])(self.likelihoods[t], self.gps[t], num_data=n_t)
            self.mlls_val.append(mll_t)

        # Freeze everything except Z_val
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.Z_val.parameters():
            p.requires_grad_(True)

    # !We construct this *from* an already-trained model!
    @classmethod
    def from_trained(cls, trained_model: SeqGPLVM, X_val: torch.Tensor, Y_val: torch.Tensor, init_z: torch.Tensor = None):
        # 1) deep-copy the trained module so we keep buffers, modules, etc.
        model = deepcopy(trained_model).eval()

        # 2) "upgrade" the copied instance to this subclass
        model.__class__ = cls

        # 3) freeze all original params
        for p in model.parameters():
            p.requires_grad_(False)

        # 4) stash validation data
        model.X_val = X_val
        model.Y_val = Y_val

        # 5) build a fresh variational Z for validation patients
        N_val, T, *_ = X_val.shape
        Q = model.Q
        if init_z is None:
            init_z = torch.randn(N_val, Q, device=X_val.device)

        model.T_val = T
        Z_val_prior = prior_same_type_resized(
            trained_model.Z.prior_x, N_val, Q, device=X_val.device)
        model.Z_val = VariationalLatentVariable(
            N_val, T, Q, init_z, Z_val_prior
        )

        # 6) build per-time MLLs for validation, using the *same* gps/likelihoods
        model.mlls_val = []
        n_per_t = []
        for t in range(T):
            subset = ~(torch.isnan(Y_val[:, t]).reshape(-1).detach().cpu())
            n_t = int(subset.float().sum().item())
            n_per_t.append(n_t)

            if model.use_titsias:
                # ExactMarginalLogLikelihood(lik_t, gp_t)
                mll_t = type(model.mlls[t])(model.likelihoods[t], model.gps[t])
            else:
                # VariationalELBO(lik_t, gp_t, num_data=n_t)
                mll_t = type(model.mlls[t])(model.likelihoods[t], model.gps[t], num_data=n_t)

            model.mlls_val.append(mll_t)
        
        model.val_meta = {"N_val": N_val, "n_per_t": n_per_t}

        return model

    def sample_latent_variable_val(self):
        return self.Z_val()

    def forward(self):
        """
        Validation loss: only Z_val is learnable; GPs/likelihoods are re-used.
        """
        Z_sample = self.sample_latent_variable_val()
        loss = 0.0

        for t, (gp, mll) in enumerate(zip(self.gps, self.mlls_val)): # note that if mmls_val is shorter that self.gps (so we have less time points in val than in train) this loop only itterates over the val time points
            Yt = self.Y_val[:, t]
            Xt = self.X_val[:, t, :]
            subset = ~torch.isnan(Yt).reshape(-1).detach()
            if subset.sum() == 0:
                continue

            yt = Yt[subset]
            zt = Z_sample[subset, :]
            xt = Xt[subset, :]
            Xjoint = torch.cat([xt, zt], dim=1)

            if self.use_titsias:
                gp.set_train_data(inputs=(Xjoint,), targets=yt, strict=False)

            with gpytorch.settings.cholesky_jitter(1e-3):
                f_dist = gp(Xjoint)
                loss += -mll(f_dist, yt).sum()
        
        kl_z_val = sum(term.loss() for term in self.Z_val.added_loss_terms())
        loss += self.T_val * kl_z_val


        return loss