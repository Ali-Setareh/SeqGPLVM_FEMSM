from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.priors import NormalPrior, GammaPrior
from gpytorch.models.gplvm import  VariationalLatentVariable
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from torch.distributions import Distribution
import gpytorch
import torch 
import torch.nn as nn
from models.GPLVM import GPLVM, SGPRModel
from copy import deepcopy
from tqdm.notebook import trange
from typing import Optional

class SeqGPLVM(nn.Module):
    '''
    dim(Y) = N x T
    dim(X_cov) = N x T x (dim_cov + 1) (dim_cov for covariates and 1 for last treatment)
    dim(Z) = latent_dim
    '''
    def __init__(
        self,
        *,
        N: int, # number of patients
        T: int, # number of time points
        C: int, # number of covariates
        latent_dim: int, # latent dimension Q
        n_inducing_x: int, # number of inducing points in the covariate space
        n_inducing_hidden: int, # number of inducing points in the hidden (latent) space
        init_z: torch.Tensor = None, # optional initial Z if none  the model will assign them N(0,1)
        device = None, 
        lik = None, #gpytorch.likelihoods.GaussianLikelihood()
        learn_inducing_locations = False , 
        use_titsias = False 
    ):
        """
        Y: (N, T) tensor of targets for each of T GPs
        X_cov: (N, T, C) tensor of covariates per data and task
        latent_dim: Q
        n_inducing: number of inducing points per GP
        init_z: (N, Q) init for shared latent; if None, random
        """
        super().__init__()

        self.config = {
            'N': N,
            'T': T,
            'C': C,
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
        self.N, self.T, self.C, self.Q = N, T, C, latent_dim
        self.lik = lik

        # LatentVariable
        if init_z is None:
             init_z = torch.nn.Parameter(torch.randn(N, latent_dim))

        # Define prior for Z
        Z_prior_mean = torch.zeros(N, latent_dim, requires_grad=False, device=device)  # shape: N x Q
        prior_Z = NormalPrior(Z_prior_mean, 0.5*torch.ones_like(Z_prior_mean, requires_grad=False, device=device)) #requires_grad=False because we dont want to learn the prior hyperparams 

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
        
        Xu_Z  = torch.randn(n_inducing_hidden, latent_dim).to(device)

        for t in range(T):
            ### NEW ###
            mask  = ~torch.isnan(X_cov[:,t,:]).any(dim=1)
            covs = X_cov[:,t,:][mask]
            perm = torch.randperm(covs.size(0))[:n_inducing_x]
            # choose some of the non nan Xs to be the inducing points. 
            Xu_X  = covs [perm].to(device)
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
        #Z_sample_batch = Z_sample[batch_idx]            # (B, Q)
        #Xcov_batch = self.X_cov[batch_idx]                  # (B, T, C)
        #losses = []
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
    

def val_model(trained_model, X_val, Y_val):

        """
        Returns a *deterministic* copy of the trained SeqGPLVM in which
        only the variational parameters of Z are left unfrozen and
        initialised for the N_val validation patients.
        """
        
        model = deepcopy(trained_model).eval()          # freeze weights
        for p in model.parameters():                    # global params
            p.requires_grad_(False)

        # ── create a brand-new VariationalLatentVariable ─────────────────
        N_val, T, *_ = X_val.shape
        Q            = model.Q
        init_z       = torch.randn(N_val, Q, device=X_val.device)

        model.X_val = X_val
        model.Y_val = Y_val
      
        model.Z_val  = VariationalLatentVariable(
                        N_val, T, Q, init_z, model.Z.prior_x
                    )
     
        model.mlls_val = [] 
        for t in range(T): 
            subset = ~(torch.isnan(Y_val[:,t]).reshape(-1).detach().cpu())
            if model.use_titsias: 
                model.mlls_val.append(type(model.mlls[t])(   # grabs the *original class*
                            model.likelihoods[t],            # same likelihood object
                            model.gps[t]                     # the per-step GP module
                        ))
            else: 
                model.mlls_val.append(type(model.mlls[t])(       # grabs the *original class*
                                model.likelihoods[t],            # same likelihood object
                                model.gps[t],                    # the per-step GP module
                                num_data=(subset.float().sum())  # only that column’s targets
                            ))

        
        # the only learnable params will be those of Z_val
        return model

def forward_val(model):      #, batch_idx: torch.Tensor):
        # Sample shared latent for batch
        Z_sample = model.Z_val()  # q(Z|·) sample
        loss = 0.0
        
        for t, (gp, mll) in enumerate(zip(model.gps, model.mlls_val)):
            Yt = model.Y_val[:, t]
            Xt = model.X_val[:,t,:]
            subset = ~torch.isnan(Yt).reshape(-1).detach()
            if subset.sum() > 0:
                yt = Yt[subset]
                zt = Z_sample[subset, :]
                xt = Xt[subset, :]

                Xjoint = torch.cat([xt,zt], dim=1)    # (B, Q+C)
                
                if model.use_titsias: 
                    gp.set_train_data(
                        inputs=(Xjoint,),
                        targets=yt,
                        strict=False,           # allow changing the size of the data
                        )
                
                with gpytorch.settings.cholesky_jitter(1e-3):
                    f_dist = gp(Xjoint)
                    elbo = -mll(f_dist, yt).sum()
                    loss+= elbo 
        
        return loss  



from utils.inspectors import get_actuals_via_getters

def fit_Z_posterior(model, steps=500, lr=0.01):
    keywords = ["chol_variational_covar", "variational_mean"] # these two are too big to save so we ommit them during the book keeping
    param_hist = {name: [] for name, _ in model.named_parameters() if not any(kw in name for kw in keywords) }


    iterator = trange(steps, leave=True)

    opt  = torch.optim.Adam(model.Z_val.parameters(), lr=lr)
    loss_list = []
    model.train()
    for i in iterator:
        opt.zero_grad()             
        loss = forward_val(model)        
        loss.backward()
        opt.step()
        iterator.set_description(f"Loss: {loss:.4f}, iter {i}")
        for name, p in model.named_parameters():
                if not any(kw in name for kw in keywords):
                    param_hist[name].append(p.data.clone().detach().cpu().numpy())
        
        loss_list.append(loss.item())



        real_params = get_actuals_via_getters(model)
    model.eval()

    return param_hist,real_params,loss_list



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

        model.Z_val = VariationalLatentVariable(
            N_val, T, Q, init_z, model.Z.prior_x  # reuse prior from the trained model
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

        for t, (gp, mll) in enumerate(zip(self.gps, self.mlls_val)):
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

        return loss