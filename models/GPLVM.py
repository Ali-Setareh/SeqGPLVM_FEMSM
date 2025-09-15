from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.priors import GammaPrior


from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution


from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, LinearKernel
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal

from torch.distributions import Gamma

import torch 
import numpy as np 
from helpers import grid_helper

class GPLVM(ApproximateGP):
    def __init__(self, n, x_inducing, z_inducing, learn_inducing_locations = False, kernel = "RBF"):
        self.n = n
        #self.batch_shape = torch.Size([data_dim])

        # Locations of inducing points in both covariate and latent space 
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.inducing_inputs = torch.cat([self.x_u,self.z_u], dim = 1)
        self.n_inducing = self.inducing_inputs.shape[0]
        #self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(self.n_inducing) #, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u,
                                   learn_inducing_locations=learn_inducing_locations)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        #super().__init__(Z, q_f)
        super().__init__(q_f)



        # Kernel (acting on latent dimensions)
        #self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        if kernel=="RBF": 
            lengthscale_prior = GammaPrior(10.0, 1.0) #GammaPrior(10.0, 1.0) # if you use GammaPrior you can also learn the hyperprameters of the prior but for now we skip it 
            outputscale_prior = GammaPrior(1.0, 1.0) #GammaPrior(1.0, 1.0)

            self.mean_module = ConstantMean(ard_num_dims= self.inducing_inputs.shape[1])
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.inducing_inputs.shape[1],
                                                    lengthscale_prior = lengthscale_prior),
                                            outputscale_prior=outputscale_prior)
            
            #self.covar_module.base_kernel.register_constraint("raw_lengthscale", GreaterThan(1e-1))
            # Initialize lengthscale and outputscale to mean of priors
            self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean #paper code : 2
            self.covar_module.outputscale = outputscale_prior.mean #paper code: 0.7  
        elif kernel=="linear": 
            variance_prior = GammaPrior(1.0, 1.0)
            self.mean_module = ConstantMean(ard_num_dims= self.inducing_inputs.shape[1])
            self.covar_module = LinearKernel(ard_num_dims=self.inducing_inputs.shape[1],lengthscale_prior = lengthscale_prior,variance_prior=variance_prior)



        #for name, param in self.covar_module.named_parameters():
            # any parameter that belongs to a Prior or Constraint we don’t want to learn
         #   if "prior" in name or "constraint" in name:
          #      param.requires_grad_(False)

        

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
    

class SGPRModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_inducing, z_inducing, mean_init=None):
        self.train_x = train_x
        self.trian_y = train_y 
        super().__init__(self.train_x, self.trian_y, likelihood)
        
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.inducing_inputs = torch.cat([self.x_u,self.z_u], dim = 1)

        self.mean_module = ConstantMean(ard_num_dims= self.inducing_inputs.shape[1])

        if mean_init is not None:
            self.mean_module.constant.data.fill_(mean_init)

        base_covar = ScaleKernel(
            RBFKernel(ard_num_dims=train_x.size(-1))
        )

        lengthscale_prior = GammaPrior(10.0, 1.0) #GammaPrior(10.0, 1.0) # if you use GammaPrior you can also learn the hyperprameters of the prior but for now we skip it 
        outputscale_prior = GammaPrior(1.0, 1.0) #GammaPrior(1.0, 1.0)

        
        base_covar = ScaleKernel(RBFKernel(ard_num_dims=self.inducing_inputs.shape[1],
                                                   lengthscale_prior = lengthscale_prior),
                                        outputscale_prior=outputscale_prior)
        
        
        self.covar_module = InducingPointKernel(
            base_covar,
            inducing_points= self.inducing_inputs,
            likelihood=likelihood
        )

        self.covar_module.base_kernel.base_kernel.lengthscale = 2#lengthscale_prior.mean
        self.covar_module.base_kernel.outputscale = 0.7#outputscale_prior.mean

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)