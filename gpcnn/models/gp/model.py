"""
Adapted from https://gpytorch.readthedocs.io/en/latest/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
"""
import math

import gpytorch
import torch


class GPLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64, mixing_params=False, sum_output=False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size,
            batch_size=num_dim
        )
        variational_strategy = gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
            self,
            grid_size=grid_size,
            grid_bounds=[grid_bounds],
            num_dim=num_dim,
            variational_distribution=variational_distribution,
            mixing_params=mixing_params,
            sum_output=sum_output,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
