from typing import Tuple

import gpytorch
import torch
import torch.nn as nn

from gpcnn.models.gp.model import GPLayer


class GPCNNModel(gpytorch.Module):
    def __init__(self, feature_extractor: nn.Module, num_dim: int, num_classes: int, num_data: int, grid_bounds: Tuple[int, int] = (-10., 10.)):
        super(GPCNNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPLayer(
            num_dim=num_dim,
            grid_bounds=grid_bounds,
            grid_size=num_dim
        )
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_features=num_dim,
            num_classes=num_classes
        )
        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self.gp_layer,
            num_data=num_data
        )

    def forward(self, x: torch.tensor):
        _, features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(
            features,
            self.grid_bounds[0],
            self.grid_bounds[1]
        )
        res = self.gp_layer(features)
        return res
