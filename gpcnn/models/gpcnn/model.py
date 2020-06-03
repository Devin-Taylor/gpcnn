import gpytorch
from gpcnn.models.gp.model import GPLayer

class GPCNNModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, num_classes, num_data, grid_bounds=(-10., 10.)):
        super(GPCNNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_dim, num_classes=num_classes)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_layer, num_data=num_data)

    def forward(self, x):
        _, features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
