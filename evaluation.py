import os

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from gpcnn.utils import mkdir, rmdir
from gpcnn.utils.evaluation_helpers import (intraclass_variance,
                                            intraclass_viariance_distribution,
                                            softmax_forward_sampling)


def _create_subfolder(results_dir: str, subfolder: str):
    path = os.path.join(results_dir, subfolder)
    rmdir(path)
    mkdir(path)
    return path

def evaluate(model: gpytorch.Module, test_loader: DataLoader, device: torch.device, n_classes: int, results_dir: str, n_samples: int = 100):
    model.eval()
    model.likelihood.eval()
    all_means = []
    all_std = []
    labels = []
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            model_output = model(data)
            output, std_dev = softmax_forward_sampling(model_output, model.likelihood.mixing_weights, n_samples=n_samples)
            all_means += list(output.probs.cpu().numpy())
            all_std += list(std_dev.cpu().numpy())
            labels += list(target.cpu().numpy())
    all_means = np.array(all_means)
    all_std = np.array(all_std)
    labels = np.array(labels)

    distribution_path = _create_subfolder(results_dir, 'distributions')
    variance_path = _create_subfolder(results_dir, 'variances')

    for idx in range(n_classes):
        intraclass_viariance_distribution(means=all_means, stds=all_std, true_y=labels, class_id=idx, results_dir=distribution_path)
        intraclass_variance(means=all_means, stds=all_std, true_y=labels, class_id=idx, results_dir=variance_path)
