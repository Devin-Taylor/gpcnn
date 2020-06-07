import os

import matplotlib.pyplot as plt
import torch


def softmax_forward_sampling(latent_func, mixing_weights, n_samples=10, num_features=128, n_classes=10):
    samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
    if samples.dim() == 2:
        samples = samples.unsqueeze(-1).transpose(-2, -1)
    samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
    if samples.ndimension() != 3:
        raise RuntimeError("f should have 3 dimensions: features x data x samples")
    num_features, n_data, _ = samples.size()
    if num_features != num_features:
        raise RuntimeError("There should be %d features" % num_features)
    mixed_fs = mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
    softmax = torch.nn.functional.softmax(mixed_fs.t(), 1).view(n_data, n_samples, n_classes)
    return torch.distributions.Categorical(probs=softmax.mean(1)), softmax.std(dim=1)

def intraclass_viariance_distribution(means, stds, class_id, results_dir):
    img_name = os.path.join(results_dir, f"class_{class_id}.png")

    preds = means.argmax(axis=1)
    correct_idx = preds == class_id
    correct_stds = stds[correct_idx, preds[correct_idx]]
    incorrect_stds = stds[~correct_idx, preds[~correct_idx]]

    fig = plt.figure()
    plt.hist(correct_stds, alpha=0.75, bins=20)
    plt.hist(incorrect_stds, alpha=0.75, bins=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Standard Deviation", fontsize=12)
    plt.ylabel("Freqency", fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.legend(['Correct', 'Incorrect'], fontsize=12)
    plt.savefig(img_name, format="png", bbox_inches="tight")
    plt.close(fig)

def intraclass_variance(means, stds, true_y, class_id, results_dir):
    img_name = os.path.join(results_dir, f"class_{class_id}.png")

    means_filt = means[true_y == class_id, class_id]
    stds_filt = stds[true_y == class_id, class_id]

    order_idx = means_filt.argsort()
    means_filt = means_filt[order_idx]
    stds_filt = stds_filt[order_idx]

    fig = plt.figure()
    plt.plot(means_filt)
    plt.fill_between(range(len(means_filt)), means_filt - 2*stds_filt, means_filt + 2*stds_filt, alpha=.2, color="red")
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(["mean", "variance"], fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(img_name, format="png", bbox_inches="tight")
    plt.close(fig)
