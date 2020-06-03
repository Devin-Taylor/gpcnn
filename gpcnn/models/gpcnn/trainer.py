import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from gpcnn.trainers import Trainer


class GPCNNTrainer(Trainer):
    def __init__(self, model: nn.Module, optimizer: optim, save_every_n: int, model_path: str, use_checkpoint: bool = False, checkpoint_number: int = None, show_progress_bar: bool = True):
        super().__init__(model, optimizer, save_every_n, model_path, use_checkpoint, checkpoint_number)
        self.show_progress_bar = show_progress_bar

    def train_step(self, train_loader, epoch_num, device):
        self.model.train()
        self.model.likelihood.train()
        acc_loss = 0

        kwargs = dict(
            desc="Epoch {}".format(epoch_num),
            leave=False,
            disable=not self.show_progress_bar
        )

        with trange(len(train_loader), **kwargs) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -self.model.mll(output, target)
                loss.backward()
                self.optimizer.step()

                t.set_postfix(loss=np.asscalar(loss))
                t.update()

                acc_loss += np.asscalar(loss)
                break
        return {
            "loss" : acc_loss / (batch_idx + 1)
        }

    def validation_step(self, valid_loader, device):
        self.model.eval()
        self.model.likelihood.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                valid_loss += -self.model.mll(output, target).item()
                break
        return {
            'loss' : valid_loss / (batch_idx + 1)
        }

    def test_step(self, test_loader, device):
        self.model.eval()
        self.model.likelihood.eval()
        test_loss = 0
        test_mean = 0
        test_var = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += -self.model.mll(output, target).item()
                test_mean += output.mean
                test_var += output.variance
        return {
            'loss' : test_loss / (batch_idx + 1)
            # 'mean' : test_mean / (batch_idx + 1),
            # 'variance' : test_var / (batch_idx + 1)
        }
