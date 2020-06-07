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
        return {
            "loss" : acc_loss / (batch_idx + 1)
        }

    def validation_step(self, valid_loader, device):
        self.model.eval()
        self.model.likelihood.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                model_output = self.model(data)
                output = self.model.likelihood(model_output)
                pred = output.probs.mean(0).argmax(1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                valid_loss += -self.model.mll(model_output, target).item()
        return {
            'loss' : valid_loss / (batch_idx + 1),
            'accuracy' : int(correct) / float(len(valid_loader.dataset))
        }

    def test_step(self, test_loader, device):
        self.model.eval()
        self.model.likelihood.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                model_output = self.model(data)
                output = self.model.likelihood(model_output)
                pred = output.probs.mean(0).argmax(1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                test_loss += -self.model.mll(model_output, target).item()
        return {
            'loss' : test_loss / (batch_idx + 1),
            'accuracy' : int(correct) / float(len(test_loader.dataset))
        }
