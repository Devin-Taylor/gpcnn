import abc
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from gpcnn.utils import mkdir, rmdir


class Trainer(metaclass=abc.ABCMeta):
    def __init__(self, model: nn.Module, optimizer: optim, save_every_n: int, model_path: str, use_checkpoint: bool = False, checkpoint_number: int = None):
        self.model = model
        self.optimizer = optimizer
        self.save_every_n = save_every_n
        self.use_checkpoint = use_checkpoint
        self.checkpoint_number = checkpoint_number

        self.model_path = model_path

    def __call__(self, train_loader, valid_loader, test_loader, num_epochs, device):

        if self.use_checkpoint:
            if self.checkpoint_number is None:
                checkpoint_path = self._get_latest_checkpoint()
            else:
                checkpoint_path = os.path.join(self.model_path, f'ckpt_{self.checkpoint_number}.pt')

            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']

            print(f"Loaded model from checkpoint: {checkpoint_path}")
        else:
            start_epoch = 0
            rmdir(self.model_path)
            mkdir(self.model_path)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_metrics = self.train_step(train_loader, epoch, device)
            valid_metrics = self.train_step(valid_loader, epoch, device)
            test_metrics = self.train_step(test_loader, epoch, device)
            print(f"Epoch: {epoch} train: {train_metrics} validation: {valid_metrics} test: {test_metrics}")

            if ((epoch + 1) % self.save_every_n) == 0:
                fn = os.path.join(self.model_path, f'ckpt_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, fn)
                print(f"Saving checkpoint: {fn}")

    def _get_latest_checkpoint(self):
        files = glob(os.path.join(self.model_path, "*.pt"))
        if len(files) == 0:
            raise Exception(f"No models found in path '{self.model_path}'")
        epochs = [x.split('.')[0].split('_')[-1] for x in files]
        return files[np.argmax(epochs)]

    @abc.abstractmethod
    def train_step(self, train_loader, epoch, device):
        pass

    @abc.abstractmethod
    def validation_step(self, valid_loader, epoch, device):
        pass

    @abc.abstractmethod
    def test_step(self, test_loader, epoch, device):
        pass

class MNISTTrainer(Trainer):
    def __init__(self, model: nn.Module, optimizer: optim, save_every_n: int, model_path: str, use_checkpoint: bool = False, checkpoint_number: int = None, show_progress_bar: bool = True):
        super().__init__(model, optimizer, save_every_n, model_path, use_checkpoint, checkpoint_number)
        self.show_progress_bar = show_progress_bar

    def train_step(self, train_loader, epoch_num, device):
        self.model.train()
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
                loss = self.model.loss(output, target)
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
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                valid_loss += self.model.loss(output, target).item()

        valid_loss /= (batch_idx + 1)
        return valid_loss

    def test_step(self, test_loader, device):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += self.model.loss(output, target).item()

        test_loss /= (batch_idx + 1)
        return test_loss
