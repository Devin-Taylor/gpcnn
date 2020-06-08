import abc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gpcnn.utils.model_helpers import Checkpointer


class Trainer(metaclass=abc.ABCMeta):
    def __init__(self, model: nn.Module, optimizer: optim, save_every_n: int, model_path: str, use_checkpoint: bool = False, checkpoint_number: int = None):
        self.model = model
        self.optimizer = optimizer
        self.save_every_n = save_every_n
        self.use_checkpoint = use_checkpoint
        self.checkpoint_number = checkpoint_number

        self.checkpointer = Checkpointer(
            model=model,
            optimizer=optimizer,
            model_path=model_path
        )

        self.model_path = model_path

    def __call__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, num_epochs: int, device: torch.device):

        start_epoch = self.checkpointer.restore(use_checkpoint=self.use_checkpoint, checkpoint_number=self.checkpoint_number)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_metrics = self.train_step(train_loader, epoch, device)
            valid_metrics = self.validation_step(valid_loader, device)
            test_metrics = self.test_step(test_loader, device)
            print(f"Epoch: {epoch} train: {train_metrics} validation: {valid_metrics} test: {test_metrics}")

            if ((epoch + 1) % self.save_every_n) == 0:
                self.checkpointer.save(epoch=epoch)

    @abc.abstractmethod
    def train_step(self, train_loader: DataLoader, epoch: int, device: torch.device):
        pass

    @abc.abstractmethod
    def validation_step(self, valid_loader: DataLoader, device: torch.device):
        pass

    @abc.abstractmethod
    def test_step(self, test_loader: DataLoader, device: torch.device):
        pass
