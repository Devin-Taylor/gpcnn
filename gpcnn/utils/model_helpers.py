import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gpcnn.utils import mkdir, rmdir


class Checkpointer(object):

    def __init__(self, model: nn.Module, optimizer: optim, model_path: str):
        self.model_path = model_path
        self.model = model
        self.optimizer = optimizer

    def restore(self, use_checkpoint: bool, checkpoint_number: int = None):
        if use_checkpoint:
            if checkpoint_number is None:
                checkpoint_path = self._get_latest_checkpoint()
            else:
                checkpoint_path = os.path.join(self.model_path, f'ckpt_{checkpoint_number}.pt')

            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']

            print(f"Loaded model from checkpoint: {checkpoint_path}")
        else:
            start_epoch = 0
            rmdir(self.model_path)
            mkdir(self.model_path)
        return start_epoch

    def save(self, epoch: int):
        fn = os.path.join(self.model_path, f'ckpt_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, fn)
        print(f"Saving checkpoint: {fn}")

    def _get_latest_checkpoint(self):
        files = glob(os.path.join(self.model_path, "*.pt"))
        if not files:
            raise Exception(f"No models found in path '{self.model_path}'")
        epochs = [x.split('.')[0].split('_')[-1] for x in files]
        return files[np.argmax(epochs)]
