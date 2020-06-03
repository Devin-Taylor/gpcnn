import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """
    Taken from: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        output = F.relu(features)
        output = self.dropout2(output)
        output = self.fc2(output)
        return output, features
