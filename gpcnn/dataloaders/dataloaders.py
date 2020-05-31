import math

import torch
from torchvision import datasets, transforms


def mnist_dataloader(fmt: str, data_path: str, batch_size: int, validation_split: float = None, **kwargs):
    if fmt == 'train':
        dataset = datasets.MNIST(data_path, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        validation_size = int(math.floor(validation_split * len(dataset)))
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset)-validation_size, validation_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        return train_loader, valid_loader
    elif fmt == 'test':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        return test_loader
    else:
        raise Exception(f"Unknown data format of type '{fmt}'")
