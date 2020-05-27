import argparse
import os
import sys
import math
import torch
import torch.optim as optim
import yaml
from torchvision import datasets, transforms

from gpcnn.models.mnist.model import MNISTModel
from gpcnn.models.mnist.trainer import MNISTTrainer

MODELS = ['mnist']
RESULTS_ROOT = 'results'
DATA_ROOT = 'data'

def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser(description='GP CNN')

    parser.add_argument('-e', '--experiment', type=str, help='name of experiment', required=True)
    parser.add_argument('-m', '--model', type=str, help='which model to use', choices=MODELS, required=True)
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')

    parsed_args = parser.parse_args(args_to_parse)
    return parsed_args

def get_settings(file_path):
    with open(file_path) as fd:
        return yaml.safe_load(fd)

def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    results_path = os.path.join(RESULTS_ROOT, args.experiment)
    data_path = os.path.join(DATA_ROOT, args.model)
    settings_path = os.path.join(f'gpcnn/models/{args.model}/settings.yaml')

    params = get_settings(settings_path)

    if args.model == 'mnist':
        dataset = datasets.MNIST(data_path, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        validation_size = int(math.floor(params['train']['validation_split'] * len(dataset)))
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset)-validation_size, validation_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params['train']['batch_size'],
            shuffle=True,
            **kwargs
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=params['train']['batch_size'],
            shuffle=True,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=params['test']['batch_size'], shuffle=True, **kwargs)

        model = MNISTModel().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=params['train']['lr'])

        trainer = MNISTTrainer(model, optimizer)
        trainer(train_loader, valid_loader, test_loader, params['train']['epochs'], device)
    else:
        raise Exception(f'Unsupported model of type {args.model}')

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
