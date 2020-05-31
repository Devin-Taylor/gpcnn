import argparse
import math
import os
import sys

import torch
import torch.optim as optim
import yaml
from torchvision import datasets, transforms

from gpcnn.dataloaders import mnist_dataloader
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
        train_loader, valid_loader = mnist_dataloader(
            fmt='train',
            data_path=data_path,
            batch_size=params['train']['batch_size'],
            validation_split=params['train']['validation_split'],
            **kwargs
        )
        test_loader = mnist_dataloader(
            fmt='test',
            data_path=data_path,
            batch_size=params['test']['batch_size'],
            **kwargs
        )

        model = MNISTModel().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=params['train']['lr'])

        trainer = MNISTTrainer(model, optimizer)
        trainer(train_loader, valid_loader, test_loader, params['train']['epochs'], device)
    else:
        raise Exception(f'Unsupported model of type {args.model}')

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
