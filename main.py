import argparse
import math
import os
import sys

import torch
import torch.optim as optim
import yaml
from torchvision import datasets, transforms

from gpcnn.dataloaders import mnist_dataloader
from gpcnn.models.gpcnn.model import GPCNNModel
from gpcnn.models.gpcnn.trainer import GPCNNTrainer
from gpcnn.models.mnist.model import MNISTModel
from gpcnn.models.mnist.trainer import MNISTTrainer
from gpcnn.utils import mkdir
from gpcnn.utils.model_helpers import Checkpointer
from evaluation import evaluate

GENERATOR_MODELS = ['mnist']
MODELS = GENERATOR_MODELS + ['gpcnn']
RESULTS_ROOT = 'results'
DATA_ROOT = 'data'
MODELS_ROOT = 'models'

def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser(description='GP CNN')

    parser.add_argument('-e', '--experiment', type=str, help='name of experiment', required=True)
    parser.add_argument('-m', '--model', type=str, help='which model to use', choices=MODELS, required=True)
    parser.add_argument('-g', '--generator', type=str, help='which feature generator model to use', choices=GENERATOR_MODELS, required=False)

    parser.add_argument('--use-checkpoint', action='store_true', default=False, help='train from checkpoints')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--evaluate', action='store_true', default=False, help='perform uncertainty evaluations')

    parsed_args = parser.parse_args(args_to_parse)
    return parsed_args

def get_settings(file_path):
    with open(file_path) as fd:
        return yaml.safe_load(fd)



def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    mkdir(RESULTS_ROOT)
    mkdir(DATA_ROOT)
    mkdir(MODELS_ROOT)

    results_path = os.path.join(RESULTS_ROOT, args.experiment)
    data_path = os.path.join(DATA_ROOT, args.model)
    settings_path = os.path.join(f'gpcnn/models/{args.model}/settings.yaml')
    model_path = os.path.join(MODELS_ROOT, args.model)

    mkdir(results_path)
    mkdir(data_path)
    mkdir(model_path)

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

        trainer = MNISTTrainer(
            model=model,
            optimizer=optimizer,
            save_every_n=params['train']['save_every_n'],
            model_path=model_path,
            use_checkpoint=args.use_checkpoint
        )
    elif args.model == 'gpcnn':
        if args.generator not in GENERATOR_MODELS:
            raise Exception(f"Unsupported feature geneator model of type '{args.generator}'")

        if args.generator == 'mnist':
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

            gen_model = MNISTModel().to(device)
            gen_optimizer = optim.Adadelta(gen_model.parameters(), lr=params['train']['lr'])

            checkpointer = Checkpointer(
                model=gen_model,
                optimizer=gen_optimizer,
                model_path=os.path.join(MODELS_ROOT, args.generator)
            )
            checkpointer.restore(use_checkpoint=True)

        gen_model.eval() # NOTE only used for feature generation
        model = GPCNNModel(
            feature_extractor=gen_model,
            num_dim=params['num_features'],
            num_classes=params['num_classes'],
            num_data=len(train_loader.dataset)
        ).to(device)

        optimizer = optim.Adam([
            {'params': model.gp_layer.hyperparameters(), 'lr': params['train']['lr'] * 0.01},
            {'params': model.gp_layer.variational_parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=params['train']['lr'], weight_decay=0)

        trainer = GPCNNTrainer(
            model=model,
            optimizer=optimizer,
            save_every_n=params['train']['save_every_n'],
            model_path=model_path,
            use_checkpoint=args.use_checkpoint
        )
    else:
        raise Exception(f'Unsupported model of type {args.model}')

    if args.evaluate and args.model != 'gpcnn':
        raise Exception(f"Evaluation for '{args.model}' is currently not supported")
    elif args.evaluate:
        checkpointer = Checkpointer(
            model=model,
            optimizer=optimizer,
            model_path=os.path.join(MODELS_ROOT, args.model)
        )
        checkpointer.restore(use_checkpoint=True)

        evaluate(model, test_loader, device, n_classes=params['num_classes'], results_dir=results_path)
    else:
        trainer(train_loader, valid_loader, test_loader, params['train']['epochs'], device)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
