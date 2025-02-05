import torch
import os
import logging
import argparse
from torchvision import datasets, transforms
from Pinnoloc.utils.experiments import set_seed
from Pinnoloc.torch_dataset.sequantial_image import SequentialImage2Classify
from Pinnoloc.utils.saving import save_data

tasks = ['smnist']

parser = argparse.ArgumentParser(description='Build Classification task.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--task', required=True, choices=tasks, help='Name of classification task.')

args, unknown = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
args = parser.parse_args()

logging.info(f"Setting seed: {args.seed}")
set_seed(args.seed)

logging.info(f"Building task: {args.task}")

if args.task == 'smnist':
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to pytorch tensor with values in [0, 1] and shape (C, H, W)
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])
    develop_dataset = SequentialImage2Classify(datasets.MNIST(root='data_storage/',
                                                              train=True,
                                                              transform=transform,
                                                              download=True))
    test_dataset = SequentialImage2Classify(datasets.MNIST(root='data_storage/',
                                                           train=False,
                                                           transform=transform,
                                                           download=True))
else:
    raise ValueError('Task not found')

logging.info('Saving datasets')
# save data in Pinnoloc/datasets/task_name, now I'm in Pinnoloc/
save_data(develop_dataset, os.path.join('datasets', args.task, 'develop_dataset'))
save_data(test_dataset, os.path.join('datasets', args.task, 'test_dataset'))
