import logging
import os
import torch
from torch import optim
from Pinnoloc.utils.experiments import set_seed
from torch.utils.data import DataLoader
from Pinnoloc.utils.split_data import random_split_dataset
from Pinnoloc.utils.printing import print_num_trainable_params, print_parameters
from Pinnoloc.models.vector import StackedVectorModel, PositionModel
from Pinnoloc.ml.optimization import setup_optimizer
from Pinnoloc.ml.loss import PositionLoss
from Pinnoloc.ml.training import TrainPhysicsModel
from Pinnoloc.ml.evaluation import EvaluateRegressor
from Pinnoloc.ml.selection import random_search
from Pinnoloc.data.preprocessing import compute_mean_std, StandardizeDataset
from Pinnoloc.utils.saving import load_data
from Pinnoloc.utils.check_device import check_model_device
from Pinnoloc.utils.experiments import read_yaml_to_dict, save_dict_to_yaml
from Pinnoloc.utils.saving import save_data
from run_ble_position import run_ble_position
import logging
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import csv


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run BLE Position Static')
    parser.add_argument('--seed_search', type=int, help='Random seed for hyperparameter search', default=42)
    parser.add_argument('--seed_run', type=int, help='Random seed for model run', default=42)
    parser.add_argument('--device', type=str, help='The device to run the model', default='cpu')
    parser.add_argument('--n_configs', type=int, help='Number of configurations to generate', required=True)
    parser.add_argument('--develop', type=str, help='Choose the dataset to develop', required=True,
                        choices=['calibration',
                                 'static_east', 'static_north', 'static_south', 'static_west',
                                 'static_all'])
    parser.add_argument('--test', type=str, help='Choose the dataset to test', required=True,
                        choices=['calibration',
                                 'static_east', 'static_north', 'static_south', 'static_west',
                                 'static_all',
                                 'mobility_use-case1_run1', 'mobility_use-case1_run2', 'mobility_use-case1_run3', 'mobility_use-case1_run4',
                                 'mobility_use-case2_run1', 'mobility_use-case2_run2', 'mobility_use-case2_run3', 'mobility_use-case2_run4',
                                 'mobility_use-case3_run1', 'mobility_use-case3_run2', 'mobility_use-case3_run3', 'mobility_use-case3_run4'])

    # If and only if develop is distinct from test, require the test_split argument
    args, _ = parser.parse_known_args()
    if args.develop == args.test:
        parser.add_argument('--test_split', type=float, help='The test split ratio', required=True)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    original_log_level = logging.getLogger().getEffectiveLevel()

    args = parse_args()
    seed_search = args.seed_search
    seed_run = args.seed_run
    n_configs = args.n_configs
    device = args.device
    develop = args.develop
    test = args.test

    hyperparameters = {
        'n_layers': [2],
        'hidden_units': [[8]],
        'batch_size': [256],
        'lr': [0.1],
        'weight_decay': [0.01],
        'val_split': [0.2],
        'patience': [10],
        'reduce_plateau': [0.1],
        'num_epochs': [500],
        'lambda_data': (0.0, 1.0),
        'lambda_rss': (0.0, 1.0),
        'lambda_azimuth': (0.0, 1.0),
        # 'lambda_bc': (0.0, 1.0),
        'lambda_bc': [0.0],
        'n_collocation': [20000],
        'n_boundary_collocation': [256],
        'resampling_period': [10]
    }

    logging.info(f"Setting seed for search: {seed_search}")
    set_seed(seed_search)

    logging.info("Random search for hyperparameters.")
    hyperparameters_list = random_search(hyperparameters=hyperparameters, n_configs=n_configs)

    logging.info(f"Setting seed for run: {seed_run}")
    set_seed(seed_run)

    task_name_develop = f'ble_position_{develop}'
    task_name_test = f'ble_position_{test}'

    logging.info(f'Loading {task_name_develop} develop and {task_name_test} test datasets.')
    try:
        if develop == test:
            test_split = args.test_split
            develop_split = 1.0 - test_split
            full_dataset = load_data(os.path.join('datasets', task_name_develop, 'full_dataset'))
            develop_dataset, test_dataset = random_split_dataset(full_dataset, val_split=args.test_split)
        else:
            develop_split = 1.0
            test_split = 1.0
            develop_dataset = load_data(os.path.join('datasets', task_name_develop, 'full_dataset'))
            test_dataset = load_data(os.path.join('datasets', task_name_test, 'full_dataset'))
        develop_args = read_yaml_to_dict(os.path.join('datasets', task_name_develop, 'dataset_args.yaml'))
        develop_args = {f'develop_{key}': value for key, value in develop_args.items()}
        develop_args['develop_split'] = develop_split
        test_args = read_yaml_to_dict(os.path.join('datasets', task_name_test, 'dataset_args.yaml'))
        test_args = {f'test_{key}': value for key, value in test_args.items()}
        test_args['test_split'] = test_split
    except FileNotFoundError:
        logging.error(f"Dataset not found for {task_name_develop} develop and/or {task_name_test} test.")

    # create a csv file to save the hyperparameters and scores
    file_path = os.path.join('results', f'ble_position_{develop}_{test}.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open the file in append model
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is not created or empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(['seed_search', 'seed_run'] + list(hyperparameters.keys()) +
                            ['test_mse', 'test_rmse', 'test_50th', 'test_75th', 'test_90th', 'test_mae', 'test_min_ae', 'test_max_ae'] +
                            list(develop_args.keys()) + list(test_args.keys()))

    # Set the log level to a higher level, e.g., WARNING or CRITICAL
    logging.disable(logging.CRITICAL)
    # Run your tests here
    for hyperparameters in tqdm(hyperparameters_list):
        scores = run_ble_position(seed_run, device, develop_dataset, test_dataset, hyperparameters)
        # save hyperparameters and scores in a new row of a CSV file
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([seed_search, seed_run] +  list(hyperparameters.values())
                            + list(scores.values()) +
                            list(develop_args.values()) + list(test_args.values()))
     
    # Restore the original log level after the tests
    logging.disable(original_log_level)


if __name__ == "__main__":
    main()
    