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
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import csv


cardinal_directions = {
    'east': (1.0, 0.0),
    'north': (0.0, 1.0),
    'south': (0.0, -1.0),
    'west': (-1.0, 0.0)
}


anchor_positions = {
    '6501': (0.0, 3.0),
    '6502': (6.0, 0.0),
    '6503': (12.0, 3.0),
    '6504': (6.0, 6.0)
}


def path_loss_extimation(dataloader, anchor_x, anchor_y):
    pa_list = []
    xy_list = []
    for input_, target in dataloader:
        pa_list.append(input_.numpy())
        xy_list.append(target.numpy())
    pa = np.concatenate(pa_list, axis=0)
    xy = np.concatenate(xy_list, axis=0)

    # distance from y[:, 0] and y[:, 1]
    exp10_Z = np.sqrt((xy[:, 0:1] - anchor_x) ** 2 + (xy[:, 1:2] - anchor_y) ** 2)
    Z = np.log10(exp10_Z)
    p = pa[:, 0]
    model = LinearRegression()
    model.fit(Z, p)
    path_loss_exponent = - model.coef_[0] / 10
    rss_1m = model.intercept_
    print('Estimated path loss exponent (n): ', path_loss_exponent)
    print('Estimated RSS at 1 meter (dBm): ', rss_1m)

    plt.figure(num=1)
    plt.scatter(exp10_Z, p, color='blue', marker='.', alpha=0.3, label='RSS values')
    plt.scatter(exp10_Z, model.predict(Z), color='red', marker='*', label='Log10 Regression Model')
    plt.xlabel('Distance')
    plt.ylabel('RSS')
    plt.title('RSS vs Distance')
    plt.grid(True)
    plt.legend()
    plt.show()

    return path_loss_exponent, rss_1m


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run BLE Position Static')
    parser.add_argument('--seed_search', type=int, help='Random seed for hyperparameter search', default=42)
    parser.add_argument('--seed_run', type=int, help='Random seed for model run', default=42)
    parser.add_argument('--n_configs', type=int, help='Number of configurations to generate', default=10)
    parser.add_argument('--device', type=str, help='The device to run the model', default='cpu')
    parser.add_argument('--develop', type=str, help='Choose the dataset to develop', required=True, choices=['east', 'north', 'south', 'west'])
    parser.add_argument('--test', type=str, help='Choose the dataset to test', required=True, choices=['east', 'north', 'south', 'west'])

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
        'hidden_units': [[32]],
        'batch_size': [256],
        'lr': [0.1],
        'weight_decay': [0.01],
        'val_split': [0.2],
        'patience': [10],
        'reduce_plateau': [0.1],
        'num_epochs': [500],
        # 'lambda_data': [1.0],
        # 'lambda_rss': [0.0],
        # 'lambda_azimuth': [0.0],
        # 'lambda_bc': [0.0],
        'lambda_data': (0.0, 1.0),
        'lambda_rss': (0.0, 1.0),
        'lambda_azimuth': (0.0, 1.0),
        'lambda_bc': (0.0, 1.0),
        'n_collocation': [512]
    }

    hyperparameters_list = random_search(seed=seed_search, hyperparameters=hyperparameters, n_configs=n_configs)

    # Set the log level to a higher level, e.g., WARNING or CRITICAL
    logging.disable(logging.CRITICAL)

    # create a csv file to save the hyperparameters and scores
    file_path = os.path.join('results', f'ble_position_static_{develop}_static_{test}.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open the file in append mode
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is being created
        if not file_exists:
            writer.writerow(list(hyperparameters.keys()) +
                            ['test_mse', 'test_rmse', 'test_50th', 'test_75th', 'test_90th', 'test_mae', 'test_min_ae', 'test_max_ae'])

    # Run your tests here
    for hyperparameters in tqdm(hyperparameters_list):
        scores = run_ble_position_static(seed_run, device, develop, test, hyperparameters)
        # save hyperparameters and scores in a new row of a CSV file
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list(hyperparameters.values()) + list(scores.values()))

        # if scores['test_rmse'] < best_score:
        #     best_score = scores['test_rmse']
        #     save_dict_to_yaml(hyperparameters, os.path.join('configs', f'ble_position_static_{develop}_static_{test}', 'best', 'hyperparameters.yaml'))
        #     save_dict_to_yaml(scores, os.path.join('configs', f'ble_position_static_{develop}_static_{test}', 'best', 'scores.yaml'))
     
    # Restore the original log level after the tests
    logging.disable(original_log_level)


def run_ble_position_static(seed, device, develop, test, hyperparameters, plot=False):

    task_name_develop = f'ble_position_static_{develop}'
    task_name_test = f'ble_position_static_{test}'
    logging.info(f"Develop on dataset: {task_name_develop}, Test on dataset: {task_name_test}")

    n_layers = hyperparameters['n_layers']
    hidden_units = hyperparameters['hidden_units']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    weight_decay = hyperparameters['weight_decay']
    val_split = hyperparameters['val_split']
    patience = hyperparameters['patience']
    reduce_plateau = hyperparameters['reduce_plateau']
    num_epochs = hyperparameters['num_epochs']
    lambda_data = hyperparameters['lambda_data']
    lambda_rss = hyperparameters['lambda_rss']
    lambda_azimuth = hyperparameters['lambda_azimuth']
    lambda_bc = hyperparameters['lambda_bc']
    n_collocation = hyperparameters['n_collocation']

    logging.info(f"Setting seed: {seed}")
    set_seed(seed)

    logging.info(f'Loading {task_name_develop} develop and {task_name_test} test datasets.')
    try:
        if develop == test:
            develop_dataset = load_data(os.path.join('datasets', task_name_develop, 'develop_dataset'))
            test_dataset = load_data(os.path.join('datasets', task_name_test, 'test_dataset'))
        else:
            develop_dataset = load_data(os.path.join('datasets', task_name_develop, 'full_dataset'))
            test_dataset = load_data(os.path.join('datasets', task_name_test, 'full_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for {task_name_develop} develop and/or {task_name_test} test.")

    d_input = develop_dataset[0][0].shape[0]

    logging.info('Splitting develop data into training and validation data.')
    train_dataset, val_dataset = random_split_dataset(dataset=develop_dataset, val_split=val_split)

    logging.info('Standardizing datasets.')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # path_loss_exponent, rss_1m = path_loss_extimation(train_dataloader, anchor_x, anchor_y)

    x_mean, x_std, y_mean, y_std = compute_mean_std(train_dataloader)

    train_dataset = StandardizeDataset(base_dataset=train_dataset,
                                       mean_input=x_mean, std_input=x_std,
                                       mean_target=y_mean, std_target=y_std)
    val_dataset = StandardizeDataset(base_dataset=val_dataset,
                                     mean_input=x_mean, std_input=x_std,
                                     mean_target=y_mean, std_target=y_std)
    develop_dataset = StandardizeDataset(base_dataset=develop_dataset,
                                         mean_input=x_mean, std_input=x_std,
                                         mean_target=y_mean, std_target=y_std)
    test_dataset = StandardizeDataset(base_dataset=test_dataset,
                                      mean_input=x_mean, std_input=x_std,
                                      mean_target=y_mean, std_target=y_std)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    develop_dataloader = DataLoader(develop_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f'Initializing model.')
    # model = PositionModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units,
    #                       anchor_x=anchor_x, anchor_y=anchor_y,
    #                       path_loss_exponent=path_loss_exponent, rss_1m=rss_1m)
    model = PositionModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units)

    # print_num_trainable_params(model)

    logging.info(f'Moving model to {device}.')
    model.to(device=torch.device(device))

    logging.info('Setting optimizer and trainer.')
    criterion = PositionLoss(lambda_data=lambda_data, lambda_rss=lambda_rss, lambda_azimuth=lambda_azimuth, lambda_bc=lambda_bc,
                             n_collocation=n_collocation,
                             mean_input=x_mean, std_input=x_std, mean_target=y_mean, std_target=y_std)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = TrainPhysicsModel(model=model, optimizer=optimizer, criterion=criterion,
                            develop_dataloader=develop_dataloader)

    logging.info(f'Fitting model for {task_name_develop}.')
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            patience=patience, reduce_plateau=reduce_plateau, num_epochs=num_epochs, verbose=False)
    logging.info(f"Model fitted for {task_name_test}.")

    logging.info('Evaluating model on develop set.')
    eval_dev = EvaluateRegressor(model=model, dataloader=develop_dataloader)
    eval_dev.evaluate(mean=y_mean, std=y_std, verbose=False)

    logging.info('Evaluating model on test set.')
    eval_test = EvaluateRegressor(model=model, dataloader=test_dataloader)
    eval_test.evaluate(mean=y_mean, std=y_std, verbose=False)
    scores = {'test_mse': eval_test.mse, 'test_rmse': eval_test.rmse,
              'test_50th': eval_test.p50, 'test_75th': eval_test.p75, 'test_90th': eval_test.p90,
              'test_mae': eval_test.mae, 'test_min_ae': eval_test.min_ae, 'test_max_ae': eval_test.max_ae}

    if plot:
        # scatter plot of y_true vs y_pred
        model.eval()
        with torch.no_grad():
            predictions = []
            targets = []
            
            for input_, target in tqdm(test_dataloader):
                input_ = input_.to(device)
                target = target.to(device)

                output = model(input_)  # Model outputs continuous values (not logits)
                predictions.append(output)
                targets.append(target)

            # Concatenate all tensors into one for proper calculations
            predictions, targets = torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
            predictions = predictions * y_std + y_mean
            targets = targets * y_std + y_mean

        plt.figure(num=1)
        plt.arrow(5.0, 7.5, cardinal_directions[f'{test}'][0], cardinal_directions[f'{test}'][1], head_width=0.3, head_length=0.3, fc='black', ec='black')
        plt.text(5.0, 7.5, test, fontsize=10)
        plt.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k-')
        x_anchors, y_anchors = zip(*anchor_positions.values())
        plt.scatter(x_anchors, y_anchors, s=100, color='red', marker='s', label='Anchors')
        for anchor_id, (x, y) in anchor_positions.items():
            plt.text(x, y, anchor_id, fontsize=10)
        plt.scatter(targets[:,0:1], targets[:,1:2], color='red', marker='.', alpha=0.3, label='Target Points')
        plt.scatter(predictions[:,0:1], predictions[:,1:2], color='blue', marker='.', alpha=0.3, label='Predicted Points')
        plt.grid(True)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('True Positions vs Predicted Positions')
        plt.legend()
        plt.show()

    return scores


if __name__ == "__main__":
    main()
    