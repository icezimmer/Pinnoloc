import logging
import os
import torch
from torch import optim
from Pinnoloc.utils.experiments import set_seed
from torch.utils.data import DataLoader
from Pinnoloc.utils.split_data import random_split_dataset, stratified_split_dataset
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

    for anchor in range(len(anchor_x)):
        # distance from y[:, 0] and y[:, 1]
        exp10_Z = np.sqrt((xy[:, 0:1] - anchor_x[anchor]) ** 2 + (xy[:, 1:2] - anchor_y[anchor]) ** 2)
        Z = np.log10(exp10_Z)
        p = pa[:, anchor]
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
    parser.add_argument('--seed_run', type=int, help='Random seed for model run', default=42)
    parser.add_argument('--device', type=str, help='The device to run the model', default='cpu')
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
    parser.add_argument('--usage', type=float, help='The usage ratio of the develop dataset', required=True)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    seed_run = args.seed_run
    device = args.device
    develop = args.develop
    test = args.test

    hyperparameters = {
        'n_layers': 2,
        'hidden_units': [8],
        'batch_size': 256,
        'lr': 0.1,
        'weight_decay': 0.01,
        'val_split': 0.2,
        'patience': 10,
        'reduce_plateau': 0.1,
        'num_epochs': 500,
        'lambda_data': 1.0,
        'lambda_physics': 100.0,
        'lambda_bc': 0.0,
        'n_collocation': 512,
        'n_boundary_collocation': 32,
        'resampling_period': 16,
        'sigma_rss': 2.0,  # 2.0
        'sigma_aoa': 5.0,  # 5.0
    }

    logging.info(f"Setting seed: {seed_run}")
    set_seed(seed_run)

    task_name_develop = f'ble_position_{develop}'
    task_name_test = f'ble_position_{test}'

    logging.info(f'Loading {task_name_develop} develop and {task_name_test} test datasets.')
    try:
        if develop == test:
            develop_split = args.usage
            test_split = 1.0 - develop_split
            full_dataset = load_data(os.path.join('datasets', task_name_develop, 'full_dataset'))
            # develop_dataset, test_dataset = random_split_dataset(full_dataset, val_split=test_split)
            criterion_to_stratify = load_data(os.path.join('datasets', task_name_develop, 'criterion_to_stratify'))
            develop_dataset, test_dataset, criterion_to_stratify_develop, _ = stratified_split_dataset(dataset=full_dataset, stratify=criterion_to_stratify, val_split=test_split)
        else:
            develop_split = args.usage
            test_split = 1.0
            develop_dataset = load_data(os.path.join('datasets', task_name_develop, 'full_dataset'))
            if develop_split == 1.0:
                criterion_to_stratify_develop = load_data(os.path.join('datasets', task_name_develop, 'criterion_to_stratify'))
            else:
                criterion_to_stratify = load_data(os.path.join('datasets', task_name_develop, 'criterion_to_stratify'))
                develop_dataset, _, criterion_to_stratify_develop, _ = stratified_split_dataset(dataset=develop_dataset, stratify=criterion_to_stratify, val_split=1.0-develop_split)
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


    scores = run_ble_position(seed_run, device, develop_dataset, test_dataset, hyperparameters,
                              criterion_to_stratify_develop,
                              verbose=True, plot=True)

    # save hyperparameters and scores in a new row of a CSV file
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([None, seed_run] +  list(hyperparameters.values())
                        + list(scores.values()) +
                        list(develop_args.values()) + list(test_args.values()))


def run_ble_position(seed, device, develop_dataset, test_dataset, hyperparameters,
                    criterion_to_stratify_develop,
                    verbose=False, plot=False):

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
    lambda_physics = hyperparameters['lambda_physics']
    lambda_bc = hyperparameters['lambda_bc']
    n_collocation = hyperparameters['n_collocation']
    n_boundary_collocation = hyperparameters['n_boundary_collocation']
    resampling_period = hyperparameters['resampling_period']
    sigma_rss = hyperparameters['sigma_rss']
    sigma_aoa = hyperparameters['sigma_aoa']

    logging.info(f"Setting seed: {seed}")
    set_seed(seed)

    d_input = develop_dataset[0][0].shape[0]

    logging.info('Splitting develop data into training and validation data.')
    # train_dataset, val_dataset = random_split_dataset(dataset=develop_dataset, val_split=val_split)
    train_dataset, val_dataset, _, _ = stratified_split_dataset(dataset=develop_dataset, stratify=criterion_to_stratify_develop, val_split=val_split)

    logging.info('Standardizing datasets.')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # path_loss_exponent, rss_1m = path_loss_extimation(train_dataloader, anchor_x=[0.0, 6.0, 12.0, 6.0], anchor_y=[3.0, 0.0, 3.0, 6.0],)

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
    model = PositionModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units)

    # print_num_trainable_params(model)

    logging.info(f'Moving model to {device}.')
    model.to(device=torch.device(device))

    logging.info('Setting optimizer and trainer.')
    criterion = PositionLoss(lambda_data=lambda_data, lambda_physics=lambda_physics, lambda_bc=lambda_bc,
                             n_collocation=n_collocation, n_boundary_collocation=n_boundary_collocation,
                             seed=seed,
                             mean_input=x_mean, std_input=x_std, mean_target=y_mean, std_target=y_std,
                             sigma_rss=sigma_rss, sigma_aoa=sigma_aoa)
    criterion.to(device=torch.device(device))
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = TrainPhysicsModel(model=model, optimizer=optimizer, criterion=criterion,
                                resampling_period=resampling_period,
                                develop_dataloader=develop_dataloader)

    logging.info(f'Fitting model.')
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            patience=patience, reduce_plateau=reduce_plateau, num_epochs=num_epochs, verbose=verbose)
    logging.info(f"Model fitted.")

    # plt.plot(trainer.training_loss, label='Training Loss')
    # plt.plot(trainer.validation_loss, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    logging.info('Evaluating model on develop set.')
    eval_dev = EvaluateRegressor(model=model, dataloader=develop_dataloader)
    eval_dev.evaluate(mean=y_mean, std=y_std, verbose=verbose)

    logging.info('Evaluating model on test set.')
    eval_test = EvaluateRegressor(model=model, dataloader=test_dataloader)
    eval_test.evaluate(mean=y_mean, std=y_std, verbose=verbose)
    scores = {'test_mse': eval_test.mse, 'test_rmse': eval_test.rmse,
              'test_50th': eval_test.p50, 'test_75th': eval_test.p75, 'test_90th': eval_test.p90,
              'test_mae': eval_test.mae, 'test_min_ae': eval_test.min_ae, 'test_max_ae': eval_test.max_ae}

    if plot:
        # scatter plot of y_true vs y_pred
        model.to('cpu')
        model.eval()
        with torch.no_grad():
            predictions = []
            targets = []
            
            for input_, target in tqdm(test_dataloader):

                output = model(input_)  # Model outputs continuous values (not logits)
                predictions.append(output)
                targets.append(target)

            # Concatenate all tensors into one for proper calculations
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            predictions = predictions * y_std + y_mean
            targets = targets * y_std + y_mean

        plt.figure(num=1)
        # Set font size
        plt.rcParams.update({'font.size': 14})
        plt.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k-')
        x_anchors, y_anchors = zip(*anchor_positions.values())
        plt.scatter(x_anchors, y_anchors, s=100, color='yellow', marker='s', label='Anchors')
        for anchor_id, (x, y) in anchor_positions.items():
            plt.text(x, y, anchor_id, fontsize=10)
        plt.scatter(targets[:,0:1], targets[:,1:2], color='red', marker='.', alpha=0.3, label='Target Points')
        plt.scatter(predictions[:,0:1], predictions[:,1:2], color='blue', marker='.', alpha=0.3, label='Predicted Points')
        # Ploot the lines between the respctive predicted and target points
        for i in range(targets.shape[0]):
            plt.plot([targets[i,0], predictions[i,0]], [targets[i,1], predictions[i,1]], color='black', alpha=0.1)
        plt.grid(True)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.title('True Positions vs Predicted Positions')
        # Put not overlapping legend outside the plot
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='x-small')
        plt.show()

    return scores


if __name__ == "__main__":
    main()
    