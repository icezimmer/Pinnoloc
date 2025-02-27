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
from Pinnoloc.dataset.preprocessing import compute_mean_std, StandardizeDataset
from Pinnoloc.utils.saving import load_data, save_hyperparameters, update_results, update_hyperparameters
from Pinnoloc.utils.check_device import check_model_device
from Pinnoloc.utils.experiments import read_yaml_to_dict
# from codecarbon import EmissionsTracker
from Pinnoloc.utils.saving import save_data
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def main():
    logging.basicConfig(level=logging.INFO)

    task_name = 'ble_position'

    seed = 42
    device = 'cpu'
    n_layers = 2
    hidden_units = [256]
    #
    heading = 'east'  # to plot the arrow
    #
    # path_loss_exponent = 1.4335
    # rss_1m = -60.9078
    anchor_x, anchor_y = anchor_positions['6503']
    batch_size = 256
    lr = 0.1
    weight_decay = 0.01
    val_split = 0.2
    patience = 10
    reduce_plateau = 0.1
    num_epochs = 100
    lambda_data = 1.0
    lambda_rss = 0.5
    lambda_azimuth = 0.5
    lambda_bc = 0.0
    n_collocation = 1000

    logging.info(f"Setting seed: {seed}")
    set_seed(seed)

    logging.info(f'Loading {task_name} develop and test datasets.')
    try:
        develop_dataset = load_data(os.path.join('datasets', task_name, 'develop_dataset'))
        test_dataset = load_data(os.path.join('datasets', task_name, 'test_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for {task_name}. Run build.py first.")

    d_input = develop_dataset[0][0].shape[0]

    logging.info('Splitting develop data into training and validation data.')
    train_dataset, val_dataset = random_split_dataset(dataset=develop_dataset, val_split=val_split)

    logging.info('Standardizing datasets.')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    path_loss_exponent, rss_1m = path_loss_extimation(train_dataloader, anchor_x, anchor_y)

    x_mean, x_std, y_mean, y_std = compute_mean_std(train_dataloader)
    print('input_mean: ', x_mean)
    print('input_std: ', x_std)
    print('target_mean: ', y_mean)
    print('target_std: ', y_std)

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
    model = PositionModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units,
                          anchor_x=anchor_x, anchor_y=anchor_y,
                          path_loss_exponent=path_loss_exponent, rss_1m=rss_1m)

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

    logging.info(f'Fitting model for {task_name}.')
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            patience=patience, reduce_plateau=reduce_plateau, num_epochs=num_epochs)
    logging.info(f"Model fitted for {task_name}.")

    logging.info('Evaluating model on develop set.')
    eval_dev = EvaluateRegressor(model=model, dataloader=develop_dataloader)
    eval_dev.evaluate(mean=y_mean, std=y_std)

    logging.info('Evaluating model on test set.')
    eval_test = EvaluateRegressor(model=model, dataloader=test_dataloader)
    eval_test.evaluate(mean=y_mean, std=y_std)
    scores = {'test_mse': eval_test.mse}

    print(scores)

    # scatter plot of y_true vs y_pred
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        
        for input_, target in tqdm(test_dataloader):
            input_ = input_.to(device)
            target = target.to(device)

            output = model(input_)  # Model outputs continuous values (not logits)
            predictions.append(output)  # Keep as 1D tensor
            targets.append(target)  # Keep as 1D tensor

        # Concatenate all tensors into one for proper calculations
        predictions, targets = torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
        predictions = predictions * y_std + y_mean
        targets = targets * y_std + y_mean

    plt.figure(num=1)
    plt.scatter(anchor_x, anchor_y, c='red', marker='o', s=100, label='Anchor')
    plt.arrow(5.0, 7.5, cardinal_directions[f'{heading}'][0], cardinal_directions[f'{heading}'][1], head_width=0.3, head_length=0.3, fc='black', ec='black', label='Heading')
    plt.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k-')
    plt.scatter(targets[:,0:1], targets[:,1:2], color='red', marker='.', alpha=0.3, label='Target Points')
    plt.scatter(predictions[:,0:1], predictions[:,1:2], color='blue', marker='.', alpha=0.3, label='Predicted Points')
    plt.grid(True)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('True Positions vs Predicted Positions')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    