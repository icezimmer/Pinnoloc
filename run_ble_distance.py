import logging
import os
import torch
from torch import optim
from Pinnoloc.utils.experiments import set_seed
from torch.utils.data import DataLoader
from Pinnoloc.utils.split_data import random_split_dataset
from Pinnoloc.utils.printing import print_num_trainable_params, print_parameters
from Pinnoloc.models.vector import StackedVectorModel, DistanceModel
from Pinnoloc.ml.optimization import setup_optimizer
from Pinnoloc.ml.loss import DistanceLoss
from Pinnoloc.ml.training import TrainPhysicsModel
from Pinnoloc.ml.evaluation import EvaluateRegressor
from Pinnoloc.dataset.preprocessing import compute_mean_std, StandardizeDataset
from Pinnoloc.utils.saving import load_data, save_hyperparameters, update_results, update_hyperparameters
from Pinnoloc.utils.check_device import check_model_device
from Pinnoloc.utils.experiments import read_yaml_to_dict
# from codecarbon import EmissionsTracker
from Pinnoloc.utils.saving import save_data


def main():
    logging.basicConfig(level=logging.INFO)

    task_name = 'ble_distance'

    seed = 42
    device = 'cpu'
    n_layers = 2
    hidden_units = [256]
    path_loss = 1.6234
    rss_1m = -58.2685
    batch_size = 256
    lr = 0.1
    weight_decay = 0.01
    val_split = 0.2
    patience = 10
    reduce_plateau = 0.1
    num_epochs = 100
    lambda_data = 1.0
    lambda_physics = 1.0
    lambda_bc = 1.0
    n_collocation = 10000

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
    x_mean, x_std, y_mean, y_std = compute_mean_std(train_dataloader)
    # x_mean, x_std, y_mean, y_std = None, None, None, None
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
    model = DistanceModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units, path_loss=path_loss, rss_1m=rss_1m)

    # print_num_trainable_params(model)

    logging.info(f'Moving model to {device}.')
    model.to(device=torch.device(device))

    logging.info('Setting optimizer and trainer.')
    criterion = DistanceLoss(lambda_data=lambda_data, lambda_physics=lambda_physics, lambda_bc=lambda_bc,
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
    from tqdm import tqdm
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        
        for item in tqdm(test_dataloader):
            if len(item) == 3:
                input_, target, _ = item  # Handle potential extra dataset info
            else:
                input_, target = item
            
            input_ = input_.to(device)
            target = target.to(device)

            output = model(input_)  # Model outputs continuous values (not logits)
            predictions.append(output.flatten())  # Keep as 1D tensor
            targets.append(target.flatten())  # Keep as 1D tensor

        # Concatenate all tensors into one for proper calculations
        predictions, targets = torch.cat(predictions), torch.cat(targets)
        predictions = predictions * y_std + y_mean
        targets = targets * y_std + y_mean

    import matplotlib.pyplot as plt
    plt.figure(num=1)
    plt.scatter(predictions, targets, color='blue', marker='.', alpha=0.3)
    plt.plot(range(0, 14), range(0, 14), color='red')
    plt.grid(True)
    plt.xlabel('Predicted Distance')
    plt.ylabel('True Distance')
    plt.title('True Distance vs Predicted Distance')
    plt.show()


if __name__ == "__main__":
    main()
    