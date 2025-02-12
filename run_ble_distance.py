import logging
import os
import torch
from Pinnoloc.utils.experiments import set_seed
from torch.utils.data import DataLoader
from Pinnoloc.utils.split_data import random_split_dataset
from Pinnoloc.utils.printing import print_num_trainable_params, print_parameters
from Pinnoloc.models.vector import StackedVectorModel
from Pinnoloc.ml.optimization import setup_optimizer
from Pinnoloc.ml.loss import DistanceLoss
from Pinnoloc.ml.training import TrainPhysicsModel
from Pinnoloc.ml.evaluation import EvaluateRegressor
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
    n_layers = 4
    d_input = 1
    hidden_units = [256]
    d_output = 1
    batch_size = 256
    lr = 0.1
    weight_decay = 0.01
    val_split = 0.2
    patience = 10
    reduce_plateau = 0.1
    num_epochs = 100
    lambda_data = 1.0
    lambda_physics = 1.0

    logging.info(f"Setting seed: {seed}")
    set_seed(seed)

    criterion = DistanceLoss(lambda_data=lambda_data, lambda_physics=lambda_physics)

    logging.info(f'Loading {task_name} develop and test datasets.')
    try:
        develop_dataset = load_data(os.path.join('datasets', task_name, 'develop_dataset'))
        test_dataset = load_data(os.path.join('datasets', task_name, 'test_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for {task_name}. Run build.py first.")

    develop_dataloader = DataLoader(develop_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)

    logging.info(f'Initializing model.')
    model = StackedVectorModel(n_layers=n_layers, d_input=d_input, hidden_units=hidden_units, d_output=d_output)

    # print_num_trainable_params(model)

    logging.info(f'Moving model to {device}.')
    model.to(device=torch.device(device))

    logging.info('Splitting develop data into training and validation data.')
    train_dataset, val_dataset = random_split_dataset(dataset=develop_dataset, val_split=val_split)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info('Setting optimizer and trainer.')
    optimizer = setup_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    trainer = TrainPhysicsModel(model=model, optimizer=optimizer, criterion=criterion,
                            develop_dataloader=develop_dataloader)


    logging.info(f'Fitting model for {task_name}.')
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            patience=patience, reduce_plateau=reduce_plateau, num_epochs=num_epochs)
    logging.info(f"Model fitted for {task_name}.")

    logging.info('Evaluating model on develop set.')
    eval_dev = EvaluateRegressor(model=model, dataloader=develop_dataloader)
    mean = develop_dataset[0][2][0].item()
    std = develop_dataset[0][2][1].item()
    eval_dev.evaluate(mean=mean, std=std)

    logging.info('Evaluating model on test set.')
    eval_test = EvaluateRegressor(model=model, dataloader=test_dataloader)
    eval_test.evaluate(mean=mean, std=std)
    scores = {'test_mse': eval_test.mse}

    print(scores)


if __name__ == "__main__":
    main()
    
