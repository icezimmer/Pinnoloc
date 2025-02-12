import logging
import os
import torch
from Pinnoloc.utils.experiments import set_seed
from torch.utils.data import DataLoader
from Pinnoloc.utils.split_data import stratified_split_dataset
from Pinnoloc.utils.printing import print_num_trainable_params, print_parameters
from Pinnoloc.models.vector import StackedVectorModel
from Pinnoloc.ml.optimization import setup_optimizer
from Pinnoloc.ml.loss import HeadingLoss
from Pinnoloc.ml.training import TrainPhysicsModel
from Pinnoloc.ml.evaluation import EvaluateClassifier
from Pinnoloc.utils.saving import load_data, save_hyperparameters, update_results, update_hyperparameters
from Pinnoloc.utils.check_device import check_model_device
from Pinnoloc.utils.experiments import read_yaml_to_dict
# from codecarbon import EmissionsTracker
from Pinnoloc.utils.saving import save_data


def main():
    logging.basicConfig(level=logging.INFO)

    task_name = 'ble_heading'

    seed = 42
    device = 'cpu'

    logging.info(f"Setting seed: {seed}")
    set_seed(seed)

    n_layers = 4
    d_input = 5
    hidden_units = [256]
    d_output = 4
    batch_size = 256
    lr = 0.1
    weight_decay = 0.0
    val_split = 0.2
    patience = 200
    reduce_paleau = 0.0
    num_epochs = 1000
    lambda_data = 0.0
    lambda_physics = 1.0
    A_max = 10.0

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = HeadingLoss(lambda_data=lambda_data, lambda_physics=lambda_physics, A_max=A_max)

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
    train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info('Setting optimizer and trainer.')
    optimizer = setup_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    trainer = TrainPhysicsModel(model=model, optimizer=optimizer, criterion=criterion,
                            develop_dataloader=develop_dataloader)


    logging.info(f'Fitting model for {task_name}.')
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            patience=patience, reduce_plateau=reduce_paleau, num_epochs=num_epochs)
    logging.info(f"Model fitted for {task_name}.")

    logging.info('Evaluating model on develop set.')
    eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
    eval_dev.evaluate()

    logging.info('Evaluating model on test set.')
    eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
    eval_test.evaluate()
    scores = {'test_accuracy': eval_test.accuracy_value}

    print(scores)


if __name__ == "__main__":
    main()
    
