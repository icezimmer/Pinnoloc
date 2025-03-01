import random
import numpy as np
from Pinnoloc.utils.experiments import set_seed


def random_search(seed, hyperparameters, n_configs):
    """
    Random search on the hyperparameters of the model
    the hyperparameters are saved in a dictionary
    the grid search produces a list of dictionaries

    -Input:
        - hyperparameters: dictionary with the hyperparameters
        - n_configs: number of configurations to generate
    -Output:
        - list of dictionaries with the hyperparameters
    
    """
    set_seed(seed)

    hyperparameters_list = []
    for _ in range(n_configs):
        hyperparameters_copy = hyperparameters.copy()
        for key, value in hyperparameters_copy.items():
            if type(value) == list:
                hyperparameters_copy[key] = random.choice(value)
            if type(value) == tuple:
                hyperparameters_copy[key] = np.random.uniform(value[0], value[1])
        hyperparameters_list.append(hyperparameters_copy)
    return hyperparameters_list
