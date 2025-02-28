# Random search on the hyperparameters of the model
# the hyperparameters are saved in a dictionary
# the grid search produces a list of dictionaries

import random
import numpy as np

def random_search(hyperparameters, n=10):
    """
    Random search on the hyperparameters of the model
    the hyperparameters are saved in a dictionary
    the grid search produces a list of dictionaries

    Parameters:
    hyperparameters: dictionary
        dictionary of hyperparameters
    n: int
        number of random searches

    Returns:
    list of dictionaries
    """
    hyperparameters_list = []
    for _ in range(n):
        hyperparameters_copy = hyperparameters.copy()
        for key, value in hyperparameters_copy.items():
            if type(value) == list:
                hyperparameters_copy[key] = random.choice(value)
            if type(value) == tuple:
                hyperparameters_copy[key] = np.random.uniform(value[0], value[1])
        hyperparameters_list.append(hyperparameters_copy)
    return hyperparameters_list
