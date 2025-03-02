import torch
import random
import numpy as np
# import tensorflow as tf
import yaml
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # tf.random.set_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def save_dict_to_yaml(dictionary, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(dictionary, file)
