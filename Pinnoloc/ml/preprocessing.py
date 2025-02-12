# given a torch dataset compute the mean and standard deviation for each feature
# and return the mean and standard deviation for each feature

import torch

def compute_mean_std(dataset):
    inputs = torch.stack([sample[0] for sample in dataset])
    