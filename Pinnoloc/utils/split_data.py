from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def random_split_dataset(dataset, val_split):
    """
    Randomly split dataset into training and validation sets.
    - Input:
        - dataset: torch.utils.data.Dataset
        - val_split: float, fraction of dataset to include in validation set
    - Output:
        - train_dataset: torch.utils.data.Subset
        - val_dataset: torch.utils.data.Subset
    """
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def stratified_split_dataset(dataset, stratify, val_split):
    """
    Stratified split dataset into training and validation sets.
    Usable for classification tasks.
    - Input:
        - dataset: torch.utils.data.Dataset
        - stratify: list, array of target values, criterion for stratification
        - val_split: float, fraction of dataset to include in validation set
    - Output:
        - train_dataset: torch.utils.data.Subset
        - val_dataset: torch.utils.data.Subset
        - train_stratify: list, array of target values for training set
        - val_stratify: list, array of target values for validation set
    """
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=val_split,
        shuffle=True,
        stratify=stratify
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_stratify = [stratify[i] for i in train_indices]
    val_stratify = [stratify[i] for i in val_indices]

    return train_dataset, val_dataset, train_stratify, val_stratify
