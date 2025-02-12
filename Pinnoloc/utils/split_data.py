from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def random_split_dataset(dataset, val_split):
    """
    Randomly split dataset into training and validation sets.
    - Input:
        - dataset: torch.utils.data.Dataset
        - val_split: float, fraction of dataset to include in validation set
    """
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def stratified_split_dataset(dataset, val_split):
    """
    Stratified split dataset into training and validation sets.
    Usable for classification tasks.
    - Input:
        - dataset: torch.utils.data.Dataset
        - val_split: float, fraction of dataset to include in validation set
    """
    # Automatically extract targets by assuming dataset[i] returns (data, target) or (data, target, length)
    if len(dataset[0]) == 2:
        targets = [target for _, target in dataset]
    elif len(dataset[0]) == 3:
        targets = [target for _, target, _ in dataset]
    else:
        raise ValueError("Dataset must return either (inputs, targets) or (inputs, targets, physics)")

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=val_split,
        shuffle=True,
        stratify=targets
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
