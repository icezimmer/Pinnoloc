import torch
from torch.utils.data import Dataset


class BLEDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            transform (callable, optional): Optional transformation on features.
        """

        # Convert features, targets, and physics data to tensors
        self.features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_column].values, dtype=torch.float32)  # Regression task

        # Apply transform if provided
        if transform:
            self.features = transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BLEDatasetIMG(Dataset):
    def __init__(self, dataframe, key_columns, sort_columns, feature_columns, target_column, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            transform (callable, optional): Optional transformation on features.
        """
        self.data = []
        for key, group in dataframe.groupby(key_columns):
            group = group.sort_values(sort_columns)
            input_ = torch.tensor(group[feature_columns].values, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor(group[target_column].values, dtype=torch.float32).squeeze(-1)
            self.data.append((input_, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
