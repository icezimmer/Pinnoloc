import torch
from torch.utils.data import Dataset


class BLEDatasetDistance(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, physics_columns=None, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            physics_columns (list): List of columns containing physics-based information.
            transform (callable, optional): Optional transformation on features.
        """
        self.physics_columns = physics_columns

        # Convert features, targets, and physics data to tensors
        self.features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_column].values, dtype=torch.float32)  # Regression
        if physics_columns:
            self.physics = torch.tensor(dataframe[physics_columns].values, dtype=torch.float32)

        # Apply transform if provided
        if transform:
            self.features = transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.physics_columns:
            return self.features[idx], self.targets[idx], self.physics[idx]
        else:
            return self.features[idx], self.targets[idx]


class BLEDatasetDistanceIMG(Dataset):
    def __init__(self, dataframe, key_columns, sort_columns, feature_columns, target_column, physics_columns=None, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            physics_columns (list): List of columns containing physics-based information.
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
