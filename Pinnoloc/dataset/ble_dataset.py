import torch
from torch.utils.data import Dataset


class BLEDatasetHeading(Dataset):
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
        self.targets = torch.tensor(dataframe[target_column].values, dtype=torch.long)  # Classification
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


class BLEIMGDataset(Dataset):
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
        self.features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32).reshape(-1, len(feature_columns), 4)
        self.targets = torch.tensor(dataframe[target_column].values, dtype=torch.long)  # Classification
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