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
        

class BLEDatasetHeadingIMG(Dataset):
    def __init__(self, dataframe, key_columns, feature_columns, target_column, physics_columns=None, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            physics_columns (list): List of columns containing physics-based information.
            transform (callable, optional): Optional transformation on features.
        """
        self.feature_columns = feature_columns
        self.target_columns = target_column
        self.grouped = dataframe.groupby(key_columns)
        self.keys = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Get the (x_bin, y_bin) group corresponding to this index.
        key = self.keys[idx]
        group = self.grouped.get_group(key)
        
        # Ensure a consistent order for anchors (if anchor_id exists)
        group = group.sort_values('Anchor_ID')
        
        input_ = group[self.feature_columns].values
        input_ = torch.tensor(input_, dtype=torch.float32).unsqueeze(0)

        target = group[self.target_columns].values[0]
        target = torch.tensor(target, dtype=torch.long)
        
        # Convert the features to a torch tensor.
        # The resulting tensor will have shape (number_of_anchors, number_of_features)
        return input_, target


class BLEDatasetDistanceIMG(Dataset):
    def __init__(self, dataframe, key_columns, feature_columns, target_column, physics_columns=None, transform=None):
        """
        BLE dataset with physics-based information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            feature_columns (list): List of columns used as input features.
            target_column (str): The column name of the target variable.
            physics_columns (list): List of columns containing physics-based information.
            transform (callable, optional): Optional transformation on features.
        """
        self.feature_columns = feature_columns
        self.target_columns = target_column
        self.grouped = dataframe.groupby(key_columns)
        self.keys = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Get the (x_bin, y_bin) group corresponding to this index.
        key = self.keys[idx]
        group = self.grouped.get_group(key)
        
        # Ensure a consistent order for anchors (if anchor_id exists)
        group = group.sort_values('Anchor_ID')

        print(group)

        input_ = group[self.feature_columns].values
        input_ = torch.tensor(input_, dtype=torch.float32).unsqueeze(0)

        target = group[self.target_columns].values[0]
        target = torch.tensor(target, dtype=torch.float32)
        
        # Convert the features to a torch tensor.
        # The resulting tensor will have shape (number_of_anchors, number_of_features)
        return input_, target
