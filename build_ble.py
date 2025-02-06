import pandas as pd
from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, load_grid_dataset
import torch
from torch.utils.data import Dataset


class HeadingDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing the data.
            feature_columns (list): List of column names used as input features.
            target_column (str): The column name of the target variable.
            transform (callable, optional): Optional transform to be applied on features.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Convert features and targets to tensors
        self.features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_column].values, dtype=torch.long)

        # Apply transform if provided
        if transform:
            self.features = transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def main():


    data_east = load_raw_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_east.txt')
    # data_east['Label'] = 0
    data_north = load_raw_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_north.txt')
    # data_north['Label'] = 1
    data_south = load_raw_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_south.txt')
    # data_south['Label'] = 2
    data_west = load_raw_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_west.txt')
    # data_west['Label'] = 3

    # # concatenate the data
    # data = pd.concat([data_east, data_north, data_south, data_west])
    # # sort the data by the epoch time
    # data = data.sort_values(by='Epoch_Time')
    # # reset the index
    # data = data.reset_index(drop=True)
    # print(data)


    gt_east = load_gt_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_east.txt')
    gt_east['Label'] = 0
    gt_north = load_gt_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_north.txt')
    gt_north['Label'] = 1
    gt_south = load_gt_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_south.txt')
    gt_south['Label'] = 2
    gt_west = load_gt_dataset('data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_west.txt')
    gt_west['Label'] = 3

    # merge the ground truth data with the data on timestamp: if timestamp is in the interval [Start_Time, End_Time]
    # then the label is the corresponding label
    data_east = pd.merge_asof(data_east, gt_east, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    data_east = data_east[data_east['Epoch_Time'] <= data_east['End_Time']]
    # remove start and end time columns
    data_east = data_east.drop(columns=['Start_Time', 'End_Time'])
    print(data_east)

    data_north = pd.merge_asof(data_north, gt_north, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    data_north = data_north[data_north['Epoch_Time'] <= data_north['End_Time']]
    # remove start and end time columns
    data_north = data_north.drop(columns=['Start_Time', 'End_Time'])
    print(data_north)

    data_south = pd.merge_asof(data_south, gt_south, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    data_south = data_south[data_south['Epoch_Time'] <= data_south['End_Time']]
    # remove start and end time columns
    data_south = data_south.drop(columns=['Start_Time', 'End_Time'])
    print(data_south)

    data_west = pd.merge_asof(data_west, gt_west, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    data_west = data_west[data_west['Epoch_Time'] <= data_west['End_Time']]
    # remove start and end time columns
    data_west = data_west.drop(columns=['Start_Time', 'End_Time'])
    print(data_west)

    # concatenate the data
    data = pd.concat([data_east, data_north, data_south, data_west])
    # sort the data by the epoch time
    data = data.sort_values(by='Epoch_Time')
    # reset the index
    data = data.reset_index(drop=True)
    print(data.head(10))

    # create the BLE dataset
    dataset = HeadingDataset(data, feature_columns=['RSS_1st_Pol', 'AoA_Az', 'AoA_El', 'RSS_2nd_Pol'], target_column='Label')
    print(dataset[0])



if __name__ == "__main__":
    main()