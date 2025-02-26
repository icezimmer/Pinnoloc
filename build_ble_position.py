import pandas as pd
import numpy as np
from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, load_grid_dataset, create_anchors_dataset
from Pinnoloc.dataset.ble_dataset import BLEDatasetDistance
from Pinnoloc.utils.split_data import random_split_dataset
from Pinnoloc.utils.saving import save_data
import logging
import os
from scipy import stats


# Create dictionary with the cardinal directions and the corresponding labels
cardinal_directions = {
    'east': 0,
    'north': 1,
    'south': 2,
    'west': 3
}


def create_df():

    def load_gt():
        def load_gt_(cardinal_direction, label):
            gt = load_gt_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_{cardinal_direction}.txt')
            gt['Heading'] = label
            return gt
        
        gt_list = [load_gt_(cardinal_direction, label) for cardinal_direction, label in cardinal_directions.items()]
        return gt_list


    def load_data():
        def load_data_(cardinal_direction):
            data = load_raw_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_{cardinal_direction}.txt')
            return data
        
        # Create a list of dataframes with the data for each cardinal direction
        data_list = [load_data_(cardinal_direction) for cardinal_direction, _ in cardinal_directions.items()]
        return data_list
    
    gt_list = load_gt()
    data_list = load_data()
    # Merge the data with the ground truth data
    df_list = [pd.merge_asof(data, gt, left_on='Epoch_Time', right_on='Start_Time', direction='backward') for data, gt in zip(data_list, gt_list)]
    # Filter the rows where the epoch time is within the interval [Start_Time, End_Time]
    df_list = [df[df['Epoch_Time'] <= df['End_Time']] for df in df_list]
    # Concatenate the dataframes
    df = pd.concat(df_list)
    # Sort the data by the epoch time
    df = df.sort_values(by='Epoch_Time')
    # Reset the index
    df = df.reset_index(drop=True)

    grid = load_grid_dataset()
    anchors = create_anchors_dataset()

    # Merge the data with the grid and anchors data (grid and anchors data are ground truth data)
    df = pd.merge(df, grid, how='inner', on=['X', 'Y', 'Anchor_ID'])
    df = pd.merge(df, anchors, how='inner', on='Anchor_ID')

    return df


def preprocess_df(df):
    # Convert cm distances to meters
    df['X'] = df['X'] / 100
    df['Y'] = df['Y'] / 100
    df['Anchor_x'] = df['Anchor_x'] / 100
    df['Anchor_y'] = df['Anchor_y'] / 100

    # Convert angles to radians
    df['AoA_Az'] = np.radians(df['AoA_Az'])
    # df['AoA_El'] = np.radians(df['AoA_El'])
    # df['Az_Arrival'] = np.radians(df['Az_Arrival'])
    # df['El_Arrival'] = np.radians(df['El_Arrival'])

    # Transform the observed azimuth angle of arrival to the canonical reference system
    df.loc[df['Anchor_ID'] == 6501, 'AoA_Az'] = - df.loc[df['Anchor_ID'] == 6501, 'AoA_Az']  # Left Anchor (East direction)
    df.loc[df['Anchor_ID'] == 6502, 'AoA_Az'] = (np.pi / 2) - df.loc[df['Anchor_ID'] == 6502, 'AoA_Az']  # Bottom Anchor (North direction)
    df.loc[df['Anchor_ID'] == 6503, 'AoA_Az'] = np.pi - df.loc[df['Anchor_ID'] == 6503, 'AoA_Az']  # Right Anchor (West direction)
    df.loc[df['Anchor_ID'] == 6504, 'AoA_Az'] = - (np.pi / 2) - df.loc[df['Anchor_ID'] == 6504, 'AoA_Az']  # Top Anchor (South direction)

    # # Transform the gt azimuth angle of arrival to the canonical reference system
    # df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival'] = - df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival']  # East Anchor
    # df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival'] = (np.pi / 2) - df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival']  # North Anchor
    # df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival'] = - (np.pi / 2) - df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival']  # South Anchor
    # df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival'] = np.pi - df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival']  # West Anchor

    print(df.columns)
    
    # Take only the Anchor_ID = 6501
    df = df[df['Anchor_ID'] == 6501]
    # Take only the Channel = 37 with the 2nd polarization
    df = df[df['Channel'] == 37]
    df = df[df['Heading'] == cardinal_directions['west']]

    # For each Distance take the Z score of RSS_2nd_Pol less than 2
    # df['zscore'] = df.groupby('Distance')['RSS_2nd_Pol'].transform(lambda x: stats.zscore(x))
    # df = df[df['zscore'].abs() < 2]

    # Take only the second polarization
    df['feature/RSS'] = df['RSS_2nd_Pol']
    # # cos and sin of the azimuth angle
    # df['feature/AoA_Az_x'] = df['AoA_Az'].apply(lambda x: np.cos(x))
    # df['feature/AoA_Az_y'] = df['AoA_Az'].apply(lambda x: np.sin(x))
    df['feature/AoA_Az'] = np.arctan2(np.sin(df['AoA_Az']), np.cos(df['AoA_Az']))  # set the angle between -pi and pi

    df['target/X'] = df['X']
    df['target/Y'] = df['Y']

    # # Scatter plot of AoA_Az_x vs AoA_Az_y
    # import matplotlib.pyplot as plt
    # plt.figure(num=1)
    # plt.scatter(df['feature/AoA_Az_x'], df['feature/AoA_Az_y'])
    # plt.grid(True)
    # plt.xlabel('cos(AoA_Az)')
    # plt.ylabel('sin(AoA_Az)')
    # plt.show()

    return df


def main():
    logging.basicConfig(level=logging.INFO)

    task_name = 'ble_position'

    df = create_df()
    print(df)
    df = preprocess_df(df)
    print(df)

    feature_columns = [col for col in df.columns if col.startswith('feature/')]
    target_column = [col for col in df.columns if col.startswith('target/')]
    physics_columns = [col for col in df.columns if col.startswith('physics/')]
    dataset = BLEDatasetDistance(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
        physics_columns=physics_columns
        )
    
    print('(', feature_columns, '), ', '(', target_column, '), ', '(', physics_columns, ')')
    print(dataset[0])

    develop_dataset, test_dataset = random_split_dataset(dataset, val_split=0.2)

    logging.info('Saving datasets')
    save_data(develop_dataset, os.path.join('datasets', task_name, 'develop_dataset'))
    save_data(test_dataset, os.path.join('datasets', task_name, 'test_dataset'))


if __name__ == "__main__":
    main()
