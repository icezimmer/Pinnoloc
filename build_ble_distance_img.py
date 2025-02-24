import pandas as pd
import numpy as np
from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, load_grid_dataset, create_anchors_dataset
from Pinnoloc.dataset.ble_dataset import BLEDatasetDistanceIMG
from Pinnoloc.utils.split_data import random_split_dataset
from Pinnoloc.utils.saving import save_data
import logging
import os


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

    # Compute the euclidean distance between the anchor (Anchor_x, Anchor_y) and the tag (X, Y) coordinates
    df['Distance'] = ((df['X'] - df['Anchor_x'])**2 + (df['Y'] - df['Anchor_y'])**2)**0.5

    # Compute the angle of departure (Azimuth) from the tag (X, Y) to the anchor (Anchor_x, Anchor_y)
    df['Az_Departure'] = np.arctan2(df['Anchor_y'] - df['Y'], df['Anchor_x'] - df['X'])

    # Convert angles to radians
    df['AoA_Az'] = np.radians(df['AoA_Az'])
    df['AoA_El'] = np.radians(df['AoA_El'])
    df['Az_Arrival'] = np.radians(df['Az_Arrival'])
    df['El_Arrival'] = np.radians(df['El_Arrival'])

    df['AoA_Az_x'] = df['AoA_Az'].apply(lambda x: np.cos(x))
    df['AoA_Az_y'] = df['AoA_Az'].apply(lambda x: np.sin(x))
    df['AoA_El_x'] = df['AoA_El'].apply(lambda x: np.cos(x))
    df['AoA_El_y'] = df['AoA_El'].apply(lambda x: np.sin(x))

    df['Pos_ID'] = df.groupby(['X', 'Y']).ngroup()
    
    df = df.groupby(['Pos_ID', 'Anchor_ID']).agg({
        'RSS_1st_Pol': 'mean',
        'RSS_2nd_Pol': 'mean',
        'AoA_Az_x': 'mean',
        'AoA_Az_y': 'mean',
        'AoA_El_x': 'mean',
        'AoA_El_y': 'mean',
        'Distance': 'mean'
    }).reset_index()

    df = df.sort_values(by=['Pos_ID', 'Anchor_ID'])

    # rename columns
    df = df.rename(columns={
        'Pos_ID': 'key/Pos_ID',
        'Anchor_ID': 'sort/Anchor_ID',
        'RSS_1st_Pol': 'feature/RSS_1st_Pol',
        'RSS_2nd_Pol': 'feature/RSS_2nd_Pol',
        'AoA_Az_x': 'feature/AoA_Az_x',
        'AoA_Az_y': 'feature/AoA_Az_y',
        'AoA_El_x': 'feature/AoA_El_x',
        'AoA_El_y': 'feature/AoA_El_y',
        'Distance': 'target/Distance'
    })

    # df_grouped = df.groupby(['X', 'Y', 'Heading'])
    # keys = list(df_grouped.groups.keys())
    # print(keys)
    # for key in keys:
    #     print('Prova')
    #     print(key)
    #     group = df_grouped.get_group(key)
    #     group = group.sort_values('Anchor_ID')
    #     print(group)
    #     # group = group.groupby('Anchor_ID').agg({
    #     #     'RSS_1st_Pol': 'mean',
    #     #     'RSS_2nd_Pol': 'mean',
    #     #     'Distance': 'mean'
    #     # }).reset_index()
    #     # print(group)

    # df[df['Heading'] == 0]['Az_Departure'] = - df[df['Heading'] == 0]['Az_Departure']  # East
    # df[df['Heading'] == 1]['Az_Departure'] = (np.pi / 2) - df[df['Heading'] == 1]['Az_Departure']  # North
    # df[df['Heading'] == 2]['Az_Departure'] = - (np.pi / 2) - df[df['Heading'] == 2]['Az_Departure']  # South
    # df[df['Heading'] == 3]['Az_Departure'] = np.pi - df[df['Heading'] == 3]['Az_Departure']  # West

    # df = df[['Epoch_Time', 'RSS_1st_Pol', 'AoA_Az', 'AoA_El',
    #    'RSS_2nd_Pol', 'Channel', 'Anchor_ID', 'X',
    #    'Y', 'Distance', 'Az_Arrival', 'El_Arrival', 'Az_Departure', 'Heading']]

    # # Group by X, Y, Anchor_Id, Label
    # df = df.groupby(['X', 'Y', 'Anchor_ID', 'Heading']).agg({
    #     'RSS_1st_Pol': 'mean',
    #     'RSS_2nd_Pol': 'mean',
    #     'AoA_Az': 'mean',
    #     'AoA_El': 'mean',
    #     'Distance': 'mean',
    #     'Az_Arrival': 'mean',
    #     'El_Arrival': 'mean',
    #     'Az_Departure': 'mean'
    # }).reset_index()

    # print(df.head())

    # # df = df.pivot(index=['X', 'Y', 'Heading'], columns='Anchor_ID', values=['RSS_1st_Pol', 'RSS_2nd_Pol', 'Distance'])

    # # print(df)
    # # import torch
    # # print(torch.tensor(df[['RSS_1st_Pol', 'RSS_2nd_Pol']].values, dtype=torch.float32).reshape(-1, 2, 4))
    # # print(torch.tensor(df['Distance'].values, dtype=torch.float32))

    # # Standardize the RSS_1st_Pol, AoA_Az, AoA_El, RSS_2nd_Pol columns respectively
    # df['feature/RSS_1st_Pol'] = (df['RSS_1st_Pol'] - df['RSS_1st_Pol'].mean()) / df['RSS_1st_Pol'].std()
    # df['feature/RSS_2nd_Pol'] = (df['RSS_2nd_Pol'] - df['RSS_2nd_Pol'].mean()) / df['RSS_2nd_Pol'].std()
    # # cos and sin of the azimuth angle
    # df['feature/AoA_Az_x'] = df['AoA_Az'].apply(lambda x: np.cos(x))
    # df['feature/AoA_Az_y'] = df['AoA_Az'].apply(lambda x: np.sin(x))
    # # cos and sin of the elevation angle
    # df['feature/AoA_El_x'] = df['AoA_El'].apply(lambda x: np.cos(x))
    # df['feature/AoA_El_y'] = df['AoA_El'].apply(lambda x: np.sin(x))

    # # df['feature/Distance'] = df['Distance']
    # # df['feature/Az_Departure'] = df['Az_Departure']

    # df['physics/Distance'] = df['Distance']
    # df['physics/Az_Departure'] = df['Az_Departure']
    # # df['physics/Az_Departure_x'] = df['Az_Departure'].apply(lambda x: np.cos(x))
    # # df['physics/Az_Departure_y'] = df['Az_Departure'].apply(lambda x: np.sin(x))
    # df['physics/RSS_1st_Pol'] = df['RSS_1st_Pol']
    # df['physics/RSS_2nd_Pol'] = df['RSS_2nd_Pol']

    # df = df.pivot(index=['X', 'Y', 'Heading'],
    #               columns='Anchor_ID',
    #               values=['feature/RSS_1st_Pol', 'feature/RSS_2nd_Pol', 'physics/Distance'])

    return df


def main():
    logging.basicConfig(level=logging.INFO)

    task_name = 'ble_distance_img'

    df = create_df()
    print(df.head())
    df = preprocess_df(df)
    print(df.head())

    print(df.columns)

    dataset = BLEDatasetDistanceIMG(
        dataframe=df,
        key_columns=[col for col in df.columns if col.startswith('key/')],
        sort_columns=[col for col in df.columns if col.startswith('sort/')],
        feature_columns=[col for col in df.columns if col.startswith('feature/')],
        target_column=[col for col in df.columns if col.startswith('target/')],
        physics_columns=[col for col in df.columns if col.startswith('physics/')]
        )
    print(dataset[0])
    print(dataset[0][0].shape, dataset[0][1].shape)

    develop_dataset, test_dataset = random_split_dataset(dataset, val_split=0.2)

    logging.info('Saving datasets')
    save_data(develop_dataset, os.path.join('datasets', task_name, 'develop_dataset'))
    save_data(test_dataset, os.path.join('datasets', task_name, 'test_dataset'))


if __name__ == "__main__":
    main()
