import pandas as pd
import numpy as np
from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, load_grid_dataset, create_anchors_dataset
from Pinnoloc.data.ble_dataset import BLEDataset
from Pinnoloc.utils.saving import save_data
from Pinnoloc.utils.experiments import save_dict_to_yaml
import logging
import os
from scipy import stats
import argparse


def create_df(task):
    
    def load_data(task):
        folder_0 = task.split('_')[0]
        data = load_raw_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/{folder_0}/beacons/beacons_{task}.txt')
        return data

    def load_gt(task):
        folder_0 = task.split('_')[0]
        gt = load_gt_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/{folder_0}/gt/gt_{task}.txt')
        return gt
    
    if task == 'static_all':
        task_list = ['static_east', 'static_north', 'static_south', 'static_west']
        gt_list = [load_gt(task) for task in task_list]
        data_list = [load_data(task) for task in task_list]
        gt = pd.concat(gt_list)
        data = pd.concat(data_list)
        gt = gt.sort_values(by='Start_Time')
        data = data.sort_values(by='Epoch_Time')
    else: 
        gt = load_gt(task)
        data = load_data(task)

    # Merge the data with the ground truth data
    df = pd.merge_asof(data, gt, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    # Filter the rows where the epoch time is within the interval [Start_Time, End_Time]
    df = df[df['Epoch_Time'] <= df['End_Time']]
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


def preprocess_df(df, channel, polarization, preprocess, buffer):
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

    # df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival'] = - df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival']  # Left Anchor (East direction)
    # df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival'] = (np.pi / 2) - df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival']  # Bottom Anchor (North direction)
    # df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival'] = np.pi - df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival']  # Right Anchor (West direction)
    # df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival'] = - (np.pi / 2) - df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival']  # Top Anchor (South direction)

    # df['AoA_Az'] = np.arctan2(np.sin(df['AoA_Az']), np.cos(df['AoA_Az']))  # set the angle between -pi and pi
    # df['Az_Arrival'] = np.arctan2(np.sin(df['Az_Arrival']), np.cos(df['Az_Arrival']))  # set the angle between -pi and pi

    df['AoA_Az_x'] = np.cos(df['AoA_Az'])
    df['AoA_Az_y'] = np.sin(df['AoA_Az'])

    if channel != -1:
        df = df[df['Channel'] == channel]
    if polarization == 'mean':
        df['RSS'] = (df['RSS_1st_Pol'] + df['RSS_2nd_Pol']) / 2
    else:
        df['RSS'] = df[f'RSS_{polarization}_Pol']

    # pause = input("\nRSS for position (1.2, 1.2). Press Enter to continue...")
    # df_bc1 = df.loc[(df['X'] == 1.2) & (df['Y'] == 1.2), ['Anchor_ID', 'Az_Arrival', 'AoA_Az', 'RSS']]
    # df_bc1 = df_bc1.groupby(['Anchor_ID', 'Az_Arrival']).agg({'AoA_Az': 'mean', 'RSS': 'mean'}).reset_index()
    # print(df_bc1)

    # pause = input("\nRSS for position (10.8, 1.2). Press Enter to continue...")
    # df_bc2 = df.loc[(df['X'] == 10.8) & (df['Y'] == 1.2), ['Anchor_ID', 'Az_Arrival', 'AoA_Az', 'RSS']]
    # df_bc2 = df_bc2.groupby(['Anchor_ID', 'Az_Arrival']).agg({'AoA_Az': 'mean', 'RSS': 'mean'}).reset_index()
    # print(df_bc2)

    # pause = input("\nRSS for position (10.8, 4.8). Press Enter to continue...")
    # df_bc3 = df.loc[(df['X'] == 10.8) & (df['Y'] == 4.8), ['Anchor_ID', 'Az_Arrival', 'AoA_Az', 'RSS']]
    # df_bc3 = df_bc3.groupby(['Anchor_ID', 'Az_Arrival']).agg({'AoA_Az': 'mean', 'RSS': 'mean'}).reset_index()
    # print(df_bc3)

    # pause = input("\nRSS for position (1.2, 4.8). Press Enter to continue...")
    # df_bc4 = df.loc[(df['X'] == 1.2) & (df['Y'] == 4.8), ['Anchor_ID', 'Az_Arrival', 'AoA_Az', 'RSS']]
    # df_bc4 = df_bc4.groupby(['Anchor_ID', 'Az_Arrival']).agg({'AoA_Az': 'mean', 'RSS': 'mean'}).reset_index()
    # print(df_bc4)

    pause = input("\nPress Enter to continue...")
    if preprocess:
        # For each Distance take the Z score of RSS less than 2
        df['Distance'] = ((df['X'] - df['Anchor_x'])**2 + (df['Y'] - df['Anchor_y'])**2)**0.5
        df['zscore_RSS'] = df.groupby(['Anchor_ID', 'Distance'])['RSS'].transform(lambda x: stats.zscore(x))
        df = df[df['zscore_RSS'].abs() < 2]
        # df['zscore_AoA_Az'] = df.groupby(['Anchor_ID', 'X', 'Y'])['AoA_Az'].transform(lambda x: stats.zscore(x))
        # df = df[df['zscore_AoA_Az'].abs() < 2]
        df['zscore_AoA_Az_x'] = df.groupby(['Anchor_ID', 'Distance'])['AoA_Az_x'].transform(lambda x: stats.zscore(x))
        df = df[df['zscore_AoA_Az_x'].abs() < 2]
        df['zscore_AoA_Az_y'] = df.groupby(['Anchor_ID', 'Distance'])['AoA_Az_y'].transform(lambda x: stats.zscore(x))
        df = df[df['zscore_AoA_Az_y'].abs() < 2]

    print(df)
    pause = input("Press Enter to continue...")

    df["Epoch_Time_Buffer"] = (df["Epoch_Time"] // buffer).astype(int)

    grouped = (
        df.groupby(["Epoch_Time_Buffer", "X", "Y", "Anchor_ID"], as_index=False)
        .agg({
          "RSS": "last", # "RSS": lambda x: x.mode().iloc[0],  # Take first mode if tie
          # "AoA_Az": "last"
          "AoA_Az_x": "last",
          "AoA_Az_y": "last"
      })
    )

    df_wide = grouped.pivot(
    index=["Epoch_Time_Buffer", "X", "Y"], 
    columns="Anchor_ID", 
    # values=["RSS", "AoA_Az"]
    values=["RSS", "AoA_Az_x", "AoA_Az_y"]
    )
    print(df_wide)
    pause = input("Press Enter to continue...")

    df_wide = (
        df_wide
        .sort_values(by="Epoch_Time_Buffer")
        .groupby(["X", "Y"], group_keys=False)
        .apply(lambda g: g.ffill().bfill())
    )
    print(df_wide)
    pause = input("Press Enter to continue...")

    # Reset the index to have 'X' and 'Y' as columns
    df_wide = df_wide.reset_index()
    print(df_wide)
    pause = input("Press Enter to continue...")

    return df_wide


# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Build BLE Position dataset')
    parser.add_argument('--task', type=str, help='The cardinal direction', required=True,
                        choices=['calibration',
                                 'static_east', 'static_north', 'static_south', 'static_west',
                                 'static_all'])
    parser.add_argument('--channel', type=int, help='The BLE channel', required=True, choices=[37, 38, 39, -1])
    parser.add_argument('--polarization', type=str, help='The BLE polarization', required=True, choices=['1st', '2nd', 'mean'])
    parser.add_argument('--buffer', type=int, help='The buffer time in milliseconds (ms)', required=True)
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data removing the outliers')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Take the arguments from the command line
    args = parse_args()
    channel = args.channel
    polarization = args.polarization
    task = args.task
    preprocess = args.preprocess
    buffer = args.buffer

    task_name = f'ble_position_{task}'

    logging.info(f"Creating BLE Position Dataset for task: {task}")

    df = create_df(task)
    print(df)
    df = preprocess_df(df, channel=channel, polarization=polarization, preprocess=preprocess, buffer=buffer)
    print(df)

    # feature_columns = ['RSS', 'AoA_Az']
    feature_columns = ['RSS', 'AoA_Az_x', 'AoA_Az_y']
    target_column = ['X', 'Y']
    dataset = BLEDataset(
        dataframe=df,
        feature_columns=feature_columns,
        target_column=target_column
        )
    
    print('(', feature_columns, '), ', '(', target_column, ')')
    print(dataset[0])

    logging.info('Saving dataset')
    save_data(dataset, os.path.join('datasets', task_name, 'full_dataset'))
    save_dict_to_yaml(vars(args), os.path.join('datasets', task_name, 'dataset_args.yaml'))


if __name__ == "__main__":
    main()
