from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, create_anchors_dataset, load_grid_dataset
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from scipy import stats


# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Path Loss Exponent Estimation')
    parser.add_argument('--anchor', type=int, help='The ID of the anchor node', required=True, choices=[6501, 6502, 6503, 6504, -1])
    parser.add_argument('--channel', type=int, help='The BLE channel', required=True, choices=[37, 38, 39, -1])
    parser.add_argument('--polarization', type=str, help='The BLE polarization', required=True, choices=['1st', '2nd', 'mean'])
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data removing the outliers')
    parser.add_argument('--positional', action='store_true', help='Group the data by the (X, Y) coordinates instead of the distance')

    return parser.parse_args()


def main():
    # Take the arguments from the command line
    args = parse_args()
    anchor = args.anchor
    channel = args.channel
    polarization = args.polarization

    # Load the calibration dataset
    data_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/calibration/beacons/beacons_calibration.txt"
    data = load_raw_dataset(data_file_path)

    # Load the ground truth dataset for calibration
    gt_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/calibration/gt/gt_calibration.txt"
    gt = load_gt_dataset(gt_file_path)

    df = pd.merge_asof(data, gt, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    df = df[df['Epoch_Time'] <= df['End_Time']]
    df = df.reset_index(drop=True)

    grid = load_grid_dataset()
    anchors = create_anchors_dataset()

    # Merge the data with the grid and anchors data (grid and anchors data are ground truth data)
    df = pd.merge(df, grid, how='inner', on=['X', 'Y', 'Anchor_ID'])
    df = pd.merge(df, anchors, how='inner', on='Anchor_ID')

    # Convert angles to radians
    # df['AoA_Az'] = np.radians(df['AoA_Az'])
    # df['AoA_El'] = np.radians(df['AoA_El'])
    df['Az_Arrival'] = np.radians(df['Az_Arrival'])
    # df['El_Arrival'] = np.radians(df['El_Arrival'])

    # Transform the observed azimuth angle of arrival to the canonical reference system
    # df.loc[df['Anchor_ID'] == 6501, 'AoA_Az'] = - df.loc[df['Anchor_ID'] == 6501, 'AoA_Az']  # Left Anchor (East direction)
    # df.loc[df['Anchor_ID'] == 6502, 'AoA_Az'] = (np.pi / 2) - df.loc[df['Anchor_ID'] == 6502, 'AoA_Az']  # Bottom Anchor (North direction)
    # df.loc[df['Anchor_ID'] == 6503, 'AoA_Az'] = np.pi - df.loc[df['Anchor_ID'] == 6503, 'AoA_Az']  # Right Anchor (West direction)
    # df.loc[df['Anchor_ID'] == 6504, 'AoA_Az'] = - (np.pi / 2) - df.loc[df['Anchor_ID'] == 6504, 'AoA_Az']  # Top Anchor (South direction)

    df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival'] = - df.loc[df['Anchor_ID'] == 6501, 'Az_Arrival']  # Left Anchor (East direction)
    df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival'] = (np.pi / 2) - df.loc[df['Anchor_ID'] == 6502, 'Az_Arrival']  # Bottom Anchor (North direction)
    df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival'] = np.pi - df.loc[df['Anchor_ID'] == 6503, 'Az_Arrival']  # Right Anchor (West direction)
    df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival'] = - (np.pi / 2) - df.loc[df['Anchor_ID'] == 6504, 'Az_Arrival']  # Top Anchor (South direction)
    
    # Take only the data for the selected anchor, channel, and polarization
    if anchor != -1:
        df = df[df['Anchor_ID'] == anchor]
    if channel != -1:
        df = df[df['Channel'] == channel]
    if polarization == 'mean':
        df['RSS'] = (df['RSS_1st_Pol'] + df['RSS_2nd_Pol']) / 2
    else:
        df['RSS'] = df[f'RSS_{polarization}_Pol']

    if args.positional:
        # Set an ID for each (X, Y) coordinate
        df['Pos_ID'] = df.groupby(['X', 'Y']).ngroup()
        if args.preprocess:
            # For each ID take the Z score of RSS_2nd_Pol less than 2
            df['zscore_Az_Arrival'] = df.groupby('RSS')['Az_Arrival'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_Az_Arrival'].abs() < 2]
        df['Az_Arrival_mean'] = df.groupby('Pos_ID')['Az_Arrival'].transform('mean')
    else:
        if args.preprocess:
            df['zscore_Az_Arrival'] = df.groupby('RSS')['Az_Arrival'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_Az_Arrival'].abs() < 2]
        df['Az_Arrival_mean'] = df.groupby('RSS')['Az_Arrival'].transform('mean')

    # Sort the data by the Distance
    df = df.sort_values(by='RSS')
    df = df.reset_index(drop=True)

    # Plot the Ground Truth Static Points and Anchors
    plt.figure(num=1)
    plt.scatter(gt['X'] / 100, gt['Y'] / 100, c='blue', marker='o', label='GT Points')
    plt.scatter(anchors[anchors['Anchor_ID'] == anchor]['Anchor_x'] / 100, anchors[anchors['Anchor_ID'] == anchor]['Anchor_y'] / 100, c='red', marker='o', s=100, label='Anchor')
    plt.scatter(anchors[anchors['Anchor_ID'] != anchor]['Anchor_x'] / 100, anchors[anchors['Anchor_ID'] != anchor]['Anchor_y'] / 100, c='red', marker='o', label='Other Anchors')
    # Annotating points with labels
    for i, label in enumerate(anchors['Anchor_ID']):
        plt.text(anchors['Anchor_x'][i] / 100, anchors['Anchor_y'][i] / 100, label, fontsize=10)
    # plot the room
    plt.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k-')
    plt.xlim(-2, 14)
    plt.ylim(-2, 8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Room Layout')
    plt.grid(True)
    plt.legend()
    plt.show()


    # Take only the Distance and RSS_mean columns for distinct distances
    if args.positional:
        df_mini = df.drop_duplicates(subset=['Pos_ID'])
    else:
        df_mini = df.drop_duplicates(subset=['RSS'])

    X = df_mini['RSS'].values
    X_all = df['RSS'].values
    y = df_mini['Az_Arrival_mean'].values
    y_all = df['Az_Arrival'].values

    plt.figure(num=2)
    plt.scatter(X_all, y_all, color='blue', marker='.', alpha=0.3, label='Azimuth values')
    plt.scatter(X, y, color='red', marker='*', label='Mean Azimuth values')
    plt.xlabel('RSS')
    plt.ylabel('Azimuth')
    plt.title(f'Anchor:{anchor}, Channel:{channel}, Polarization:{polarization}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # exp10_X = df_mini['Distance'].values
    # # Take the logarithm (base 10) of the distances
    # X = np.log10(exp10_X).reshape(-1, 1)
    # y = df_mini['RSS_mean'].values
    # # Fit a linear regression model
    # model = LinearRegression()
    # model.fit(X, y)
    # slope = model.coef_[0]
    # intercept = model.intercept_
    # # From the model:
    # # RSS = intercept + slope * log10(distance)
    # # slope = -10 * alpha  => alpha = -slope / 10
    # alpha = -slope / 10
    # RSS_1m = intercept  # If your reference distance d_0 = 1 meter
    # print("Estimated path loss exponent (alpha):", alpha)
    # print("Estimated RSS at 1 meter (RSS_1m):", RSS_1m)

    # Plot the distances to the anchor vs. the RSS values
    # exp10_X_all = df['Distance'].values
    # y_all = df['RSS'].values
    # plt.figure(num=2)
    # plt.scatter(exp10_X_all, y_all, color='blue', marker='.', alpha=0.3, label='RSS values')
    # plt.scatter(exp10_X, y, color='red', marker='*', label='Mean RSS values')
    # plt.plot(exp10_X, model.predict(X), color='black', label='Log10 Regression Model')
    # plt.xlabel('Distance')
    # plt.ylabel('RSS')
    # plt.title(f'Anchor:{anchor}, Channel:{channel}, Polarization:{polarization}')
    # plt.grid(True)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
