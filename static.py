from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, create_anchors_dataset, load_grid_dataset
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from scipy import stats


# Create dictionary with the cardinal directions and the corresponding labels
cardinal_directions = {
    'east': (0.5, 0.0),
    'north': (0.0, 0.5),
    'south': (0.0, -0.5),
    'west': (-0.5, 0.0)
}


# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Path Loss Exponent Estimation')
    parser.add_argument('--heading', type=str, help='The cardinal direction', required=True, choices=['east', 'north', 'south', 'west'])
    parser.add_argument('--anchor', type=int, help='The ID of the anchor node', required=True, choices=[6501, 6502, 6503, 6504])
    parser.add_argument('--channel', type=int, help='The BLE channel', required=True, choices=[37, 38, 39])
    parser.add_argument('--polarization', type=str, help='The BLE polarization', required=True, choices=['1st', '2nd'])
    parser.add_argument('--positional', action='store_true', help='Group the data by the (X, Y) coordinates instead of the distance')

    return parser.parse_args()


def main():
    # Take the arguments from the command line
    args = parse_args()
    heading = args.heading
    anchor = args.anchor
    channel = args.channel
    polarization = args.polarization

    # Load the static dataset
    data_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_{heading}.txt"
    data = load_raw_dataset(data_file_path)

    # Load the ground truth dataset for static
    gt_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_{heading}.txt"
    gt = load_gt_dataset(gt_file_path)

    df = pd.merge_asof(data, gt, left_on='Epoch_Time', right_on='Start_Time', direction='backward')
    df = df[df['Epoch_Time'] <= df['End_Time']]
    df = df.reset_index(drop=True)

    grid = load_grid_dataset()
    anchors = create_anchors_dataset()

    # Merge the data with the grid and anchors data (grid and anchors data are ground truth data)
    df = pd.merge(df, grid, how='inner', on=['X', 'Y', 'Anchor_ID'])
    df = pd.merge(df, anchors, how='inner', on='Anchor_ID')

    # Convert cm distances to meters
    df['X'] = df['X'] / 100
    df['Y'] = df['Y'] / 100
    df['Anchor_x'] = df['Anchor_x'] / 100
    df['Anchor_y'] = df['Anchor_y'] / 100

    # Compute the euclidean distance between the anchor (Anchor_x, Anchor_y) and the tag (X, Y) coordinates
    df['Distance'] = ((df['X'] - df['Anchor_x'])**2 + (df['Y'] - df['Anchor_y'])**2)**0.5
    
    # Take only the Channel = 37 with the 2nd polarization
    df = df[df['Channel'] == channel]
    # Take only the Anchor_ID = 6501
    df = df[df['Anchor_ID'] == anchor]
    df = df.rename(columns={f'RSS_{polarization}_Pol': 'RSS'})

    if args.positional:
        # Set an ID for each (X, Y) coordinate
        df['ID'] = df.groupby(['X', 'Y']).ngroup()
        # For each ID take the Z score of RSS_2nd_Pol less than 2
        df['zscore'] = df.groupby('ID')['RSS'].transform(lambda x: stats.zscore(x))
        df = df[df['zscore'].abs() < 2]
        df['RSS_mean'] = df.groupby('ID')['RSS'].transform('mean')
    else:
        # For each Distance take the Z score of RSS_2nd_Pol less than 2
        df['zscore'] = df.groupby('Distance')['RSS'].transform(lambda x: stats.zscore(x))
        df = df[df['zscore'].abs() < 2]
        df['RSS_mean'] = df.groupby('Distance')['RSS'].transform('mean')
    
    # sort the data by the Distance
    df = df.sort_values(by='Distance')
    df = df.reset_index(drop=True)

    # Plot the Ground Truth Static Points and Anchors
    plt.figure(num=1)
    plt.scatter(gt['X'] / 100, gt['Y'] / 100, c='blue', marker='o', label='GT Points')
    plt.scatter(anchors[anchors['Anchor_ID'] == anchor]['Anchor_x'] / 100, anchors[anchors['Anchor_ID'] == anchor]['Anchor_y'] / 100, c='red', marker='o', s=100, label='Anchor')
    plt.scatter(anchors[anchors['Anchor_ID'] != anchor]['Anchor_x'] / 100, anchors[anchors['Anchor_ID'] != anchor]['Anchor_y'] / 100, c='red', marker='o', label='Other Anchors')
    # Annotating points with labels
    for i, label in enumerate(anchors['Anchor_ID']):
        plt.text(anchors['Anchor_x'][i] / 100, anchors['Anchor_y'][i] / 100, label, fontsize=10)
    # plot an arrow
    plt.arrow(6, 3, cardinal_directions[f'{heading}'][0], cardinal_directions[f'{heading}'][1], head_width=0.2, head_length=0.2, fc='black', ec='black')
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

    if args.positional:
        # Take only the Distance and RSS_mean columns for distinct (X, Y) coordinates
        df_mini = df.drop_duplicates(subset=['ID'])
    else:
        # Take only the Distance and RSS_mean columns for distinct distances
        df_mini = df.drop_duplicates(subset=['Distance'])

    # Take only the rows with distinct values of Distance
    df_mini = df.drop_duplicates(subset='Distance', keep='first')
    exp10_X = df_mini['Distance'].values
    # Take the logarithm (base 10) of the distances
    X = np.log10(exp10_X).reshape(-1, 1)
    y = df_mini['RSS_mean'].values
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    # From the model:
    # RSS = intercept + slope * log10(distance)
    # slope = -10 * alpha  => alpha = -slope / 10
    alpha = -slope / 10
    RSS_1m = intercept  # If your reference distance d_0 = 1 meter
    print("Estimated path loss exponent (alpha):", alpha)
    print("Estimated RSS at 1 meter (RSS_1m):", RSS_1m)

    # Plot the distances to the anchor vs. the RSS values
    exp10_X_all = df['Distance'].values
    y_all = df['RSS'].values
    plt.figure(num=2)
    plt.scatter(exp10_X_all, y_all, color='blue', marker='.', alpha=0.3, label='RSS values')
    plt.scatter(exp10_X, y, color='red', marker='*', label='Mean RSS values')
    plt.plot(exp10_X, model.predict(X), color='black', label='Log10 Regression Model')
    plt.xlabel('Distance')
    plt.ylabel('RSS')
    plt.title(f'Anchor:{anchor}, Heading:{heading}, Channel:{channel}, Polarization:{polarization}')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
