from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, create_anchors_dataset, load_grid_dataset
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from scipy import stats


cardinal_arrows = {
    'east': (1.0, 0.0),
    'north': (0.0, 1.0),
    'south': (0.0, -1.0),
    'west': (-1.0, 0.0)
}

# Create dictionary with the cardinal directions and the corresponding labels
cardinal_directions = {
    'east': 0,
    'north': 1,
    'south': 2,
    'west': 3
}

anchor_positions = {
    '6501': (0.0, 3.0),
    '6502': (6.0, 0.0),
    '6503': (12.0, 3.0),
    '6504': (6.0, 6.0)
}


def create_df(heading):

    def load_gt(heading):
        gt = load_gt_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_{heading}.txt')
        gt['Heading'] = cardinal_directions[heading]
        return gt


    def load_data(heading):
        data = load_raw_dataset(f'data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_{heading}.txt')
        return data
    
    gt = load_gt(heading)
    data = load_data(heading)
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


# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Path Loss Exponent Estimation')
    parser.add_argument('--anchor', type=int, help='The ID of the anchor node', required=True, choices=[6501, 6502, 6503, 6504, -1])
    parser.add_argument('--channel', type=int, help='The BLE channel', required=True, choices=[37, 38, 39, -1])
    parser.add_argument('--polarization', type=str, help='The BLE polarization', required=True, choices=['1st', '2nd', 'mean'])
    parser.add_argument('--heading', type=str, help='The cardinal direction', required=True, choices=['east', 'north', 'south', 'west'])
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data removing the outliers')
    parser.add_argument('--positional', action='store_true', help='Group the data by the (X, Y) coordinates instead of the distance')

    return parser.parse_args()


def main():
    # Take the arguments from the command line
    args = parse_args()
    heading = args.heading
    anchor = args.anchor
    channel = args.channel
    polarization = args.polarization

    df = create_df(heading)

    # Convert cm distances to meters
    df['X'] = df['X'] / 100
    df['Y'] = df['Y'] / 100
    df['Anchor_x'] = df['Anchor_x'] / 100
    df['Anchor_y'] = df['Anchor_y'] / 100

    # Compute the euclidean distance between the anchor (Anchor_x, Anchor_y) and the tag (X, Y) coordinates
    df['Distance'] = ((df['X'] - df['Anchor_x'])**2 + (df['Y'] - df['Anchor_y'])**2)**0.5
    
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
            df['zscore_RSS'] = df.groupby('Pos_ID')['RSS'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_RSS'].abs() < 2]
            df['zscore_AoA_Az'] = df.groupby('Pos_ID')['AoA_Az'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_AoA_Az'].abs() < 2]
        df['RSS_mean'] = df.groupby('Pos_ID')['RSS'].transform('mean')
    else:
        if args.preprocess:
            df['zscore_RSS'] = df.groupby('Distance')['RSS'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_RSS'].abs() < 2]
            df['zscore_AoA_Az'] = df.groupby(['X', 'Y'])['AoA_Az'].transform(lambda x: stats.zscore(x))
            df = df[df['zscore_AoA_Az'].abs() < 2]
        df['RSS_mean'] = df.groupby('Distance')['RSS'].transform('mean')
    
    # sort the data by the Distance
    df = df.sort_values(by='Distance')
    df = df.reset_index(drop=True)

    # Plot the Ground Truth Static Points and Anchors
    plt.figure(num=1)
    plt.scatter(df['X'], df['Y'], c='blue', marker='o', label='GT Points')
    plt.scatter(anchor_positions[str(anchor)][0], anchor_positions[str(anchor)][1], c='red', marker='s', s=100, label='Anchor')
    # other anchors
    xy = [(x, y) for x, y in anchor_positions.values() if x != anchor_positions[str(anchor)][0] or y != anchor_positions[str(anchor)][1]]        
    plt.scatter([x for x, y in xy], [y for x, y in xy], c='red', marker='s', label='Other Anchors')
    # Annotating anchors with labels
    for anchor_id, (x, y) in anchor_positions.items():
        plt.text(x, y, anchor_id, fontsize=10)
    if heading != 'all': 
        # plot an arrow
        plt.arrow(5.0, 7.5, cardinal_arrows[f'{heading}'][0], cardinal_arrows[f'{heading}'][1], head_width=0.3, head_length=0.3, fc='black', ec='black')
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
        df_mini = df.drop_duplicates(subset=['Pos_ID'])
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
    plt.title(f'Anchor:{anchor}, Channel:{channel}, Polarization:{polarization}, Heading:{heading}')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
