from data_storage.Dataset_AoA_RSS_BLE51.data_utils import load_raw_dataset, load_gt_dataset, create_anchors_dataset
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse

# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Path Loss Exponent Estimation')
    parser.add_argument('--heading', type=str, help='The cardinal direction', required=True, choices=['east', 'north', 'south', 'west'])
    parser.add_argument('--anchor', type=int, help='The ID of the anchor node', required=True, choices=[6501, 6502, 6503, 6504])
    parser.add_argument('--channel', type=int, help='The BLE channel', required=True, choices=[37, 38, 39])
    parser.add_argument('--polarization', type=str, help='The BLE polarization', required=True, choices=['1st', '2nd'])

    return parser.parse_args()


def main():
    # Take the arguments from the command line
    args = parse_args()
    heading = args.heading
    anchor = args.anchor
    channel = args.channel
    polarization = args.polarization

    # Load the static dataset
    static_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/static/beacons/beacons_static_{heading}.txt"
    static_df = load_raw_dataset(static_file_path)
    print(f"\nStatic Data with {heading} heading:")
    print(static_df)

    # Load the ground truth dataset for static
    gt_static_file_path = f"data_storage/Dataset_AoA_RSS_BLE51/static/gt/gt_static_{heading}.txt"
    gt_static_df = load_gt_dataset(gt_static_file_path)
    # cm to m
    gt_static_df['X'] = gt_static_df['X'] / 100
    gt_static_df['Y'] = gt_static_df['Y'] / 100
    print("\nGround Truth Static Data:")
    print(gt_static_df)

    # Create a dataset with the anchor nodes' information
    anchors_df = create_anchors_dataset()
    # cm to m
    anchors_df['Anchor_x'] = anchors_df['Anchor_x'] / 100
    anchors_df['Anchor_y'] = anchors_df['Anchor_y'] / 100
    print("\nAnchors Data:")
    print(anchors_df)

    # Filter static data
    static_df_filtered = static_df[static_df['Anchor_ID'] == anchor].reset_index(drop=True)
    print(f"Static Data for {anchor}:")
    print(static_df_filtered)

    # Plot the Ground Truth Static Points and Anchors
    plt.figure(num=1)
    plt.scatter(gt_static_df['X'], gt_static_df['Y'], c='blue', marker='o', label='GT Points')
    plt.scatter(anchors_df['Anchor_x'], anchors_df['Anchor_y'], c='red', marker='o', s=100, label='Anchors')
    # Annotating points with labels
    for i, label in enumerate(anchors_df['Anchor_ID']):
        plt.text(anchors_df['Anchor_x'][i], anchors_df['Anchor_y'][i], label, fontsize=10)
    # plot the room
    plt.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k-')
    plt.xlim(-2, 14)
    plt.ylim(-2, 8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ground Truth Static Points')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create an IntervalIndex from the start/end times
    intervals = pd.IntervalIndex.from_arrays(gt_static_df['Start_Time'], gt_static_df['End_Time'], closed='both')  
    # For each RSS timestamp, find which interval it falls into
    interval_positions = intervals.get_indexer(static_df_filtered['Epoch_Time'])
    static_df_filtered['GT_ID'] = interval_positions
    static_df_filtered = static_df_filtered[static_df_filtered['GT_ID'] != -1]
    static_df_filtered = static_df_filtered.reset_index(drop=True)
    print(f"\nStatic Data for {anchor} with GT_ID:")
    print(static_df_filtered)

    # The best is the Channel = 37 with the 2nd polarization
    rss_df = static_df_filtered[static_df_filtered['Channel'] == channel]
    rss_df = rss_df[[f'RSS_{polarization}_Pol', 'GT_ID']]
    rss_df = rss_df.rename(columns={f'RSS_{polarization}_Pol': 'RSS'})
    # Drop the RSS values that have Z scores greater than 2
    rss_df = rss_df[np.abs(rss_df['RSS'] - rss_df['RSS'].mean()) <= (2 * rss_df['RSS'].std())]
    # Group by GT_ID and compute mean and list aggregation
    rss_df = rss_df.groupby('GT_ID')['RSS'].agg(Mean_RSS='mean', RSS_List=list).reset_index()
    print("\nRSS Data:")
    print(rss_df)

    anchors_df_filtered = anchors_df[anchors_df['Anchor_ID'] == anchor].reset_index(drop=True)
    # Extract the points as Nx2 and Mx2 arrays
    anchors_points = anchors_df_filtered[['Anchor_x', 'Anchor_y']].values  # Shape (M, 2)
    gt_points = gt_static_df[['X','Y']].values  # Shape (N, 2)
    # Compute the distance matrix between each pair of points in the anchors dataset and the ground truth dataset
    distance_matrix = cdist(anchors_points, gt_points, metric='euclidean')
    # Set new columns in gt_static_df with the distances to each anchor
    for i in range(distance_matrix.shape[0]):
        gt_static_df[f"Distance_to_{anchors_df_filtered['Anchor_ID'][i]}"] = distance_matrix[i, :]
    # Print a preview of the ground truth dataset for static with the distances to each anchor
    print("\nGround Truth Static Data with Distances:")
    print(gt_static_df)

    # Merge the RSS data with the ground truth static data
    cal_df = gt_static_df.merge(rss_df, left_index=True, right_on='GT_ID')
    cal_df = cal_df[['GT_ID', f'Distance_to_{anchor}', 'Mean_RSS', 'RSS_List']]
    # Sort the static data by distance to the anchor
    cal_df = cal_df.sort_values(by=f'Distance_to_{anchor}')
    cal_df = cal_df.reset_index(drop=True)
    print("\nGround Truth Static Data with Distances sorted and RSS:")
    print(cal_df)

    # Take the logarithm (base 10) of the distances
    exp10_X = cal_df[f'Distance_to_{anchor}'].values.reshape(-1, 1)
    X = np.log10(exp10_X)
    y = cal_df['Mean_RSS'].values
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
    plt.figure(num=2)
    # Plot the RSS list values
    all_rss_df = cal_df.explode('RSS_List')
    exp10_XX = all_rss_df[f'Distance_to_{anchor}'].values
    z = all_rss_df['RSS_List'].values
    plt.scatter(exp10_XX, z, color='blue', marker='.', alpha=0.3, label='RSS values')
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
