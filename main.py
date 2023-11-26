# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import wgrib2
import glob
import os
import numpy as np
import pandas as pd

import grib_parser
import globals
import prism_data_parser

# ************************************************************************************* #
# This application requires that we have a file called wget
# wget contains a list of...you guessed it... wget commands
# We download the grib2 files for the region
# Because I'm lazy, we don't have a wget script for the PRISM files. Assume they're a given
# ************************************************************************************* #

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def join_data(prism_station_subset, gfs_daily_min, gfs_daily_max, prism_daily_min, prism_daily_max):
    # Make sure the latitude and longitude columns are in the same format
    # Convert to float if necessary, and round if necessary to match precision
    prism_station_subset['Latitude'] = prism_station_subset['Latitude'].astype(float)
    prism_station_subset['Longitude'] = prism_station_subset['Longitude'].astype(float)

    prism_daily_min['Latitude'] = prism_daily_min['Latitude'].astype(float)
    prism_daily_min['Longitude'] = prism_daily_min['Longitude'].astype(float)

    # Join the dataframes on Latitude and Longitude
    prism_with_elevation_merged_df = pd.merge(prism_daily_min, prism_station_subset,
                                              how='inner',
                                              left_on=['Latitude', 'Longitude'],
                                              right_on=['Latitude', 'Longitude'])

    prism_result = join_on_closest_lat_lon(prism_daily_min, prism_station_subset)
    gfs_result = join_on_closest_lat_lon(gfs_daily_min, prism_with_elevation_merged_df)

    # Check for any NaN values or duplicates and handle them as necessary
    # merged_df.dropna(inplace=True)  # Uncomment this line to drop NaN values
    # merged_df.drop_duplicates(inplace=True)  # Uncomment this line to drop duplicates

    # The merged_df now contains the joined data
    print("Merged all data")


def join_on_closest_lat_lon(A, B):
    # now join to gfs based on closest lat/lon :((((
    # Iterate over each row in dataset A
    closest = pd.DataFrame()
    for index, row in A.iterrows():
        # Calculate the distance to each point in dataset A
        distances = B.apply(
            lambda x: haversine(row['Latitude'], row['Longitude'], x['Latitude'], x['Longitude']),
            axis=1
        )
        # Find the index of the closest point
        closest_idx = distances.idxmin()
        # Append the closest point to the closest DataFrame
        closest = closest.append(B.loc[closest_idx])
    # Reset index for consistency
    closest.reset_index(drop=True, inplace=True)
    # Join the closest coordinates with dataset F
    result = pd.concat([A, closest], axis=1)
    return result


if __name__ == '__main__':
    # Download gribs and filter to our spatial region
    ## only do this if we haven't processed files...
    # grib_parser.download_grib_files('data/wget', globals.latN, globals.latS, globals.lonE, globals.lonW)

    # only aggregate gribs if we haven't already
    dir = os.listdir(f"data/{globals.gfs_daily_min_max}")
    if len(dir) == 0:
        # Fix GFS tmin/tmax aggregation
        grib_parser.aggregate_3hr_tmin_or_tmax('tmax')
        grib_parser.aggregate_3hr_tmin_or_tmax('tmin')

    # Process PRISM files
    dirs = glob.glob(f"data/{globals.prism_processed_files}/prism*")
    if len(dirs) == 0:
        prism_data_parser.process_prism_files('tmin')
        prism_data_parser.process_prism_files('tmax')

    # Create NN INPUT matrix from ** clean ** file
    nn_input = pd.DataFrame({"Latitude", "Longitude", "Date", "elevation", "Fcst_TMin", "Fcst_TMax"})

    # 1. Assign each PRISM point to a GFS D1 forecast
    gfs_daily_min = pd.read_csv(f"data/{globals.gfs_daily_min_max}/tmin_202304.csv")
    gfs_daily_max = pd.read_csv(f"data/{globals.gfs_daily_min_max}/tmax_202304.csv")
    prism_daily_min = pd.read_pickle(f"data/{globals.prism_processed_files}/prism_tmin_202304.p")
    prism_daily_max = pd.read_pickle(f"data/{globals.prism_processed_files}/prism_tmax_202304.p")

    # 2. Assign each PRISM point to its station elevation
    station_file_name = f'data/{globals.prism_processed_files}/station_data.p'
    exists = os.path.isfile(station_file_name)
    prism_stn_data = None
    if not exists:
        prism_stn_data = prism_data_parser.get_station_data()
    else:
        prism_stn_data = pd.read_pickle(station_file_name)
    prism_station_subset = prism_stn_data[["Longitude", "Latitude", "Elevation(m)"]]

    join_data(prism_station_subset, gfs_daily_min, gfs_daily_max, prism_daily_min, prism_daily_max)

    # Divide into training data - train on 1/2 of latitudes, validate on 1/4, test on 1/4 of latitudes

    # Create learning curves
    # Calculate error
