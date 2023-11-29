# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import wgrib2
import datetime
import glob
import os
import numpy as np
import pandas as pd
import pickle
import grib_parser
import globals
import prism_data_parser
from data_merger import join_data
import Visuals

# ************************************************************************************* #
# This application requires that we have a file called wget
# wget contains a list of...you guessed it... wget commands
# We download the grib2 files for the region
# Because I'm lazy, we don't have a wget script for the PRISM files. Assume they're a given
# ************************************************************************************* #


def split_mlp_train_validation_data(merged_data):
    global x_train, y_train
    trn_filter = merged_data["Latitude"].astype(float) < (48.5)
    val_filter_S = merged_data["Latitude"] >= 48.5
    val_filter_N = merged_data["Latitude"] < 48.75
    tst_filter_S = merged_data["Latitude"] >= 48.75
    tst_filter_N = merged_data["Latitude"] <= 49
    # set training
    train = merged_data[trn_filter]
    x_train = train[["Latitude", "Longitude", "us", "gfs_tmin", "gfs_tmax"]]
    y_train = train[["tmin_K", "tmax_K"]]
    print(x_train.loc[1:2, :])
    # validation is what we don't pick for training
    val = merged_data[val_filter_N]
    val = val[val_filter_S]
    # pick some day to validate on
    validation_date = datetime.date(2023, 4, 15)  # because why not
    x_val = val[["Latitude", "Longitude", "us", "gfs_tmin", "gfs_tmax", "Date"]]
    x_val['Date'] = pd.to_datetime(x_val['Date'])
    date_filter = x_val["Date"] == pd.Timestamp(2023, 4, 15)
    x_val = x_val[date_filter]
    print(x_val.head(3))
    y_val = val[["tmin_K", "tmax_K", "Date"]]
    y_val = y_val[y_val["Date"] == validation_date]
    y_val = y_val[["tmin_K", "tmax_K", ]]
    print(y_val.head(3))
    # standardize
    vars = ["gfs_tmin", "gfs_tmax", "us"]
    for var in vars:
        x_mean = x_train[var].mean()
        x_std = np.std(x_train[var])
        x_train[var] = (x_train[var] - x_mean) / x_std
        x_val[var] = (x_val[var] - x_mean) / x_std
    y_vars = ["tmin_K", "tmax_K"]
    for var in y_vars:
        y_mean = y_train[var].mean()
        y_std = np.std(y_train[var])
        y_train[var] = (y_train[var] - y_mean) / y_std
        y_val[var] = (y_val[var] - y_mean) / y_std

    pickle.dump(x_train, open("data/merged_data/x_train.p", "wb"))
    pickle.dump(y_train, open("data/merged_data/y_train.p", "wb"))
    pickle.dump(x_train, open("data/merged_data/x_val.p", "wb"))
    pickle.dump(y_train, open("data/merged_data/y_val.p", "wb"))

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
        prism_data_parser.process_prism_files('us')

    # Create NN INPUT matrix from ** clean ** file
    nn_input = pd.DataFrame({"Latitude", "Longitude", "Date", "elevation", "Fcst_TMin", "Fcst_TMax"})

    # 1. Assign each PRISM point to a GFS D1 forecast
    gfs_daily_min = pd.read_csv(f"data/{globals.gfs_daily_min_max}/tmin_202304.csv")
    gfs_daily_max = pd.read_csv(f"data/{globals.gfs_daily_min_max}/tmax_202304.csv")
    prism_daily_min = pd.read_pickle(f"data/{globals.prism_processed_files}/prism_tmin_202304.p")
    prism_daily_max = pd.read_pickle(f"data/{globals.prism_processed_files}/prism_tmax_202304.p")

    # 2. Assign each PRISM point to its station elevation
    station_file_name = f'data/{globals.prism_processed_files}/prism_us_202304.p'
    exists = os.path.isfile(station_file_name)
    prism_stn_data = None
    if not exists:
        prism_stn_data = prism_data_parser.get_station_data()
    else:
        prism_stn_data = pd.read_pickle(station_file_name)

    # Divide into training data - train on 1/2 of latitudes, validate on 1/4, test on 1/4 of latitudes
    merged_file_name = "prism_gfs_all_columns_result.p"
    dirs = glob.glob(f"data/merged_data/{merged_file_name}")
    merged_data = None
    if len(dirs) == 0:
        merged_data = join_data(prism_stn_data, gfs_daily_min, gfs_daily_max, prism_daily_min, prism_daily_max)
    else:
        merged_data = pd.read_pickle(f"data/merged_data/{merged_file_name}")

    # Create learning curves
    # Calculate error
    Visuals.visualize_prism(2023, 4, 15, '')
    Visuals.visualize_gfs(2023, 4, 15, '')