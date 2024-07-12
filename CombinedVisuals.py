import datetime

from osgeo import gdal

import globals

gdal.VersionInfo()
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import interpolate


def visualize_gfs(year, month, day, pickle_file_name, variable):
    df = pd.read_pickle(pickle_file_name)

    # Filter by date
    df_filtered = df
    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_filter = df["Date"] == pd.Timestamp(year, month, day)
        df_filtered = df[date_filter]

    ############### nicer map
    # Create a grid to interpolate the data.
    num_cols, num_rows = 72, 23  # You can adjust the resolution of your grid
    lon_lin = np.linspace(df_filtered['Longitude'].min(), df_filtered['Longitude'].max(), num_cols)
    lat_lin = np.linspace(df_filtered['Latitude'].min(), df_filtered['Latitude'].max(), num_rows)

    # Interpolate the temperature data
    temp_values = df_filtered[variable].values.reshape(len(lat_lin), len(lon_lin))
    interpolator = interpolate.RegularGridInterpolator((lat_lin, lon_lin), temp_values)

    # Define the grid for pcolormesh
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)

    # Interpolate the data
    grid_temp = interpolator((grid_lat, grid_lon))

    # Set up Basemap
    m = Basemap(projection='merc', llcrnrlat=48, urcrnrlat=lat_lin,
                llcrnrlon=-122, urcrnrlon=-119, lat_ts=20, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    x, y = m(grid_lon, grid_lat)
    # Plot using pcolormesh
    plt.pcolormesh(x, y, grid_temp, cmap='hot', shading='auto')
    # Add a colorbar
    return plt


#
def visualize_prism(year, month, day, pickle_file_name, variable_name, stage=None, resolution=None):
    df = pd.read_pickle(pickle_file_name)

    # Filter by date
    df_filtered = df
    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_filter = df["Date"] == pd.Timestamp(year, month, day)
        df_filtered = df[date_filter]
    if "Date_x" in df.columns:
        df['Date_x'] = pd.to_datetime(df['Date_x'])
        date_filter = df["Date_x"] == pd.Timestamp(year, month, day)
        df_filtered = df[date_filter]

    ############### nicer map
    # Create a grid to interpolate the data.
    # validation size is 23760
    # 72 x 23 for full
    # 72 x 6 for validation
    # 72 x 6 for test
    print(df.shape)
    x, y = 0, 0
    if resolution is None:
        y = df.shape[0] / 72
        x = 72
        if stage is None:
            y = 24
        elif stage == "final":
            y = 23
    else:
        x = resolution[0]
        y = resolution[1]
    num_cols, num_rows = x, int(y)

    lat_min, lat_max = [None, None]
    if stage == "test":
        lat_min, lat_max = globals.tst_filter
    elif stage == "train":
        lat_min, lat_max = globals.trn_filter
    elif stage == "validation":
        lat_min, lat_max = globals.val_filter
    else:
        lat_min, lat_max = 48, 49

    lon_lin = np.linspace(-122, -119, num_cols)
    lat_lin = np.linspace(lat_min, lat_max, num_rows)

    # Interpolate the temperature data
    df_filtered = df_filtered.sort_values(by=["Latitude", "Longitude"])
    temp_values = df_filtered[variable_name].values.reshape(len(lat_lin), len(lon_lin))
    interpolator = interpolate.RegularGridInterpolator((lat_lin, lon_lin), temp_values)

    # Define the grid for pcolormesh
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)

    # Interpolate the data
    grid_temp = interpolator((grid_lat, grid_lon))

    # Set up Basemap
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=-122, urcrnrlon=-119, lat_ts=20, resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    x, y = m(grid_lon, grid_lat)

    # Plot using pcolormesh
    plt.pcolormesh(x, y, grid_temp, cmap='hot', shading='auto')
    # plt.show()
    return x, y, grid_temp


# wdir = 'data/model_outputs'
wdir = 'data/model2_outputs'

# tmax
xarr = list()
yarr = list()
grid_arr = list()

x0, y0, grid0 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'tmax_K', stage="test")
x1, y1, grid1 =  visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'pred_tmax', stage="test")
# plt3 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'gfs_tmax', stage="test")
x2, y2, grid2 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'point_error_tmax', stage="test")

# tmin
x3, y3, grid3 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'tmin_K', stage="test")
x4, y4, grid4 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'pred_tmin', stage="test")
# plt3 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'gfs_tmax', stage="test")
x5, y5, grid5 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'point_error_tmin', stage="test")

xarr.append(x0)
xarr.append(x3)

xarr.append(x1)
xarr.append(x4)

xarr.append(x2)
xarr.append(x5)

yarr.append(y0)
yarr.append(y3)

yarr.append(y1)
yarr.append(y4)

yarr.append(y2)
yarr.append(y5)

grid_arr.append(grid0)
grid_arr.append(grid3)

grid_arr.append(grid1)
grid_arr.append(grid4)

grid_arr.append(grid2)
grid_arr.append(grid5)

fig = plt.figure()
# Suppose you want a 2x2 grid of plots
pcm, ax = None, None
for i in range(6):
    ax = fig.add_subplot(3, 2, i + 1)  # Rows, columns, subplot number
        # Plot using pcolormesh
    pcm = plt.pcolormesh(xarr[i], yarr[i], grid_arr[i], cmap='hot', shading='auto')
                         # vmin=-4, vmax=4
    plt.colorbar()

plt.show()
print("wow")