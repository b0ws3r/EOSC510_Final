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
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
    # Interpolate the temperature data
    temp_values = df_filtered[variable].values.reshape(len(lat_lin), len(lon_lin))
    interpolator = interpolate.RegularGridInterpolator((lat_lin, lon_lin), temp_values)
    # Define the grid for pcolormesh
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
    # Interpolate the data
    grid_temp = interpolator((grid_lat, grid_lon))
    # Set up Basemap
    fig = plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=48, urcrnrlat=49,
                llcrnrlon=-122, urcrnrlon=-119, lat_ts=20, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    x, y = m(grid_lon, grid_lat)
    # Plot using pcolormesh
    plt.pcolormesh(x, y, grid_temp, cmap='hot', shading='auto')
    # Add a colorbar
    plt.colorbar(label='Temperature')
    plt.title('Single day TMin over N central Washington (observed)')
    plt.show()
    fig.savefig(f'figs/gfs_heatmap_1day_{variable}.png')


#
def visualize_prism(year, month, day, pickle_file_name, variable_name, stage = None, resolution = None):
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

    lat_min, lat_max = [None,None]
    if stage=="test":
        lat_min, lat_max = globals.tst_filter
    elif stage=="train":
        lat_min, lat_max = globals.trn_filter
    elif stage == "validation":
        lat_min, lat_max = globals.val_filter
    else:
        lat_min, lat_max = 48, 49

    lon_lin = np.linspace(-122,-119, num_cols)
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

    fig = plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=-122, urcrnrlon=-119, lat_ts=20, resolution='c')
               
    m.drawcoastlines()
    m.drawcountries()
    x, y = m(grid_lon, grid_lat)
    # Plot using pcolormesh
    plt.pcolormesh(x, y, grid_temp, cmap='hot', shading='auto')
    # Add a colorbar
    plt.colorbar(label='Temperature')
    plt.title(f'Single day {variable_name} over N central Washington')
    plt.show()

    file_name = f'figs/heatmap_1day_{variable_name}_{stage}.png'
    if stage is None:
        file_name = f'figs/heatmap_1day_all_{variable_name}.png'
    fig.savefig(file_name)
    return plt

wdir = 'data/model_outputs'

# # just one day (observed WHOLE map)
# prism_dir = 'data/prism_pp'
# visualize_prism(2023, 4, 15, f'{prism_dir}/prism_tmin_202304.p', 'tmin_K')
#
# ## validation results
# # tmin
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'tmin_K', stage="validation")
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'pred_tmin', stage="validation")
# #
# # # tmax
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'tmax_K', stage="validation")
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'pred_tmax', stage="validation")
# #
# # # point errors
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'point_error_tmin', stage="validation")
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'point_error_tmax', stage="validation")
#
# # GFS pred
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'gfs_tmin', stage="validation")
# visualize_prism(2023, 4, 15, f'{wdir}/validation_data_and_predictions.p', 'gfs_tmax', stage="validation")
# # larger grid (currently not working due to corrupted pickles)
# gfs_dir = 'data/merged_data'
#
# # todo
# # visualize_gfs(2023, 4, 15, f'{gfs_dir}/gfs_result_tmin.p', 'tmin')
#
# # test results
# # tmax
# plt1 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'tmax_K', stage="test")
# plt2 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'pred_tmax', stage="test")
# plt3 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'gfs_tmax', stage="test")
# plt4 = visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'point_error_tmax', stage="test")
#
# # plt.figure(1)
# # plt.subplot(211)
# # plt.plot(ax=plt1)
# # plt.subplot(212)
# # plt.plot(ax=plt2)
# # plt.show()
# # # tmin
# visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'tmin_K', stage="test")
# visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'pred_tmin', stage="test")
# visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'point_error_tmin', stage="test")
#
# # ALL results
# # tmax
visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'tmax_K',  resolution=[36, 12])
visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'pred_tmax',  resolution=[36, 12])
visualize_prism(2023, 4, 15, f'{wdir}/test_data_and_predictions_all.p', 'point_error_tmax',  resolution=[36, 12])

# tmin
visualize_prism(2023, 4, 15, f'{wdir}/all_data_and_predictions_all.p', 'tmin_K', stage="final",   resolution=[36, 12])
visualize_prism(2023, 4, 15, f'{wdir}/all_data_and_predictions_all.p', 'pred_tmin', stage="final",   resolution=[36, 12])
visualize_prism(2023, 4, 15, f'{wdir}/all_data_and_predictions_all.p', 'point_error_tmin', stage="final",   resolution=[36, 12])
