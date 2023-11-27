import datetime

from osgeo import gdal
gdal.VersionInfo()
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import interpolate

wdir = 'data/merged_data'
# ds = gdal.Open("../sample_files/dem.tiff")
lat_lon_pickle = f'{wdir}/prism_with_elevation_merged_tmin.p'
df = pd.read_pickle(lat_lon_pickle)
# Load the data
df_filtered = df[df["Date_x"] == datetime.date(2023, 4, 1)]
# Create a new figure
fig = plt.figure(figsize=(12, 8))

# Set up the Basemap
# You can adjust the projection and area as needed
m = Basemap(projection='merc', llcrnrlat=48, urcrnrlat=49,
            llcrnrlon=-121, urcrnrlon=-119, lat_ts=20, resolution='c')
# Draw map boundaries and coastlines
m.drawcoastlines()
m.drawstates(linewidth=1)
# m.etopo()

# Convert lat and lon to map projection coordinates
lons, lats = m(df_filtered['Longitude'].values, df_filtered['Latitude'].values)

# Plot the data points
sc = m.scatter(lons, lats, c=df_filtered['tmin'], cmap='hot', marker='o', edgecolor='k', alpha=0.7)
# sc = m.pcolormesh([lons, lats], df_filtered['tmin'], cmap='hot', marker='o', edgecolor='k', alpha=0.7)

# Add a colorbar
cbar = m.colorbar(sc, location='bottom', pad="5%")
cbar.set_label('Temperature')

# Show the plot
plt.title('Temperature Map')
plt.show()
fig.savefig('figs/scatter_1day_tmin.pdf')
plt.cla()

############### nicer map

# Create a grid to interpolate the data.
num_cols, num_rows = 72, 23  # You can adjust the resolution of your grid
lon_lin = np.linspace(df_filtered['Longitude'].min(), df_filtered['Longitude'].max(), num_cols)
lat_lin = np.linspace(df_filtered['Latitude'].min(), df_filtered['Latitude'].max(), num_rows)
lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

# Interpolate the temperature data
temp_values = df_filtered['tmin'].values.reshape(len(lat_lin), len(lon_lin))
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
fig.savefig('figs/heatmap_1day_tmin.png')