import glob
import pickle

import numpy as np
import pandas as pd


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
    prism_station_subset['Latitude'] = prism_station_subset['Latitude'].round(decimals=5)
    prism_station_subset['Longitude'] = prism_station_subset['Longitude'].round(decimals=5)
    prism_daily_min['Latitude'] = prism_daily_min['Latitude'].round(decimals=5)
    prism_daily_min['Longitude'] = prism_daily_min['Longitude'].round(decimals=5)#.astype(float)
    prism_daily_max['Latitude'] = prism_daily_max['Latitude'].round(decimals=5)
    prism_daily_max['Longitude'] = prism_daily_max['Longitude'].round(decimals=5)#.astype(float)

    # Join the dataframes on Latitude and Longitude
    tmin_elevation_joined_filename = 'data/merged_data/prism_with_elevation_merged_tmin.p'
    dirs = glob.glob(tmin_elevation_joined_filename)
    prism_with_elevation_merged_df = None
    if(len(dirs)== 0):
        prism_with_elevation_merged_df = prism_daily_min.merge(prism_station_subset,
                                              how='inner',
                                              on=['Latitude', 'Longitude'])
                                              #  right_on=['Latitude', 'Longitude'])
        pickle.dump(prism_with_elevation_merged_df, open(tmin_elevation_joined_filename, "wb"))
    else:
        prism_with_elevation_merged_df = pd.read_pickle(tmin_elevation_joined_filename)

    # we have tmin and elevation, now add tmax
    prism_with_elevation_merged_df = prism_with_elevation_merged_df.drop(['Date_y'], axis=1)
    prism_with_elevation_merged_df = prism_with_elevation_merged_df.rename(columns={"Date_x":"Date"})
    prism_with_elevation_merged_df = prism_with_elevation_merged_df.merge(prism_daily_max,
                                              how='inner',
                                              on=['Latitude', 'Longitude', 'Date'])

    pickle.dump(prism_with_elevation_merged_df, open("data/merged_data/prism_with_elevation_merged_tmin_tmax.p", "wb"))

    # now add gfs tmin
    gfs_prism_joined_tmin_filename = 'data/merged_data/gfs_result_tmin.p'
    dirs = glob.glob(gfs_prism_joined_tmin_filename)
    prism_gfs_tmin = None
    if len(dirs) == 0:
        prism_gfs_tmin = join_on_closest_lat_lon(prism_with_elevation_merged_df, gfs_daily_min)
        pickle.dump(prism_gfs_tmin, open(gfs_prism_joined_tmin_filename, "wb"))
    else:
        prism_gfs_tmin = pd.read_pickle(gfs_prism_joined_tmin_filename)

    # rename lat/lon of gfs
    dupe_cols = prism_gfs_tmin.columns.duplicated()
    indexes = np.where(dupe_cols == True)

    prism_split1 = prism_gfs_tmin.iloc[:,:8]
    prism_split2 = prism_gfs_tmin.iloc[:,10:]

    prism_split2 = prism_split2.rename(columns={"Longitude": "gfs_lon"})
    prism_split2 = prism_split2.rename(columns={"Latitude": "gfs_lat"})
    prism_split2 = prism_split2.rename(columns={"Temp": "gfs_tmin"})
    # prism_split2 = prism_split2.drop(index=0, axis=1)

    cleaned_prism_gfs_tmin = prism_split1
    cleaned_prism_gfs_tmin["gfs_lon"]= prism_split2["gfs_lon"]
    cleaned_prism_gfs_tmin["gfs_lat"] = prism_split2["gfs_lat"]
    cleaned_prism_gfs_tmin["gfs_tmin"] = prism_split2["gfs_tmin"]

    # clean dates
    gfs_daily_max['Date'] = pd.to_datetime(gfs_daily_max['Date'])
    cleaned_prism_gfs_tmin['Date'] = pd.to_datetime(cleaned_prism_gfs_tmin['Date'])

    # clean floats
    cleaned_prism_gfs_tmin['Latitude'] = cleaned_prism_gfs_tmin['Latitude'].round(decimals=2)
    gfs_daily_max['Latitude'] = gfs_daily_max['Latitude'].round(decimals=2)
    gfs_daily_max['Longitude'] = gfs_daily_max['Longitude'].round(decimals=2)
    cleaned_prism_gfs_tmin['Latitude'] = cleaned_prism_gfs_tmin['Latitude'].round(decimals=2)
    cleaned_prism_gfs_tmin['Longitude'] = cleaned_prism_gfs_tmin['Longitude'].round(decimals=2)

    # now join to gfs tmax forecast data
    cleaned_prism_gfs_all = cleaned_prism_gfs_tmin.merge(gfs_daily_max,
                                              how='inner',
                                              left_on=['gfs_lon', 'gfs_lat', 'Date'],
                                              right_on=['Longitude', 'Latitude', 'Date'])
    cleaned_prism_gfs_all = cleaned_prism_gfs_all.drop(columns=['Longitude_y', 'Latitude_y', 'Unnamed: 0'])
    cleaned_prism_gfs_all = cleaned_prism_gfs_all.rename(columns={"Temp": "gfs_tmax"})
    cleaned_prism_gfs_all = cleaned_prism_gfs_all.rename(columns={'Longitude_x':'Longitude', 'Latitude_x': 'Latitude'})
    pickle.dump(cleaned_prism_gfs_all, open("data/merged_data/prism_gfs_all_columns_result.p", "wb"))

    return cleaned_prism_gfs_all


def join_on_closest_lat_lon(A, B):
    # now join to gfs based on closest lat/lon :((((
    # Iterate over each row in dataset A
    closest = pd.DataFrame()
    for index, row in A.iterrows():
        # Calculate the distance to each point in dataset A
        date_ = row['Date']
        B_same_date = B[B['Date'].astype(str) == str(date_)]
        distances = B_same_date.apply(
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
