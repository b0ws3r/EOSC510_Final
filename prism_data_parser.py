import datetime

import numpy as np
import pandas as pd
import rasterio
import pickle
import glob
import globals


def process_prism_files(variable):
    prism_files = glob.glob(f'data/prism/PRISM_{variable}*.bil')
    frames = list()
    # Iterate over each pixel
    for prism_file in prism_files:
        with (rasterio.open(prism_file) as dataset):
            # Read the entire array
            data = dataset.read(1)
            # Get geographic information
            transform = dataset.transform
            degs_lon = data.shape[0]
            degs_lat = data.shape[1]
            pts = degs_lat * degs_lon
            lats = np.zeros(pts)
            lons = np.zeros(pts)
            ts = np.zeros(pts)
            idx = 0

            # get dates from file name
            parts = prism_file.split('_')
            date_part = parts[-2]  # The date part is the second last element
            date = None
            # Ensure that the extracted part is of the correct length for a date
            if len(date_part) == 8:
                # Try converting to a date to validate
                year = int(date_part[0:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                date = datetime.date(year, month, day)

            print(f"Processing file for date: {date}")
            for j in range(degs_lon):
                for i in range(degs_lat):
                    # Convert pixel coordinates to lat/lon
                    lon, lat = rasterio.transform.xy(transform, j, i)
                    # Get the TMin value
                    tmp = data[j, i]

                    # don't include NAN's
                    if tmp == -9999.0:
                        continue

                    # Now you have lat, lon, and TMin for each point
                    # print(lat, lon, tmin)
                    lats[idx] = lat
                    lons[idx] = lon
                    ts[idx] = tmp

                    idx += 1

        # we have degrees fahrenheit, lat, lon, and lots of nans...
        df = pd.DataFrame({'Latitude': lats, 'Longitude': lons, variable: ts})
        df['Date'] = date
        nan_data_filter = df['Latitude'] > 0
        filtered_dataset = df[nan_data_filter]
        filtered_dataset = filtered_dataset[globals.latS <= filtered_dataset['Latitude']]
        filtered_dataset = filtered_dataset[filtered_dataset['Latitude'] <= globals.latN]
        filtered_dataset = filtered_dataset[globals.lonW <= filtered_dataset['Longitude']]
        filtered_dataset = filtered_dataset[filtered_dataset['Longitude'] <= globals.lonE]
        # only do Kelvin conv if we're working with temp
        if variable != 'us':
            kelvins = (filtered_dataset[variable] -32)* (5 / 9) + 273.15
            filtered_dataset[f"{variable}_K"] = kelvins
        frames.append(filtered_dataset)

    df = pd.concat(frames)
    pickle_path = f"data/prism_pp/prism_{variable}_202304.p"
    pickle.dump(df, open(pickle_path, "wb"))


def get_station_data():
    prism_station_files = glob.glob('data/prism/*.stn.csv')
    stn_frames = list()
    for stn_file in prism_station_files:
        df = pd.read_csv(stn_file, header=1)
        stn_frames.append(df)
    stn_combined = pd.concat(stn_frames)

    # De-dupe and remove stns w/out elevation
    deduped_stations = stn_combined.drop_duplicates()
    nan_data_filter = deduped_stations['Elevation(m)'] != -9999.0

    filtered_dataset = deduped_stations[nan_data_filter]
    print("Combined and de-duped all station data")
    filtered_dataset.to_pickle(f"data/{globals.prism_processed_files}/station_data.p")
    return filtered_dataset
