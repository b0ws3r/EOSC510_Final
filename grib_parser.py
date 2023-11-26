import glob
import subprocess
import pandas as pd
import numpy as np
from globals import grib_repo, filtered_grib_repo


def download_grib_files(
        file_path, # path containing a list of wget commands
        latN=None,
        latS=None,
        lonE=None,
        lonW=None):
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line is a valid wget command
            command = line.strip()
            ch_dir_cmd = f"cd {grib_repo};"
            if command.startswith('wget'):
                try:
                    # download grib
                    # print(f"Executing: {command}")
                    subprocess.run(ch_dir_cmd + command, shell=True, check=True)

                    # filter grib
                    file_name = command[command.strip().rindex('/')+1:]
                    print(file_name)
                    filter_grib_command = f"wgrib2 {grib_repo}/{file_name} -small_grib {lonW}:{lonE} {latS}:{latN} {filtered_grib_repo}/{file_name}"
                    print(filter_grib_command)
                    subprocess.run(filter_grib_command, shell=True, check=True)

                    # extract out variables
                    t_min = f"wgrib2 {filtered_grib_repo}/{file_name} -match ':TMIN:' -csv data/vars/tmin_{file_name}.csv"
                    t_max = f"wgrib2 {filtered_grib_repo}/{file_name} -match ':TMAX:' -csv data/vars/tmax_{file_name}.csv"
                    subprocess.run(t_min, shell=True, check=True)
                    subprocess.run(t_max, shell=True, check=True)

                    # clean up (delete v big raw gribs)
                    rm_cmd = f"cd {grib_repo}; rm {file_name}"
                    subprocess.run(rm_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while executing: {command}")
                    print(e)
            else:
                print(f"Ignored non-wget line: {line}")


def aggregate_3hr_tmin_or_tmax(variable_name):
    # take lat, lon, t from all files per day.
    # combine files

    # tmax_gfs.0p25.2023040100.f003.grib2.csv
    # todo month and yr should come from globals
    gfs_3hrs = glob.glob(f"data/gfs_3hr_min_max/{variable_name}_gfs.0p25.202304*.csv")
    # create a parent file to hold all obs over all days
    parent_df = None
    frames = list()
    for temperature_file in gfs_3hrs:
        df = pd.read_csv(temperature_file, header=None)
        # Convert the DateTimeStart column to datetime and extract the date
        df['Date'] = pd.to_datetime(df[1]).dt.date
        # Group by Date, Longitude, and Latitude, and get the max temperature
        frames.append(df)

    # create a dataframe containing all of the csv data..
    # process all of it to get daily temp instead of 3 hour high/low temps
    parent_df = pd.concat(frames)
    daily_crit = pd.DataFrame()
    if variable_name == 'tmax':
        daily_crit = parent_df.groupby(['Date', 4, 5])[6].max().reset_index()
    else:
        daily_crit = parent_df.groupby(['Date', 4, 5])[6].min().reset_index()

    # Rename the columns for clarity
    daily_crit.columns = ['Date', 'Longitude', 'Latitude', 'Temp']

    # Now daily_max_temp contains the maximum temperature for each day and location
    # print(daily_max_temp)
    # Save result to file
    final = pd.DataFrame(daily_crit)
    final.to_csv(f"data/gfs_final/{variable_name}_202304.csv")
