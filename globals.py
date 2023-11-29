grib_repo = "data/grib2"
filtered_grib_repo = "data/filtered_grib2"
latN = 49
latS = 48
lonW = -122
lonE = -119

### GFS data dirs
grib2 = "grib2" # data where wget commands place grib2 files (and then delete them when they are re-processed)
filtered_grib2 = "filtered_grib2" # grib2 files filtered by lat/lon
gfs_3hr_min_max = "gfs_3hr_min_max" # contains CSV files created from gribs. Each csv is 3 hr valid forecast for tmin or tmax
gfs_daily_min_max = "gfs_final" # csv files containing daily tmin/tmax for each day (2 files)

### PRISM data dirs
# Raw prism files directory
raw_prism = "prism" # contains raw prism data (all available lat/lon, fahrenheit observations). These are .bil files
# Post-processed PRISM files directory name
prism_processed_files = "prism_pp" # Pickle files: filtered for NAN's, filtered for lat/lon range, converted to K



val_filter = [48.5, 48.75]# df["Latitude"] >= 48.5
trn_filter = [48, 48.5]
tst_filter = [48.75, 49]