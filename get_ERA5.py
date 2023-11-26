#!/usr/bin/env python
#
# Get ERA5 data
# Usage: python get_ERA5.py <YYYYMMMDD> 
#
# Rosie Howard
# 3 November 2022

import cdsapi
import sys

class ERA_DATA_PULL:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    D = sys.argv[1]	# date
    year = D[0:4]
    month_str = D[4:7]
    day = D[7:9]
    if month_str == 'Jan':
        month = '01'
    elif month_str == 'Feb':
        month = '02'
    elif month_str == 'Mar':
        month = '03'
    elif month_str == 'Apr':
        month = '04'
    elif month_str == 'May':
        month = '05'
    elif month_str == 'Jun':
        month = '06'
    elif month_str == 'Jul':
        month = '07'
    elif month_str == 'Aug':
        month = '08'
    elif month_str == 'Sep':
        month = '09'
    elif month_str == 'Oct':
        month = '10'
    elif month_str == 'Nov':
        month = '11'
    elif month_str == 'Dec':
        month = '12'

    filename = D[0:21]
    print('month = ', month_str)
    print('month = ', month)
    print('filename =', filename)

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature','2m_dewpoint_temperature',
                 '10u','10v',
                 'mean_sea_level_pressure','surface_pressure',
                 'snow_depth','snowfall','total_precipitation',
                 'total_cloud_cover','cloud_base_height'],
            'year': year,
            'month': month,
            'day': day,
            'time': ['00:00','01:00','02:00','03:00',
             '04:00','05:00','06:00','07:00',
             '08:00','09:00','10:00','11:00',
             '12:00','13:00','14:00','15:00',
             '16:00','17:00','18:00','19:00',
             '20:00','21:00','22:00','23:00'],
        'area': ['60','-135','45','-110'],
            'format': 'netcdf',                 # Supported format: grib and netcdf. Default: grib
        },
        filename+'.nc')                          # Output file. Adapt as you wish.
