#
# Copyright (c) 2020 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Calculate basic error metrics for an IFS forecast, using the metview library for regridding.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import metview
import os
from datetime import datetime
from DLWP import verify

#%% User parameters

# The validation file contains contains raw data. It should also contain the mean and std of the validation variables.
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
validation_file = '%s/era5_2deg_3h_validation_z-tau-t2_500-1000.nc' % root_directory
climo_file = '%s/era5_2deg_3h_1979-2010_climatology_z500-z1000-t2.nc' % root_directory
ifs_files = [
    '%s/../IFS_T42/output_42_sfc_2.8125.grib' % root_directory,
    '%s/../IFS_T42/output_42_sfc_2.8125_2.grib' % root_directory,
]
label = 'IFS T42'

# Variable/level selection
variable = 't2m'
level = None
verification_variable = 't2m/0'
grid = [2., 2.]

daily_mean = False

# Subset the validation data. We will calculate an entire verification based on this part of the data. It is acceptable
# to keep the entire validation data.
start_date = datetime(2013, 1, 1, 0)
end_date = datetime(2018, 12, 31, 18)
validation_set = pd.date_range(start_date, end_date, freq='6H')
validation_set = np.array(validation_set, dtype='datetime64[ns]')

# Error method
methods = ['acc', 'rmse']

# Save the error calculation
output_pkl_file = 'ifs_%s_t42_t2m-full.pkl'


#%% Open each file and re-grid with metview

datasets = []
for ifs in ifs_files:
    print('Opening and re-gridding %s' % ifs.split(os.sep)[-1])
    raw_data = metview.read(ifs)
    gridded_data = metview.read(data=raw_data, grid=grid)
    datasets.append(gridded_data.to_dataset())
    raw_data = gridded_data = None

# Get a DataArray
ifs_ds = xr.concat(datasets, dim='time')
if level is None:
    ifs_da = ifs_ds[variable]
else:
    ifs_da = ifs_ds[variable].sel(isobaricInhPa=level)
ifs_da = ifs_da.transpose('step', 'time', 'latitude', 'longitude')

ifs_da = ifs_da.assign_coords(step=ifs_da.step.values.astype('datetime64[ns]').astype('datetime64[h]').astype(int))
ifs_da = ifs_da.rename({'step': 'f_hour', 'latitude': 'lat', 'longitude': 'lon'})

max_f_hour = ifs_da.f_hour.max()


#%% Generate verification

print('Generating verification...')

# Open the remapped validation file
remapped_ds = xr.open_dataset(validation_file)

if 'time_step' in remapped_ds.dims:
    verification_ds = remapped_ds.isel(time_step=-1)
try:
    remapped_ds = remapped_ds.drop('targets')
except ValueError:
    pass

# Subset the validation set
verification_ds = remapped_ds.sel(sample=validation_set)

# Load the verification data
verification_ds = verification_ds.sel(varlev=verification_variable)
verification_ds.load()
verification = verify.verification_from_series(verification_ds, init_times=ifs_da.time.values,
                                               forecast_steps=max_f_hour // 6, dt=6, f_hour_timedelta_type=False,
                                               include_zero=True)
sel_mean = verification_ds['mean'].values
sel_std = verification_ds['std'].values
verification = verification * sel_std + sel_mean
verify_hours = verification.f_hour.values.copy()


#%% Find matching forecast hours and initialization times

time_intersection = np.intersect1d(ifs_da.time.values, verification.time.values, assume_unique=True)
fh_intersection = np.intersect1d(ifs_da.f_hour.values, verification.f_hour.values, assume_unique=True)

ifs_da = ifs_da.sel(f_hour=fh_intersection, time=time_intersection)
verification = verification.sel(f_hour=fh_intersection, time=time_intersection)

if daily_mean:
    verification['f_day'] = xr.DataArray(np.floor((verification.f_hour.values - 1) / 24.) + 1., dims=['f_hour'])
    verification = verification.groupby('f_day').mean('f_hour')

    ifs_da['f_day'] = xr.DataArray(np.floor((ifs_da.f_hour.values - 1) / 24.) + 1., dims=['f_hour'])
    ifs_da = ifs_da.groupby('f_day').mean('f_hour')


#%% Load the climatology data

if climo_file is None:
    climo_data = remapped_ds['predictors'].sel(varlev=verification_variable)
else:
    print('Opening climatology from %s...' % climo_file)
    climo_ds = xr.open_dataset(climo_file)
    climo_ds = climo_ds.assign_coords(lat=verification_ds.lat[:].values, lon=verification_ds.lon[:].values)
    climo_data = climo_ds['predictors'].sel(varlev=verification_variable)

climo_data = climo_data * sel_std + sel_mean
acc_climo = verify.daily_climo_time_series(climo_data, verification.time.values, verify_hours)
if daily_mean:
    acc_climo['f_day'] = xr.DataArray(np.floor((acc_climo.f_hour.values - 1) / 24. + 1), dims=['f_hour'])
    acc_climo = acc_climo.groupby('f_day').mean('f_hour')


#%% Calculate errors and save to pkl file

for method in methods:
    error = verify.forecast_error(ifs_da, verification,
                                  method=method, weighted=True, climatology=acc_climo)

    if output_pkl_file is not None:
        result = {
            'f_hours': [ifs_da.f_day.values * 24 if daily_mean else ifs_da.f_hour.values],
            'model_labels': [label],
            'mse': [error]
        }
        with open(output_pkl_file % method, 'wb') as f:
            pickle.dump(result, f)
