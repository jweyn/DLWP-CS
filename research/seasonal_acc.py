#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Evaluate 3-4 week anomaly forecasts with cosine similarity. Requires forecast output files from validate2_cs.py.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from DLWP.model import verify
from DLWP.util import remove_chars


#%% User parameters

# Configure the data files. The predictor file contains the predictors for the model, already on the cubed sphere,
# with the conversion to the face coordinate. The validation file contains contains raw data passed through the
# forward and inverse mapping so that it is processed the same way. It should also contain the mean and std of the
# validation variables.
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
climatology_file = '%s/era5_2deg_3h_1979-2010_climatology_z500-z1000-t2.nc' % root_directory
validation_file = '%s/era5_2deg_3h_validation_z500_t2m_ILL.nc' % root_directory

# Names of model files, located in the root_directory, and labels for those models
models = [
    # 'dlwp_era5_6h_CS48_tau-sfc1000-lsm_UNET2',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2',
    'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax',
    'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-NP-relumax',
    'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET3-relumax',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-psi-lsm-topo_UNET2-relumax',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm_UNET2-relumax',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax5',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relu0',
    # 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relu0max1',
]
model_labels = [
    # 'ERA-6h SFC4 LSM UNET2 ReLU-N',
    # 'ERA-6h (x3) SFC4 LSM TOPO UNET2 ReLU-N',
    '4-variable U-net',
    '4-variable U-net-NP',
    '4-variable U-net-3',
    # '5-variable U-net CNN (Psi$_{850}$)'
    # '4-variable U-net CNN, no topo',
    # 'ERA-6h (x3) SFC4 LSM TOPO UNET2 ReLU-N-5',
    # 'ERA-6h (x3) SFC4 LSM TOPO UNET2 ReLU-0',
    # 'ERA-6h (x3) SFC4 LSM TOPO UNET2 ReLU-0-1',
]

# Subset the validation data. We will calculate an entire verification based on this part of the data. It is acceptable
# to keep the entire validation data.
# validation_set = 4 * (365 * 4 + 1)
start_date = datetime(2012, 12, 31, 0)
end_date = datetime(2017, 1, 31, 18)
validation_set = pd.date_range(start_date, end_date, freq='6H')
validation_set = np.array(validation_set, dtype='datetime64[ns]')

# Select forecast initialization times. These are the actual forecast start times we will run the model and verification
# for, and will also correspond to the comparison model forecast start times.
dates_1 = pd.date_range('2013-01-01', '2013-12-31', freq='7D')
dates_2 = pd.date_range('2013-01-04', '2013-12-31', freq='7D')
dates_0 = dates_1.append(dates_2).sort_values()
dates = dates_0.copy()
for year in range(2014, 2017):
    dates = dates.append(pd.DatetimeIndex(pd.Series(dates_0).apply(lambda x: x.replace(year=year))))
initialization_dates = dates

# Forecast days, inclusive
forecast_start_day = 15
forecast_end_day = 28
dt = 6

# Latitude bounds for MSE calculation
lat_range = [-70., 70.]

# Calculate statistics for a selected variable and level, or varlev if the predictor data was produced pairwise.
# Provide as a dictionary to extract to kwargs. If None, then averages all variables. Cannot be None if using a
# barotropic model for comparison (specify Z500).
selection = {
    'varlev': 't2m/0'
}

# Optionally add another forecast
added_forecast_file = '%s/../S2S/ECMF/daily_2m_temperature__2013-2018_from_2018_ILL_2deg.nc' % root_directory
# added_forecast_file = '%s/../S2S/ECMF/geopotential_500_2013-2018_from_2018_ILL_2deg.nc' % root_directory
added_forecast_variable = 't2m'
added_forecast_label = 'S2S ECMWF control'
added_scale_factor = 1.

# Plot options
plot_directory = '/home/disk/brume/jweyn/Documents/DLWP/Plots'
acc_title = r'T2 3-4 weeks; 2013-16; -70 to 70'
acc_file_name = 'acc_t2m_week3-4_70to70.pdf'


#%% Pre-processing

climo_ds = xr.open_dataset(climatology_file)
if 'time_step' in climo_ds.dims:
    climo_ds = climo_ds.isel(time_step=-1)
climo_ds.load()

# Shortcuts for latitude range
lat_min = np.min(lat_range)
lat_max = np.max(lat_range)

# Scaling parameters: must be contained in one of the climatology or validation datasets
try:
    scale_ds = xr.open_dataset(validation_file)
    sel_mean = scale_ds.sel(**selection).variables['mean'].values
    sel_std = scale_ds.sel(**selection).variables['std'].values
    scale_ds.close()
except KeyError:
    scale_ds = xr.open_dataset(climatology_file)
    sel_mean = scale_ds.sel(**selection).variables['mean'].values
    sel_std = scale_ds.sel(**selection).variables['std'].values
    scale_ds.close()


#%% Generate verification

print('Generating verification...')
num_forecast_steps = forecast_end_day * 24 // dt

# Open the remapped validation file
verification_ds = xr.open_dataset(validation_file)
if 'time_step' in verification_ds.dims:
    verification_ds = verification_ds.isel(time_step=-1)
try:
    verification_ds = verification_ds.drop('targets')
except ValueError:
    pass

# Subset the validation set
verification_ds = verification_ds.sel(sample=validation_set)
verification_ds = verification_ds.sel(**selection)
verification_ds.load()
verification = verify.verification_from_series(verification_ds, init_times=initialization_dates,
                                               forecast_steps=num_forecast_steps, dt=dt, f_hour_timedelta_type=False)
verification = verification.sel(time=initialization_dates).isel(
    lat=((verification.lat >= lat_min) & (verification.lat <= lat_max)))
verification = verification * sel_std + sel_mean

# For each forecast date, calculate the average anomaly forecast for the period of interest
print('Calculating climatology forecasts...')
climatologies = []
for d, date in enumerate(initialization_dates):
    forecast_range = pd.date_range(date + pd.Timedelta(days=forecast_start_day),
                                   date + pd.Timedelta(days=forecast_end_day), freq='D')
    climatologies.append(climo_ds['predictors'].sel(dayofyear=forecast_range.dayofyear, **selection)
                         .isel(lat=((climo_ds.lat >= lat_min) & (climo_ds.lat <= lat_max))).mean('dayofyear'))

climo = xr.concat(climatologies, dim='time').assign_coords(time=initialization_dates,
                                                           lat=verification.lat, lon=verification.lon)
climo = climo * sel_std + sel_mean

anomaly_true = verification.sel(f_hour=slice(24 * forecast_start_day, 24 * forecast_end_day)).mean('f_hour')
anomaly_true = anomaly_true - climo


#%% Calculate the anomaly forecasts and ACC from each model

def cosine_acc(fcst, true):
    return fcst.dot(true, dims=('lat', 'lon')) / (np.linalg.norm(fcst, axis=(1, 2)) *
                                                  np.linalg.norm(true, axis=(1, 2)))


acc_forecast = []

for m, model in enumerate(models):
    print('Loading forecast from model %s...' % model)

    try:
        file_var = '_' + remove_chars(selection['varlev'])
    except KeyError:
        file_var = ''
    forecast_file = '%s/forecast_%s%s.nc' % (root_directory, remove_chars(model), file_var)
    if not os.path.isfile(forecast_file):
        print("Error: forecast '%s' does not exist! Run validate2_cs.py first." % forecast_file)
        raise IOError

    forecast_ds = xr.open_dataset(forecast_file)
    time_series = forecast_ds.forecast

    # Slice the arrays as we want. Remapping the cube sphere changes the lat coordinate; match with the verification.
    try:
        time_series = time_series.assign_coords(lat=verification.lat)
    except:
        pass
    time_series = time_series.sel(time=initialization_dates,
                                  f_hour=slice(24 * forecast_start_day, 24 * forecast_end_day)) \
        .isel(lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max)))
    time_series.load()
    time_series = time_series * sel_std + sel_mean

    # Calculate the anomaly forecast
    anomaly_forecast = time_series.mean('f_hour')
    anomaly_forecast = anomaly_forecast - climo

    # Calculate the cosine ACC of the forecasts
    # Filter out where the forecasts blow up. This is only a temporary patch for poor models.
    filter_time = anomaly_forecast.time[xr.where(anomaly_forecast.max(('lat', 'lon')) < 5000., True, False)]
    acc_forecast.append(cosine_acc(anomaly_forecast.sel(time=filter_time), anomaly_true.sel(time=filter_time)))

    # Clean up
    time_series = None
    anomaly_forecast = None
    forecast_ds = None


#%% Calculate the ACC from the added forecast

if added_forecast_file is not None:
    print('Loading added model %s...' % added_forecast_file)
    added_ds = xr.open_dataset(added_forecast_file)
    added_ds = added_ds.isel(lat=((added_ds.lat >= lat_min) & (added_ds.lat <= lat_max)))
    added_ds = added_ds.sel(time=initialization_dates, f_hour=slice(24 * forecast_start_day, 24 * forecast_end_day))
    added_ds.load()

    # Format the added forecast and calculate anomaly
    added_forecast = added_ds[added_forecast_variable].transpose('f_hour', 'time', 'lat', 'lon')

    anomaly_forecast = added_forecast.mean('f_hour')
    anomaly_forecast *= added_scale_factor  # in case the units don't match
    anomaly_forecast = anomaly_forecast - climo

    # Calculate the cosine ACC
    acc_forecast.append(cosine_acc(anomaly_forecast, anomaly_true))
    model_labels.append(added_forecast_label)


#%% Add a persistence forecast, determined from the last n days, where n is the length of the forecast period

print('Calculating persistence forecasts...')
persistences = []
persist_climo = []
for d, date in enumerate(initialization_dates):
    persistences.append(verification_ds['predictors']
                        .sel(sample=slice(date - pd.Timedelta(days=forecast_end_day - forecast_start_day), date))
                        .isel(lat=((verification_ds.lat >= lat_min) & (verification_ds.lat <= lat_max))).mean('sample'))
    forecast_range = pd.date_range(date - pd.Timedelta(days=forecast_end_day - forecast_start_day), date, freq='D')
    persist_climo.append(climo_ds['predictors'].sel(dayofyear=forecast_range.dayofyear, **selection)
                         .isel(lat=((climo_ds.lat >= lat_min) & (climo_ds.lat <= lat_max))).mean('dayofyear'))

persistence = xr.concat(persistences, dim='time').assign_coords(time=initialization_dates,
                                                                lat=verification.lat, lon=verification.lon)
persistence = persistence * sel_std + sel_mean
persist_climo = xr.concat(persist_climo, dim='time').assign_coords(time=initialization_dates,
                                                                   lat=verification.lat, lon=verification.lon)
persist_climo = persist_climo * sel_std + sel_mean

anomaly_forecast = persistence - persist_climo
acc_forecast.append(cosine_acc(anomaly_forecast, anomaly_true))
model_labels.append('Persistence')

#%% Plot the results

fig = plt.figure()
fig.set_size_inches(6, 4)

bar_values = np.array([da.mean('time').values for da in acc_forecast])
bar_min = -1 * np.array([da.min('time').values for da in acc_forecast]) + bar_values
bar_max = np.array([da.max('time').values for da in acc_forecast]) - bar_values

plt.barh(range(len(model_labels)), bar_values, xerr=(bar_min, bar_max), tick_label=model_labels,
         color=['C%d' % m for m in range(len(model_labels))])
plt.grid(True, color='lightgray', zorder=-100)
plt.xlabel('cosine ACC')
plt.xlim(-0.5, 1.0)
plt.axvline(0., color='k', linewidth=1.5)
plt.title(acc_title)
plt.savefig('%s/%s' % (plot_directory, acc_file_name), bbox_inches='tight')
plt.close()
