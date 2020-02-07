#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Simple routines for graphically evaluating the performance of a DLWP model on the cubed sphere.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import os
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from DLWP.model import SeriesDataGenerator, TimeSeriesEstimator
from DLWP.model import verify
from DLWP.model.preprocessing import get_constants
from DLWP.plot import history_plot, zonal_mean_plot
from DLWP.util import load_model, train_test_split_ind, remove_chars, is_channels_last
from DLWP.remap import CubeSphereRemap

import tensorflow as tf
# Disable warning logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Set only GPU 1
device = tf.config.list_physical_devices('GPU')[1]
tf.config.set_visible_devices([device], 'GPU')
# Allow memory growth
tf.config.experimental.set_memory_growth(device, True)


#%% User parameters

# Configure the data files. The predictor file contains the predictors for the model, already on the cubed sphere,
# with the conversion to the face coordinate. The validation file contains contains raw data passed through the
# forward and inverse mapping so that it is processed the same way. It should also contain the mean and std of the
# validation variables.
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '/home/gold/jweyn/Data/era5_2deg_3h_CS2_1979-2018_z-tau-t2_500-1000_tcwv_psi850.nc'
validation_file = '%s/era5_2deg_3h_validation_z500_t2m_ILL.nc' % root_directory
climo_file = '%s/era5_2deg_3h_1979-2010_climatology_z500-z1000-t2.nc' % root_directory

# Map files for cubed sphere remapping
map_files = ('/home/disk/brume/jweyn/Documents/DLWP/map_LL91x180_CS48.nc',
             '/home/disk/brume/jweyn/Documents/DLWP/map_CS48_LL91x180.nc')

# Names of model files, located in the root_directory, and labels for those models
models = [
    'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax',
    'dlwp_era5_6h-3_CS48_tau-sfc1000-tcwv-lsm-topo_UNET2-relumax',
    'dlwp_era5_6h-3_CS48_tau-sfc1000-tcwv-lsm-topo_UNET2-48-relumax',
]
model_labels = [
    '4-variable U-net',
    '5-variable U-net (TCWV)',
    '5-variable U-net-48 (TCWV)',
]

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need. Separate input and output selections are available for models using different inputs
# and outputs. Also specify the number of input/output time steps in each model.
input_selection = output_selection = [
    {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0']},
    {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0', 'tcwv/0']},
    {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0', 'tcwv/0']},
]

# Optional added constant inputs
constant_fields = [
    [
        ('/home/gold/jweyn/Data/era5_2deg_3h_CS2_land_sea_mask.nc', 'lsm'),
        ('/home/gold/jweyn/Data/era5_2deg_3h_CS2_scaled_topo.nc', 'z')
    ],
] * len(models)

# Other required parameters
add_insolation = [True] * len(models)
input_time_steps = [2] * len(models)
output_time_steps = [2] * len(models)

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

# Number of forward integration weather forecast time steps
num_forecast_hours = 672
dt = 6

# Latitude bounds for MSE calculation
lat_range = [-90., 90.]

# Calculate statistics for a selected variable and level, or varlev if the predictor data was produced pairwise.
# Provide as a dictionary to extract to kwargs. If None, then averages all variables. Cannot be None if using a
# barotropic model for comparison (specify Z500).
selection = {
    'varlev': 'z/500'
}

# Scale the variables to original units
scale_variables = True

# Flag to do daily averages (useful for comparison to daily-averaged ECMWF S2S forecasts)
daily_mean = False

# Optionally add another forecast
# added_forecast_file = '%s/../S2S/ECMF/daily_2m_temperature__2013-2018_from_2018_ILL_2deg.nc' % root_directory
added_forecast_file = '%s/../S2S/ECMF/geopotential_500_2013-2018_from_2018_ILL_2deg.nc' % root_directory
added_forecast_variable = 'gh'
added_forecast_label = 'S2S ECMWF control'
added_scale_factor = 9.81

# Do specific plots
plot_directory = '/home/disk/brume/jweyn/Documents/DLWP/Plots'
plot_history = False
plot_zonal = True
plot_mse = True
plot_spread = False
plot_mean = False
method = 'acc'
mse_title = r'Z500; 2013-16; global'  # '20-70$^{\circ}$N'
mse_file_name = 'acc_era_6h_CS48_z500-tcwv.pdf'
mse_pkl_file = 'acc_era_6h_CS48_z500-tcwv.pkl'


#%% Pre-processing

all_ds = xr.open_dataset(predictor_file)
if 'time_step' in all_ds.dims:
    all_ds = all_ds.isel(time_step=-1)

# Find the validation set
if isinstance(validation_set, int):
    n_sample = all_ds.dims['sample']
    train_set, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    predictor_ds = all_ds.isel(sample=val_set)
else:  # we must have a list of datetimes
    predictor_ds = all_ds.sel(sample=validation_set)

# Shortcuts for latitude range
lat_min = np.min(lat_range)
lat_max = np.max(lat_range)

# Format the predictor indexer and variable index in reshaped array
input_selection = input_selection or [None] * len(models)
output_selection = output_selection or [None] * len(models)
selection = selection or {}

# Lists to populate
mse = []
f_hours = []

# Scaling parameters
scale_ds = xr.open_dataset(validation_file)
sel_mean = scale_ds.sel(**selection).variables['mean'].values
sel_std = scale_ds.sel(**selection).variables['std'].values
scale_ds.close()


#%% Generate verification

print('Generating verification...')
num_forecast_steps = num_forecast_hours // dt

csr = CubeSphereRemap()
csr.assign_maps(*map_files)

# Open the remapped validation file
remapped_ds = xr.open_dataset(validation_file)

if 'time_step' in remapped_ds.dims:
    verification_ds = remapped_ds.isel(time_step=-1)
try:
    remapped_ds = remapped_ds.drop('targets')
except ValueError:
    pass

# Subset the validation set
if isinstance(validation_set, int):
    n_sample = all_ds.dims['sample']
    train_set, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    verification_ds = remapped_ds.isel(sample=val_set)
else:  # we must have a list of datetimes
    verification_ds = remapped_ds.sel(sample=validation_set)

# Load the verification data
verification_ds = verification_ds.sel(**selection)
verification_ds.load()
verification = verify.verification_from_series(verification_ds, init_times=initialization_dates,
                                               forecast_steps=num_forecast_steps, dt=dt, f_hour_timedelta_type=False)
verification = verification.isel(lat=((verification.lat >= lat_min) & (verification.lat <= lat_max)))
if scale_variables:
    verification = verification * sel_std + sel_mean

if daily_mean:
    verification['f_day'] = xr.DataArray(np.floor((verification.f_hour.values - 1) / 24.) + 1., dims=['f_hour'])
    verif_daily = verification.groupby('f_day').mean('f_hour')

# Load the climatology data
if climo_file is None:
    print('Generating climatology...')
    climo_data = verify.daily_climatology(remapped_ds['predictors']
                                          .isel(lat=((remapped_ds.lat >= lat_min) & (remapped_ds.lat <= lat_max)))
                                          .sel(**selection)
                                          .rename({'sample': 'time'}))
else:
    print('Opening climatology from %s...' % climo_file)
    climo_ds = xr.open_dataset(climo_file)
    climo_ds = climo_ds.assign_coords(lat=verification_ds.lat[:].values, lon=verification_ds.lon[:].values)
    climo_data = climo_ds['predictors'].isel(
        lat=((climo_ds.lat >= lat_min) & (climo_ds.lat <= lat_max))).sel(**selection)
if scale_variables:
    climo_data = climo_data * sel_std + sel_mean
acc_climo = verify.daily_climo_time_series(climo_data, verification.time.values, verification.f_hour.values)
if daily_mean:
    acc_climo['f_day'] = xr.DataArray(np.floor((acc_climo.f_hour.values - 1) / 24. + 1), dims=['f_hour'])
    acc_climo = acc_climo.groupby('f_day').mean('f_hour')


#%% Iterate through the models and calculate their stats

for m, model in enumerate(models):
    print('Loading model %s...' % model)
    # Load the model
    dlwp, history = load_model('%s/%s' % (root_directory, model), True, gpus=1)

    try:
        file_var = '_' + remove_chars(selection['varlev'])
    except KeyError:
        file_var = ''
    forecast_file = '%s/forecast_%s%s.nc' % (root_directory, remove_chars(model), file_var)
    if os.path.isfile(forecast_file):
        print('Forecast file %s already exists; using it. If issues arise, delete this file and try again.'
              % forecast_file)

    else:
        # Create data generator
        constants = get_constants(constant_fields[m])
        sequence = dlwp._n_steps if hasattr(dlwp, '_n_steps') and dlwp._n_steps > 1 else None
        val_generator = SeriesDataGenerator(dlwp, predictor_ds, rank=3, add_insolation=add_insolation[m],
                                            input_sel=input_selection[m], output_sel=output_selection[m],
                                            input_time_steps=input_time_steps[m],
                                            output_time_steps=output_time_steps[m],
                                            sequence=sequence, batch_size=64, load=False, constants=constants,
                                            channels_last=is_channels_last(dlwp))

        estimator = TimeSeriesEstimator(dlwp, val_generator)

        # Make a time series prediction
        print('Predicting with model %s...' % model_labels[m])
        samples = np.array([int(np.where(val_generator.ds['sample'] == s)[0]) for s in verification.time]) \
            - input_time_steps[m] + 1
        time_series = estimator.predict(num_forecast_steps, samples=samples, verbose=1)

        # Transpose if channels_last was used for the model
        if is_channels_last(dlwp):
            time_series = time_series.transpose('f_hour', 'time', 'varlev', 'x0', 'x1', 'x2')

        # For some reason the DataArray produced by TimeSeriesEstimator is incompatible with ncview and the remap code.
        # Change the coordinates using a different method.
        fh = np.arange(dt, time_series.shape[0] * dt + 1., dt)
        sequence = 1 if sequence is None else sequence
        time_series = verify.add_metadata_to_forecast_cs(
            time_series.values,
            fh,
            predictor_ds.sel(**output_selection[m]).sel(sample=verification.time)
        )
        time_series = time_series.sel(**selection)

        # Save and remap
        print('Remapping from cube sphere...')
        time_series.to_netcdf(forecast_file + '.cs')
        time_series = None

        csr.convert_from_faces(forecast_file + '.cs', forecast_file + '.tmp')
        csr.inverse_remap(forecast_file + '.tmp', forecast_file, '--var', 'forecast')
        os.remove(forecast_file + '.tmp')

    time_series_ds = xr.open_dataset(forecast_file)
    time_series = time_series_ds.forecast

    # Slice the arrays as we want. Remapping the cube sphere changes the lat coordinate; match with the verification.
    try:
        time_series = time_series.assign_coords(lat=verification_ds.lat[:])
    except:
        pass
    time_intersection = np.intersect1d(time_series.time.values, verification.time.values, assume_unique=True)
    fh_intersection = np.intersect1d(time_series.f_hour.values, verification.f_hour.values, assume_unique=True)
    time_series = time_series.sel(time=time_intersection, f_hour=fh_intersection).isel(
        lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max))
    )
    time_series.load()

    # Filter out where the forecasts blow up. This is only a temporary patch for poor models.
    filter_time = time_series.time[xr.where(time_series.max(('f_hour', 'lat', 'lon')) < 10., True, False)]
    time_series = time_series.sel(time=filter_time)
    time_intersection = np.intersect1d(filter_time, verification.time.values, assume_unique=True)

    if scale_variables:
        time_series = time_series * sel_std + sel_mean

    # Calculate the MSE for each forecast hour relative to observations
    if daily_mean:
        time_series['f_day'] = xr.DataArray(np.floor((time_series.f_hour.values - 1) / 24. + 1), dims=['f_hour'])
        time_series_daily = time_series.groupby('f_day').mean('f_hour')
        mse.append(verify.forecast_error(time_series_daily,
                                         verif_daily.sel(time=time_intersection,
                                                         f_day=time_series_daily.f_day),
                                         method=method, weighted=True, climatology=acc_climo))
        f_hours.append(time_series_daily.f_day.values * 24)
    else:
        mse.append(verify.forecast_error(time_series,
                                         verification.sel(time=time_intersection, f_hour=fh_intersection),
                                         method=method, weighted=True, climatology=acc_climo))
        f_hours.append(fh_intersection)

    # Plot learning curves
    if plot_history:
        history_plot(history['mean_absolute_error'], history['val_mean_absolute_error'], model_labels[m],
                     out_directory=plot_directory)

    # Plot the zonal climatology of the last forecast hour
    if plot_zonal:
        obs_zonal_mean = verification[-1].mean(axis=(0, -1))
        obs_zonal_std = verification[-1].std(axis=-1).mean(axis=0)
        pred_zonal_mean = time_series[-1].mean(axis=(0, -1))
        pred_zonal_std = time_series[-1].std(axis=-1).mean(axis=0)
        zonal_mean_plot(obs_zonal_mean, obs_zonal_std, pred_zonal_mean, pred_zonal_std, dt*num_forecast_steps // 24,
                        var_name=file_var.upper()[1:], model_name='%s%s' % (model_labels[m], file_var),
                        out_directory=plot_directory)

    # Clear the model
    dlwp = None
    time_series = None
    # tf.compat.v1.keras.backend.clear_session()


#%% Add an extra model

if added_forecast_file is not None:
    print('Loading added model %s...' % added_forecast_file)
    fcst_ds = xr.open_dataset(added_forecast_file)
    fcst_ds = fcst_ds.isel(lat=((fcst_ds.lat >= lat_min) & (fcst_ds.lat <= lat_max)))
    fcst_ds = fcst_ds.sel(time=initialization_dates)
    fcst_ds.load()

    # Select the correct forecast hours
    fh_intersection = np.intersect1d(fcst_ds.f_hour.values, verification.f_hour.values, assume_unique=True)
    fcst = fcst_ds[added_forecast_variable].transpose(
        'f_hour', 'time', 'lat', 'lon').sel(f_hour=fh_intersection)
    fcst *= added_scale_factor

    # Normalize by the same std and mean as the predictor dataset if needed
    if not scale_variables:
        fcst = (fcst - sel_mean) / sel_std

    if daily_mean:
        fcst['f_day'] = xr.DataArray(np.floor((fcst.f_hour.values - 1) / 24. + 1), dims=['f_hour'])
        fcst_daily = fcst.groupby('f_day').mean('f_hour')
        mse.append(verify.forecast_error(fcst_daily,
                                         verif_daily.sel(time=initialization_dates,
                                                         f_day=fcst_daily.f_day),
                                         method=method, weighted=True, climatology=acc_climo))
        f_hours.append(fcst_daily.f_day.values * 24)
    else:
        mse.append(verify.forecast_error(fcst,
                                         verification.sel(time=initialization_dates, f_hour=fh_intersection),
                                         method=method, weighted=True, climatology=acc_climo))
        f_hours.append(fh_intersection)

    model_labels.append(added_forecast_label)


#%% Add persistence and climatology

print('Calculating persistence forecasts...')
if daily_mean:
    init = verify.verification_from_series(verification_ds,
                                           init_times=[d - pd.Timedelta(hours=24) for d in initialization_dates],
                                           forecast_steps=24 / dt, dt=dt).mean('f_hour')
    f_hours.append(verif_daily.f_day.values * 24)
else:
    init = verification_ds.predictors.sel(sample=verification.time).isel(
        lat=((verification_ds.lat >= lat_min) & (verification_ds.lat <= lat_max)))
    init.load()
    f_hours.append(np.arange(dt, num_forecast_steps * dt + 1., dt))

if scale_variables:
    init = init * sel_std + sel_mean
if 'time_step' in init.dims:
    init = init.isel(time_step=-1)
persist = xr.concat([init] * len(f_hours[-1]), dim='f_day' if daily_mean else 'f_hour').assign_coords(
    **{'f_day' if daily_mean else 'f_hour': verif_daily.f_day if daily_mean else verification.f_hour})
if daily_mean:
    persist = persist.assign_coords(time=verif_daily.time[:])
mse.append(verify.forecast_error(persist, verif_daily if daily_mean else verification,
                                 method=method, weighted=True, climatology=acc_climo))
model_labels.append('Persistence')

print('Calculating climatology forecasts...')
mse.append(verify.monthly_climo_error(init, init.time, n_fhour=num_forecast_steps, method=method,
                                      climo_da=None if climo_file is None else climo_data, by_day_of_year=True,
                                      weighted=True))
model_labels.append('Climatology')
f_hours.append(np.arange(dt, num_forecast_steps * dt + 1., dt))


#%% Plot the combined MSE as a function of forecast hour for all models

if plot_mse:
    if plot_spread:
        fig = plt.figure()
        fig.set_size_inches(6, 4)
        for m, model in enumerate(model_labels):
            if model in [added_forecast_label, 'Persistence', 'Climatology']:
                plt.plot(f_hours[m] / 24, mse[m], label=model, linewidth=2.)
        mean = np.mean(np.array(mse[:len(models)]), axis=0)
        plt.plot(f_hours[0] / 24, mean, 'k-', label=r'DLWP mean', linewidth=1.)
        std = np.std(np.array(mse[:len(models)]), axis=0)
        plt.fill_between(f_hours[0] / 24, mean - std, mean + std,
                         facecolor=(0.5, 0.5, 0.5, 0.5), zorder=-50)
        plt.xlim([0, num_forecast_hours // 24])
        plt.xticks(np.arange(0, num_forecast_hours // 24 + 1, 2))
        if method in ['acc', 'cos']:
            plt.ylim(top=1)
        else:
            plt.ylim(bottom=0)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, color='lightgray', zorder=-100)
        plt.xlabel('forecast day')
        plt.ylabel(method.upper())
        plt.title(mse_title)
        plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
        plt.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(6, 4)
        for m, model in enumerate(model_labels):
            if model in [added_forecast_label, 'Persistence', 'Climatology']:
                plt.plot(f_hours[m] / 24, mse[m], label=model, linewidth=2.)
            else:
                if plot_mean:
                    plt.plot(f_hours[m] / 24, mse[m], label=model, linewidth=1., linestyle='--' if m < 10 else ':')
                else:
                    plt.plot(f_hours[m] / 24, mse[m], label=model, linewidth=2.)
        if plot_mean:
            plt.plot(f_hours[0] / 24, np.mean(np.array(mse[:len(models)]), axis=0), label='mean', linewidth=2.)
        plt.xlim([0, num_forecast_hours // 24])
        plt.xticks(np.arange(0, num_forecast_hours // 24 + 1, 2))
        if method in ['acc', 'cos']:
            plt.ylim(top=1)
        else:
            plt.ylim(bottom=0)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, color='lightgray', zorder=-100)
        plt.xlabel('forecast day')
        plt.ylabel(method.upper())
        plt.title(mse_title)
        plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
        plt.show()

if mse_pkl_file is not None:
    result = {'f_hours': f_hours, 'models': models, 'model_labels': model_labels, 'mse': mse}
    with open('%s/%s' % (plot_directory, mse_pkl_file), 'wb') as f:
        pickle.dump(result, f)

print('Done writing figures to %s' % plot_directory)
