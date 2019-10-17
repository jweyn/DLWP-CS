#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Simple routines for graphically evaluating the performance of a DLWP model on the cubed sphere.
"""

import keras.backend as K
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from DLWP.model import SeriesDataGenerator, TimeSeriesEstimator
from DLWP.model import verify
from DLWP.model.preprocessing import get_constants
from DLWP.plot import history_plot, forecast_example_plot, zonal_mean_plot, remove_chars
from DLWP.util import load_model, train_test_split_ind
from DLWP.data import CFSReforecast
from DLWP.remap import CubeSphereRemap


#%% User parameters

# Configure the data files. The predictor file contains the predictors for the model, already on the cubed sphere,
# with the conversion to the face coordinate. The scale file contains 'mean' and 'std' variables to perform inverse
# scaling back to real data units.
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
# predictor_file = '%s/cfs_6h_CS48_1979-2010_z3-5-7-10_tau_sfc.nc' % root_directory
# scale_file = '%s/cfs_6h_1979-2010_z3-5-7-10_tau_sfc.nc' % root_directory
predictor_file = '%s/era5_2deg_3h_CS_1979-2018_z-tau-t2_500-1000.nc' % root_directory
scale_file = '%s/era5_2deg_3h_1979-2018_z-tau-t2_500-1000.nc' % root_directory

# The remap file contains the verification data passed through the same remapping scheme as the predictors. That is,
# it contains the predictors, mapped to the cubed sphere, then remapped back with the inverse transform. If this file
# does not exist, then it will be created from the predictor file, which, notably, uses a lot of memory and is quite
# slow.
remapped_file = scale_file
reverse_lat = False

# Map files for cubed sphere remapping
map_files = ('/home/disk/brume/jweyn/Documents/DLWP/map_LL91x180_CS48.nc',
             '/home/disk/brume/jweyn/Documents/DLWP/map_CS48_LL91x180.nc')

# Names of model files, located in the root_directory, and labels for those models

# models = [
#     'dlwp_6h_CS48_3var4lev_T2_UNET'
# ]
# model_labels = [
#     '3-variable 4-level 2m-T UNET'
# ]
models = [
    'dlwp_era5_3h_CS48_tau-sol_UNET',
    'dlwp_era5_6h_CS48_tau-sol_UNET',
    'dlwp_era5_3h_CS48_tau-sfc1000_UNET',
    'dlwp_era5_3h_CS48_tau-sfc1000-lsm_UNET',
]
model_labels = [
    'ERA-3h tau SOL UNET',
    'ERA-6h tau SOL UNET',
    'ERA-3h tau z1000 t2 SOL UNET',
    'ERA-3h tau z1000 t2 SOL LSM UNET',
]

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need. Separate input and output selections are available for models using different inputs
# and outputs. Also specify the number of input/output time steps in each model.
# input_selection = output_selection = [
#     {'varlev': [
#         'HGT/200', 'HGT/500', 'HGT/850', 'HGT/1000',
#         'TMP/200', 'TMP/500', 'TMP/850', 'TMP/1000',
#         'R H/200', 'R H/500', 'R H/850', 'R H/1000',
#         'U GRD/850', 'TMP2/0'
#     ]}
# ]
input_selection = output_selection = [
    {'varlev': ['z/500', 'tau/300-700']},
    {'varlev': ['z/500', 'tau/300-700']},
    {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0']},
    {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0']},
]

# Optional added constant inputs
constant_fields = [
    None,
    None,
    None,
    [
        (os.path.join(root_directory, 'era5_2deg_3h_CS_land_sea_mask.nc'), 'lsm'),
    ],
]

# Other required parameters
add_insolation = [True] * len(models)
input_time_steps = [2] * len(models)
output_time_steps = [2] * len(models)

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects.
# validation_set = 4 * (365 * 4 + 1)
start_date = datetime(2013, 1, 1, 0)
end_date = datetime(2013, 12, 31, 18)
validation_set = pd.date_range(start_date, end_date, freq='6H')
# validation_set = [d for d in validation_set if d.month in [6, 7, 8]]
validation_set = np.array(validation_set, dtype='datetime64[ns]')

# Load a CFS Reforecast model for comparison
cfs_model_dir = '%s/CFSR/reforecast' % root_directory
cfs = CFSReforecast(root_directory=cfs_model_dir, file_id='dlwp_2week_', fill_hourly=False)
# cfs.set_dates(validation_set)
# cfs.open()
cfs_ds = None  # cfs.Dataset.isel(lat=(cfs.Dataset.lat >= 0.0))  # Northern hemisphere only

# Load a barotropic model for comparison
baro_model_file = '%s/barotropic_anal_2007-2009.nc' % root_directory
baro_ds = None  # xr.open_dataset(baro_model_file)
# baro_ds = baro_ds.isel(lat=(baro_ds.lat >= 0.0))  # Northern hemisphere only

# Number of forward integration weather forecast time steps
num_forecast_hours = 336
dt = 6

# Latitude bounds for MSE calculation
lat_range = [-90., 90.]

# Calculate statistics for a selected variable and level, or varlev if the predictor data was produced pairwise.
# Provide as a dictionary to extract to kwargs. If  None, then averages all variables. Cannot be None if using a
# barotropic model for comparison (specify Z500).
selection = {
    'varlev': 'z/500'
}

# Scale the variables to original units
scale_variables = True

# Do specific plots
plot_directory = '/home/disk/brume/jweyn/Documents/DLWP/Plots'
plot_example = None  # None to disable or the date index of the sample
plot_example_f_hour = 24  # Forecast hour index of the sample
plot_history = False
plot_zonal = True
plot_mse = True
plot_spread = False
plot_mean = False
method = 'rmse'
mse_title = r'$Z_{500}$; 2013; global'  # '20-70$^{\circ}$N'
mse_file_name = 'rmse_tau-CS48-UNET_era-sfc-badsol.pdf'
mse_pkl_file = 'rmse_tau-CS48-UNET_era-sfc-badsol.pkl'


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

# Fix negative latitude for solar radiation input
if reverse_lat:
    predictor_ds.lat.load()
    predictor_ds.lat[:] = -1. * predictor_ds.lat.values

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
scale_ds = xr.open_dataset(scale_file)
sel_mean = scale_ds.sel(**selection).variables['mean'].values
sel_std = scale_ds.sel(**selection).variables['std'].values
scale_ds.close()


#%% Generate verification

print('Generating verification...')
num_forecast_steps = num_forecast_hours // dt

csr = CubeSphereRemap()
csr.assign_maps(*map_files)

if remapped_file is None or not(os.path.exists(remapped_file)):
    print('Writing new file to remap...')
    if remapped_file is None:
        remapped_file = '%s/verification.nc' % root_directory
    predictor_ds.to_netcdf(remapped_file + '.cs')

    # Apply the same forward/backward mapping to the verification data
    print('Remapping verification from cube sphere...')
    csr.convert_from_faces(remapped_file + '.cs', remapped_file + '.tmp')
    csr.inverse_remap(remapped_file + '.tmp', remapped_file, '--var', 'predictors')
    os.remove(remapped_file + '.tmp')

# Open the remapped file; invert the incorrect latitude direction
remapped_ds = xr.open_dataset(remapped_file)
if reverse_lat:
    remapped_ds = remapped_ds.assign_coords(lat=remapped_ds.lat[::-1])

if 'time_step' in remapped_ds.dims:
    verification_ds = remapped_ds.isel(time_step=-1)
if 'varlev' in predictor_ds.dims:
    remapped_ds = remapped_ds.assign_coords(varlev=predictor_ds['varlev'])
else:
    remapped_ds = remapped_ds.assign_coords(variable=predictor_ds['variable'], level=predictor_ds['level'])
try:
    remapped_ds = remapped_ds.drop('targets')
except:
    pass

# Subset the validation set
if isinstance(validation_set, int):
    n_sample = all_ds.dims['sample']
    train_set, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    verification_ds = remapped_ds.isel(sample=val_set)
else:  # we must have a list of datetimes
    verification_ds = remapped_ds.sel(sample=validation_set)

verification_ds = verification_ds.sel(**selection)

verification_ds.load()
verification = verify.verification_from_series(verification_ds,
                                               forecast_steps=num_forecast_steps, dt=dt, f_hour_timedelta_type=False)
verification = verification.sel(lat=((verification.lat >= lat_min) & (verification.lat <= lat_max)))

if scale_variables:
    verification = verification * sel_std + sel_mean


#%% Iterate through the models and calculate their stats

for m, model in enumerate(models):
    print('Loading model %s...' % model)
    # Load the model
    dlwp, history = load_model('%s/%s' % (root_directory, model), True, gpus=1)

    forecast_file = '%s/%s_forecast.nc' % (root_directory, remove_chars(model_labels[m]))
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
                                            sequence=sequence, batch_size=64, load=False, constants=constants)

        estimator = TimeSeriesEstimator(dlwp, val_generator)

        # Make a time series prediction
        print('Predicting with model %s...' % model_labels[m])
        time_series = estimator.predict(num_forecast_steps, verbose=1)

        # For some reason the DataArray produced by TimeSeriesEstimator is incompatible with ncview and the remap code.
        # Change the coordinates using a different method.
        fh = np.arange(dt, time_series.shape[0] * dt + 1., dt)
        sequence = 1 if sequence is None else sequence
        time_series = verify.add_metadata_to_forecast_cs(
            time_series.values,
            fh,
            predictor_ds.sel(**output_selection[m]).isel(sample=slice(input_time_steps[m] - 1,
                                                                      -output_time_steps[m] * sequence))
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
    f_hours.append(time_series.f_hour.values)

    # Slice the arrays as we want. Remapping the cube sphere inverses the lat/lon directions.
    try:
        time_series = time_series.assign_coords(lat=verification_ds.lat[:])
    except:
        pass
    time_intersection = np.intersect1d(time_series.time.values, verification.time.values, assume_unique=True)
    fh_intersection = np.intersect1d(time_series.f_hour.values, verification.f_hour.values, assume_unique=True)
    time_series = time_series.sel(lat=((time_series.lat >= lat_min) & (time_series.lat <= lat_max)),
                                  time=time_intersection, f_hour=fh_intersection)
    time_series.load()
    if scale_variables:
        time_series = time_series * sel_std + sel_mean

    # Calculate the MSE for each forecast hour relative to observations
    mse.append(verify.forecast_error(time_series.values,
                                     verification.sel(time=time_intersection, f_hour=fh_intersection).values,
                                     method=method))

    # Plot learning curves
    if plot_history:
        history_plot(history['mean_absolute_error'], history['val_mean_absolute_error'], model_labels[m],
                     out_directory=plot_directory)

    # Plot an example
    if plot_example is not None:
        plot_dt = np.datetime64(plot_example)
        forecast_example_plot(predictor_ds.sel(time=plot_dt, **selection),
                              predictor_ds.sel(time=plot_dt + np.timedelta64(timedelta(hours=plot_example_f_hour)),
                                               **selection),
                              time_series.sel(f_hour=plot_example_f_hour, time=plot_dt), f_hour=plot_example_f_hour,
                              model_name=model_labels[m], out_directory=plot_directory)

    # Plot the zonal climatology of the last forecast hour
    if plot_zonal:
        obs_zonal_mean = verification[-1].mean(axis=(0, -1))
        obs_zonal_std = verification[-1].std(axis=-1).mean(axis=0)
        pred_zonal_mean = time_series[-1].mean(axis=(0, -1))
        pred_zonal_std = time_series[-1].std(axis=-1).mean(axis=0)
        zonal_mean_plot(obs_zonal_mean, obs_zonal_std, pred_zonal_mean, pred_zonal_std, dt*num_forecast_steps,
                        model_labels[m], out_directory=plot_directory)

    # Clear the model
    dlwp = None
    time_series = None
    K.clear_session()


#%% Add Barotropic model

if baro_ds is not None and plot_mse:
    print('Loading barotropic model data from %s...' % baro_model_file)
    if not selection:
        raise ValueError("specific 'variable' and 'level' for Z500 must be specified to use barotropic model")
    baro_ds = baro_ds.isel(lat=((baro_ds.lat >= lat_min) & (baro_ds.lat <= lat_max)))
    if isinstance(validation_set, int):
        baro_ds = baro_ds.isel(time=slice(input_time_steps[0] - 1, validation_set + input_time_steps[0] - 1))
    else:
        baro_ds = baro_ds.sel(time=validation_set)

    # Select the correct number of forecast hours
    baro_forecast = baro_ds.isel(f_hour=(baro_ds.f_hour > 0)).isel(f_hour=slice(None, num_forecast_steps))
    baro_forecast_steps = int(np.min([num_forecast_steps, baro_forecast.dims['f_hour']]))
    baro_f = baro_forecast.variables['Z'].values

    # Normalize by the same std and mean as the predictor dataset
    if not scale_variables:
        baro_f = (baro_f - sel_mean) / sel_std

    mse.append(verify.forecast_error(baro_f[:baro_forecast_steps], verification.values[:baro_forecast_steps],
                                     method=method))
    model_labels.append('Barotropic')
    f_hours.append(np.arange(dt, baro_forecast_steps * dt + 1., dt))
    baro_f, baro_v = None, None


#%% Add the CFS model

if cfs_ds is not None and plot_mse:
    print('Loading CFS model data...')
    if not selection:
        raise ValueError("specific 'variable' and 'level' for Z500 must be specified to use CFS model model")
    cfs_ds = cfs_ds.isel(lat=((cfs_ds.lat >= lat_min) & (cfs_ds.lat <= lat_max)))
    if isinstance(validation_set, int):
        raise ValueError("I can only compare to a CFS Reforecast with datetime validation set")
    else:
        cfs_ds = cfs_ds.sel(time=validation_set)

    # Select the correct number of forecast hours
    cfs_forecast = cfs_ds.isel(f_hour=(cfs_ds.f_hour > 0)).isel(f_hour=slice(None, num_forecast_steps))
    cfs_forecast_steps = int(np.min([num_forecast_steps, cfs_forecast.dims['f_hour']]))
    cfs_f = cfs_forecast.variables['z500'].values

    # Normalize by the same std and mean as the predictor dataset
    if not scale_variables:
        cfs_f = (cfs_f - sel_mean) / sel_std

    mse.append(verify.forecast_error(cfs_f[:cfs_forecast_steps], verification.values[:cfs_forecast_steps],
                                     method=method))
    model_labels.append('CFS')
    f_hours.append(np.arange(dt, cfs_forecast_steps * dt + 1., dt))
    cfs_f, cfs_v = None, None


#%% Add persistence and climatology

if plot_mse:
    print('Calculating persistence forecasts...')
    init = verification_ds.predictors.sel(lat=((verification_ds.lat >= lat_min) & (verification_ds.lat <= lat_max)),
                                          sample=verification.time)
    init.load()

    if scale_variables:
        init = init * sel_std + sel_mean
    if 'time_step' in init.dims:
        init = init.isel(time_step=-1)
    mse.append(verify.forecast_error(np.repeat(init.values[None, ...], num_forecast_steps, axis=0),
                                     verification.values, method=method))
    model_labels.append('Persistence')
    f_hours.append(np.arange(dt, num_forecast_steps * dt + 1., dt))

    print('Calculating climatology forecasts...')
    climo_data = remapped_ds['predictors'].sel(
        lat=((verification_ds.lat >= lat_min) & (verification_ds.lat <= lat_max)), **selection)
    if scale_variables:
        climo_data = climo_data * sel_std + sel_mean
    mse.append(verify.monthly_climo_error(climo_data, validation_set, n_fhour=num_forecast_steps, method=method))
    model_labels.append('Climatology')
    f_hours.append(np.arange(dt, num_forecast_steps * dt + 1., dt))


#%% Plot the combined MSE as a function of forecast hour for all models

if plot_mse:
    if plot_spread:
        fig = plt.figure()
        fig.set_size_inches(6, 4)
        for m, model in enumerate(model_labels):
            if model in ['Barotropic', 'CFS', 'Persistence', 'Climatology']:
                plt.plot(f_hours[m], mse[m], label=model, linewidth=2.)
        mean = np.mean(np.array(mse[:len(models)]), axis=0)
        plt.plot(f_hours[0], mean, 'k-', label=r'DLWP mean', linewidth=1.)
        std = np.std(np.array(mse[:len(models)]), axis=0)
        plt.fill_between(f_hours[0], mean - std, mean + std,
                         facecolor=(0.5, 0.5, 0.5, 0.5), zorder=-50)
        plt.xlim([0, np.max(np.array(f_hours))])
        plt.xticks(np.arange(0, np.max(np.array(f_hours)) + 1, 2 * dt))
        plt.ylim([0, 140])
        plt.yticks(np.arange(0, 141, 20))
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, color='lightgray', zorder=-100)
        plt.xlabel('forecast hour')
        plt.ylabel(method.upper())
        plt.title(mse_title)
        plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
        plt.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(6, 4)
        for m, model in enumerate(model_labels):
            if model in ['Barotropic', 'CFS', 'Persistence', 'Climatology']:
                plt.plot(f_hours[m], mse[m], label=model, linewidth=2.)
            else:
                if plot_mean:
                    plt.plot(f_hours[m], mse[m], label=model, linewidth=1., linestyle='--' if m < 10 else ':')
                else:
                    plt.plot(f_hours[m], mse[m], label=model, linewidth=2.)
        if plot_mean:
            plt.plot(f_hours[0], np.mean(np.array(mse[:len(models)]), axis=0), label='mean', linewidth=2.)
        plt.xlim([0, dt * num_forecast_steps])
        plt.xticks(np.arange(0, num_forecast_steps * dt + 1, 4 * dt))
        plt.ylim([0, 200])
        plt.yticks(np.arange(0, 201, 20))
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, color='lightgray', zorder=-100)
        plt.xlabel('forecast hour')
        plt.ylabel(method.upper())
        plt.title(mse_title)
        plt.savefig('%s/%s' % (plot_directory, mse_file_name), bbox_inches='tight')
        plt.show()

if mse_pkl_file is not None:
    result = {'f_hours': f_hours, 'models': models, 'model_labels': model_labels, 'mse': mse}
    with open('%s/%s' % (plot_directory, mse_pkl_file), 'wb') as f:
        pickle.dump(result, f)

print('Done writing figures to %s' % plot_directory)
