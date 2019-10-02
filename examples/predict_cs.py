#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Make a prediction from a DLWP model on the cubed sphere, and export to a netCDF file.
"""

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import os
from datetime import datetime

from DLWP.model import SeriesDataGenerator, TimeSeriesEstimator
from DLWP.util import load_model, train_test_split_ind
from DLWP.model import verify
from DLWP.remap import CubeSphereRemap


#%% User parameters

# Configure the data files. The predictor file contains the predictors for the model, already on the cubed sphere,
# with the conversion to the face coordinate. The scale file contains 'mean' and 'std' variables to perform inverse
# scaling back to real data units.
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_6h_CS48_1979-2010_z3-5-7-10_tau_sfc.nc' % root_directory
scale_file = '%s/cfs_6h_1979-2010_z3-5-7-10_tau_sfc.nc' % root_directory

# If True, reverse the latitude coordinate in the predicted output
reverse_lat = True

# Map files for cubed sphere remapping
map_files = ('/home/disk/brume/jweyn/Documents/DLWP/map_LL73x144_CS48.nc',
             '/home/disk/brume/jweyn/Documents/DLWP/map_CS48_LL73x144.nc')

# Names of model files, located in the root_directory, and labels for those models
model = 'dlwp_6h_CS48_surf1000_T2_UNET'
model_label = r'$\tau$-SOL-surf-1000 CS 48 UNET T2 leaky avg'

# Optional list of selections to make from the predictor dataset for each model. This is useful if, for example,
# you want to examine models that have different numbers of vertical levels but one predictor dataset contains
# the data that all models need. Separate input and output selections are available for models using different inputs
# and outputs. Also specify the number of input/output time steps in each model.
input_selection = {'varlev': ['HGT/500', 'THICK/300-700', 'HGT/1000', 'TMP2/0']}
output_selection = {'varlev': ['HGT/500', 'THICK/300-700', 'HGT/1000', 'TMP2/0']}
add_insolation = True
input_time_steps = 2
output_time_steps = 2

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects.
start_date = datetime(2003, 1, 1, 0)
end_date = datetime(2003, 12, 31, 18)
validation_set = pd.date_range(start_date, end_date, freq='6H')
# validation_set = [d for d in validation_set if d.month in [6, 7, 8]]
validation_set = np.array(validation_set, dtype='datetime64[ns]')

# Number of forward integration weather forecast time steps
num_forecast_hours = 28 * 24
dt = 6

# Scale the variables to original units
scale_variables = True


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
predictor_ds.lat.load()
predictor_ds.lat[:] = -1. * predictor_ds.lat.values


#%% Load model and data

num_forecast_steps = num_forecast_hours // dt

print('Loading model %s...' % model)
# Load the model
dlwp, history = load_model('%s/%s' % (root_directory, model), True, gpus=1)

forecast_file = '%s/%s_forecast.nc' % (root_directory, model)
if os.path.isfile(forecast_file):
    print('Forecast file %s already exists; using it. If issues arise, delete this file and try again.'
          % forecast_file)

else:
    # Create data generator
    sequence = dlwp._n_steps if hasattr(dlwp, '_n_steps') and dlwp._n_steps > 1 else None
    val_generator = SeriesDataGenerator(dlwp, predictor_ds, rank=3, add_insolation=add_insolation,
                                        input_sel=input_selection, output_sel=output_selection,
                                        input_time_steps=input_time_steps,
                                        output_time_steps=output_time_steps,
                                        sequence=sequence, batch_size=64, load=False)

    estimator = TimeSeriesEstimator(dlwp, val_generator)

    # Make a time series prediction
    print('Predicting with model %s...' % model_label)
    time_series = estimator.predict(num_forecast_steps, verbose=1)

    # Scale the time series. Smart enough to align dimensions without expand_dims
    if scale_variables:
        # Scaling parameters
        scale_ds = xr.open_dataset(scale_file)
        sel_mean = scale_ds['mean'].sel(output_selection)
        sel_std = scale_ds['std'].sel(output_selection)
        time_series = time_series * sel_std + sel_mean

    # For some reason the DataArray produced by TimeSeriesEstimator is incompatible with ncview and the remap code.
    # Change the coordinates using a different method.
    fh = np.arange(dt, time_series.shape[0] * dt + 1., dt)
    sequence = 1 if sequence is None else sequence
    time_series = verify.add_metadata_to_forecast_cs(
        time_series.values,
        fh,
        predictor_ds.sel(**output_selection).isel(sample=slice(input_time_steps - 1,
                                                               -output_time_steps * sequence))
    )

    # Save and remap
    print('Remapping from cube sphere...')
    time_series.to_netcdf(forecast_file + '.cs')
    time_series = None

    csr = CubeSphereRemap(to_netcdf4=True)
    csr.assign_maps(*map_files)
    csr.convert_from_faces(forecast_file + '.cs', forecast_file + '.tmp')
    csr.inverse_remap(forecast_file + '.tmp', forecast_file, '--var', 'forecast')
    os.remove(forecast_file + '.tmp')

    # Add the varlev dimension. Do this on disk with netCDF4 – except CubeSphereRemap outputs NETCDF3... :(
    time_series_nc = netCDF4.Dataset(forecast_file, 'a')
    print('Assigning variable/level coordinates...')
    try:
        if 'varlev' in predictor_ds.coords:
            selection = {'varlev': predictor_ds.varlev if not output_selection else output_selection['varlev']}
            varlev = time_series_nc.createVariable('varlev', str, ('varlev',))
            varlev[:] = selection['varlev']
        else:
            selection = {
                'variable': (predictor_ds.variable if 'variable' not in output_selection.keys()
                             else output_selection['variable']),
                'level': (predictor_ds.level if 'level' not in output_selection.keys()
                          else output_selection['level']),
            }
            variable = time_series_nc.createVariable('variable', str, ('variable',))
            variable[:] = selection['variable']
            level = time_series_nc.createVariable('level', str, ('level',))
            level[:] = selection['level']
    except BaseException as e:
        print('Warning: could not assign variable/level string coordinates (%s)' % e)

    # Reverse the latitude coordinate
    if reverse_lat:
        print('Reversing latitude coordinate...')

        time_series_nc.variables['lat'][:] = -1. * time_series_nc.variables['lat'][:]
    time_series_nc.close()


#%% Can re-open to do other things

time_series_ds = xr.open_dataset(forecast_file)
time_series = time_series_ds.forecast

