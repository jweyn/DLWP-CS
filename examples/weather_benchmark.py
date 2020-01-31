#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Build, train, and evaluate a simple neural network to serve as a benchmark model for the weather-benchmark dataset.
This is a baseline model that only uses Z_500 on the 5.625-degree grid. There are two models: a fully-connected model
with 1 hidden layer and a convolutional network with 1 hidden and 1 output convolutional layer.
https://github.com/raspstephan/weather-benchmark
"""

import time
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from DLWP.model import DLWPNeuralNet, SeriesDataGenerator, verify
from DLWP.model.extensions import TimeSeriesEstimator
from DLWP.util import save_model, load_model
from DLWP.custom import EarlyStoppingMin
from tensorflow.keras.callbacks import History, TensorBoard

# Random seed for reproducibility
np.random.seed(0)
tf.compat.v1.set_random_seed(0)


#%% Parameters

# File paths and names
root_directory = '/home/disk/wave2/jweyn/Data/weather-benchmark-test-v2'
predictor_file = os.path.join(root_directory, 'cfs_6h_1979-2010_z500-1000_tau_sfc_NH.nc')
model_file = os.path.join(root_directory, 'weather-benchmark-6h-CNN-1')
log_directory = os.path.join(root_directory, 'logs', 'weather-benchmark-6h-CNN-1')

# NN parameters.
model_is_convolutional = True
min_epochs = 50
max_epochs = 100
patience = 10
batch_size = 128
shuffle = True

# Data parameters. Specify the input variables/levels, output variables/levels, and time steps in/out.
# Ensure that the selections use LISTS of values (even for only 1) to keep dimensions correct.
input_selection = {}
output_selection = {}
input_time_steps = 1
output_time_steps = 1
# Option to crop the north pole. Necessary for getting an even number of latitudes for up-sampling layers.
crop_north_pole = False
# Add incoming solar radiation forcing
add_solar = False

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = 'full'

# Use multiple GPUs, if available
n_gpu = 1

# Train, validation, and test sets, as lists of pd.Timestamp() objects.
frequency = '6H'
train_set = pd.date_range(datetime(1979, 1, 1, 0), datetime(2014, 12, 31, 23), freq=frequency)
validation_set = pd.date_range(datetime(2015, 1, 1, 0), datetime(2016, 12, 31, 23), freq=frequency)
test_set = pd.date_range(datetime(2017, 1, 1, 0), datetime(2018, 12, 31, 23), freq=frequency)

# For outputs
plot_file = 'weather-benchmark-CNN-1.pdf'
plot_label = '1-hidden-layer CNN'


#%% Open data

data = xr.open_mfdataset(['%s/5.625deg/geopotential_500/geopotential_500_%s_5.625deg.nc' % (root_directory, year)
                          for year in range(1979, 2019)])
print('Loading data...')
data.load()
data = data.rename({'z': 'predictors', 'time': 'sample'}).expand_dims('varlev', 1).assign_coords(varlev=['z'])

# Normalize the data based on global mean and std. Can relax the selection of train_set if memory efficiency is needed.
data_mean = np.mean(data.predictors.sel(sample=train_set).values)
data_std = np.std(data.predictors.sel(sample=train_set).values)
data.predictors[:] = (data.predictors[:] - data_mean) / data_std

if crop_north_pole:
    data = data.isel(lat=(data.lat < 90.0))


#%% Build a model and the data generators

model_loaded = False
try:
    dlwp = load_model(model_file)
    model_loaded = True
    print('Loaded existing model %s' % model_file)
except IOError:
    dlwp = DLWPNeuralNet(is_convolutional=model_is_convolutional, time_dim=1, scaler_type=None, scale_targets=False)

# Find the validation set
if validation_set is None:
    if train_set is None:
        train_set = slice(None, None)
    validation_data = None
    train_data = data.sel(sample=train_set)
else:
    validation_data = data.sel(sample=validation_set)
    train_data = data.sel(sample=train_set)

# Build the data generators
generator = SeriesDataGenerator(dlwp, train_data, input_sel=input_selection, output_sel=output_selection,
                                input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                batch_size=batch_size, add_insolation=add_solar, load=load_memory, shuffle=shuffle)

if validation_data is not None:
    val_generator = SeriesDataGenerator(dlwp, validation_data, input_sel=input_selection, output_sel=output_selection,
                                        input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                        batch_size=batch_size, add_insolation=add_solar, load=load_memory)
else:
    val_generator = None


#%% Compile the model structure with some generator data information

if model_is_convolutional:
    # Convolutional neural network
    cs = generator.convolution_shape
    layers = (
        ('PeriodicPadding2D', ((0, 1),), {
            'data_format': 'channels_first',
            'input_shape': cs
        }),
        ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
        ('Conv2D', (32, 3), {
            'padding': 'valid',
            'activation': 'relu',
            'data_format': 'channels_first'
        }),
        ('PeriodicPadding2D', ((0, 1),), {'data_format': 'channels_first'}),
        ('ZeroPadding2D', ((1, 0),), {'data_format': 'channels_first'}),
        ('Conv2D', (1, 3), {
            'padding': 'valid',
            'activation': 'linear',
            'data_format': 'channels_first'
        })
    )
else:
    # Fully-connected neural network
    layers = (
        ('Dense', (generator.n_features,), {
            'input_shape': generator.dense_shape,
            'activation': 'linear'
        }),
        # ('Dense', (generator.n_features,), {
        #     'activation': 'linear'
        # }),
    )

# Build the model
if not model_loaded:
    try:
        dlwp.build_model(layers, loss='mse', optimizer='adam', metrics=['mae'], gpus=n_gpu)
    except ValueError:
        for layer in dlwp.base_model.layers:
            print(layer.name, layer.output_shape)
        raise
print(dlwp.base_model.summary())


#%% Train and save the model

if not model_loaded:
    # Preliminaries
    start_time = time.time()
    print('Begin training...')
    history = History()
    early = EarlyStoppingMin(min_epochs=min_epochs, monitor='val_loss' if val_generator is not None else 'loss',
                             min_delta=0., patience=patience, restore_best_weights=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_directory, batch_size=batch_size, update_freq='epoch')

    # Fit the model
    dlwp.fit_generator(generator, epochs=max_epochs, verbose=1, validation_data=val_generator,
                       use_multiprocessing=True, callbacks=[history, early])
    end_time = time.time()

    # Save the model
    if model_file is not None:
        save_model(dlwp, model_file, history=history)
        print('Wrote model %s' % model_file)

    # Print training summary
    print("\nTrain time -- %s seconds --" % (end_time - start_time))
    if validation_data is not None:
        score = dlwp.evaluate(*val_generator.generate([], scale_and_impute=False), verbose=0)
        print('Validation loss:', score[0])
        print('Validation mean absolute error:', score[1])
else:
    print('Skipping training...')


#%% Evaluate the model on continuous time series forecasts

dt = int(frequency[0])
num_forecast_steps = 24 * 5 // dt
test_set_6hour = [t for t in test_set if t.hour % 6 == 0]
print('Testing model...')

# Generate verification and scale back to raw units
test_data = data.sel(sample=test_set)
verification = verify.verification_from_series(test_data.sel(varlev='z'),
                                               forecast_steps=num_forecast_steps, dt=dt)
verification = verification.sel(time=test_set_6hour)
lat_weights = np.cos(np.deg2rad(verification.lat))  # broadcasts automatically to all DataArrays
lat_weights_norm = lat_weights / lat_weights.mean()
verification = (verification * data_std + data_mean) * lat_weights_norm

# Generate a TimeSeriesEstimator to predict from the model
test_generator = SeriesDataGenerator(dlwp, test_data.sel(sample=test_set_6hour),
                                     input_sel=input_selection, output_sel=output_selection,
                                     input_time_steps=input_time_steps, output_time_steps=output_time_steps,
                                     batch_size=batch_size, add_insolation=add_solar, load=load_memory)
estimator = TimeSeriesEstimator(dlwp, test_generator)
estimator._dt = test_data.sample[1] - test_data.sample[0]

# Predict with the estimator and scale
time_series = estimator.predict(num_forecast_steps, verbose=1)
time_series = time_series.sel(varlev='z', time=test_set_6hour[:-1])
time_series = (time_series * data_std + data_mean) * lat_weights_norm

# Calculate RMSE
intersection = np.intersect1d(time_series.time.values, verification.time.values, assume_unique=True)
rmse = verify.forecast_error(time_series.sel(time=intersection).values,
                             verification.isel(f_hour=slice(0, len(time_series.f_hour))).sel(time=intersection).values,
                             method='rmse')


#%% Persistence and climatology

print('Calculating persistence forecasts...')
init = test_data['predictors'].sel(varlev='z')
init = (init * data_std + data_mean) * lat_weights_norm

persist_rmse = verify.forecast_error(np.repeat(init.values[None, ...], num_forecast_steps, axis=0),
                                     verification.values, method='rmse')

print('Calculating climatology forecasts...')
climo_data = data['predictors'].sel(varlev='z')
climo_data = (climo_data * data_std + data_mean) * lat_weights_norm
climo_rmse = verify.monthly_climo_error(climo_data, validation_set, n_fhour=num_forecast_steps, method='rmse')


#%% Plot

fig = plt.figure()
fig.set_size_inches(6, 4)

plt.plot(time_series.f_hour, rmse, label=plot_label, linewidth=2.)
plt.plot(np.arange(dt, dt * num_forecast_steps + 1, dt), persist_rmse, label='persistence', linewidth=2.)
plt.plot(np.arange(dt, dt * num_forecast_steps + 1, dt), climo_rmse, label='climatology', linewidth=2.)

plt.xlim([0, dt * num_forecast_steps])
plt.xticks(np.arange(0, num_forecast_steps * dt + 1, 4 * dt))
plt.ylim([0, 2000])
plt.yticks(np.arange(0, 2001, 200))
plt.legend(loc='best', fontsize=8)
plt.grid(True, color='lightgray', zorder=-100)
plt.xlabel('forecast lead time (h)')
plt.ylabel('z rmse (m$^2$ s$^{-2}$)')
plt.title('2017-18 6-hourly forecast error')
plt.savefig(plot_file, bbox_inches='tight')
plt.show()

