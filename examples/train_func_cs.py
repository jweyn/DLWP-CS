#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Example of training a DLWP model on the cubed sphere with the Keras functional API.
"""

import time
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from DLWP.model import DLWPFunctional, ArrayDataGenerator
from DLWP.model.preprocessing import get_constants, prepare_data_array
from DLWP.util import save_model
from keras.callbacks import History, TensorBoard

from keras.layers import Input, MaxPooling3D, UpSampling3D, AveragePooling3D, concatenate, ReLU, Reshape, Concatenate
from DLWP.custom import CubeSpherePadding2D, CubeSphereConv2D, RNNResetStates, EarlyStoppingMin, SaveWeightsOnEpoch
from keras.models import Model
import keras.backend as K


# Set a TF session with memory growth

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


#%% Parameters

# File paths and names
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = os.path.join(root_directory, 'era5_2deg_3h_CS_1979-2018_z-tau-t2_500-1000_tcwv.nc')
model_file = os.path.join(root_directory, 'dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2')
log_directory = os.path.join(root_directory, 'logs', 'era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2')
reverse_lat = False

# Optional paths to files containing constant fields to add to the inputs
constant_fields = [
    (os.path.join(root_directory, 'era5_2deg_3h_CS_land_sea_mask.nc'), 'lsm'),
    (os.path.join(root_directory, 'era5_2deg_3h_CS_scaled_topo.nc'), 'z')
]

# NN parameters. Regularization is applied to LSTM layers by default. weight_loss indicates whether to weight the
# loss function preferentially in the mid-latitudes.
model_is_convolutional = True
min_epochs = 100
max_epochs = 1000
patience = 50
batch_size = 64
lambda_ = 1.e-4
loss_by_step = None
shuffle = True
skip_connections = True

# Data parameters. Specify the input/output variables/levels and input/output time steps. DLWPFunctional requires that
# the inputs and outputs match exactly (for now). Ensure that the selections use LISTS of values (even for only 1) to
# keep dimensions correct. The number of output iterations to train on is given by integration_steps. The actual number
# of forecast steps (units of model delta t) is io_time_steps * integration_steps. The parameter data_interval
# governs what the effective delta t is; it is a multiplier for the temporal resolution of the data file.
io_selection = {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0']}
io_time_steps = 2
integration_steps = 2
data_interval = 2
# Add incoming solar radiation forcing
add_solar = True

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = 'minimal'

# Use multiple GPUs, if available
n_gpu = 1

# Force use of the keras model.fit() method. May run faster in some instances, but uses (input_time_steps +
# output_time_steps) times more memory.
use_keras_fit = False

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2013, 1, 1, 0), datetime(2016, 12, 31, 18), freq='3H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 0), datetime(2012, 12, 31, 18), freq='3H'))


#%% Open data

data = xr.open_dataset(predictor_file, chunks={'sample': 1})
# Fix negative latitude for solar radiation input
if reverse_lat:
    data.lat.load()
    data.lat[:] = -1. * data.lat.values

has_constants = not(not constant_fields)
constants = get_constants(constant_fields or None)


#%% Create a model and the data generators

dlwp = DLWPFunctional(is_convolutional=model_is_convolutional, is_recurrent=False, time_dim=io_time_steps)

# Find the validation set
if train_set is None:
    train_set = np.isin(data.sample.values, np.array(validation_set, dtype='datetime64[ns]'),
                        assume_unique=True, invert=True)
validation_data = data.sel(sample=validation_set)
train_data = data.sel(sample=train_set)

# Build the data generators
print('Loading data to memory...')
start_time = time.time()
train_array, input_ind, output_ind, sol = prepare_data_array(train_data, input_sel=io_selection,
                                                             output_sel=io_selection, add_insolation=add_solar)
generator = ArrayDataGenerator(dlwp, train_array, rank=3, input_slice=input_ind, output_slice=output_ind,
                               input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                               sequence=integration_steps, interval=data_interval, insolation_array=sol,
                               batch_size=batch_size, shuffle=shuffle, constants=constants)
if use_keras_fit:
    p_train, t_train = generator.generate([])
if validation_data is not None:
    print('Loading validation data to memory...')
    val_array, input_ind, output_ind, sol = prepare_data_array(validation_data, input_sel=io_selection,
                                                               output_sel=io_selection, add_insolation=add_solar)
    val_generator = ArrayDataGenerator(dlwp, val_array, rank=3, input_slice=input_ind, output_slice=output_ind,
                                       input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                       sequence=integration_steps, interval=data_interval, insolation_array=sol,
                                       batch_size=batch_size, shuffle=shuffle, constants=constants)
    if use_keras_fit:
        val = val_generator.generate([])
else:
    val_generator = None
    if use_keras_fit:
        val = None

total_time = time.time() - start_time
print('Time to load data: %d m %0.2f s' % (np.floor(total_time / 60), total_time % 60))


#%% Compile the model structure with some generator data information

# Up-sampling convolutional network or U-net
cs = generator.convolution_shape
cso = generator.output_convolution_shape
input_solar = (integration_steps > 1 and (isinstance(add_solar, str) or add_solar))

# Define layers. Must be defined outside of model function so we use the same weights at each integration step.
main_input = Input(shape=cs, name='main_input')
if input_solar:
    solar_inputs = [Input(shape=generator.insolation_shape, name='solar_%d' % d) for d in range(1, integration_steps)]
if has_constants:
    constant_input = Input(shape=constants.shape, name='constants')
cube_padding_1 = CubeSpherePadding2D(1, data_format='channels_first')
pooling_2 = AveragePooling3D((2, 2, 1), data_format='channels_first')
up_sampling_2 = UpSampling3D((2, 2, 1), data_format='channels_first')
relu = ReLU(negative_slope=0.1)
conv_2d_1 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_1_2 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_2 = CubeSphereConv2D(64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_2_2 = CubeSphereConv2D(64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_3 = CubeSphereConv2D(128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_3_2 = CubeSphereConv2D(128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_4 = CubeSphereConv2D(128 if skip_connections else 256, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_4_2 = CubeSphereConv2D(256, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_5 = CubeSphereConv2D(64 if skip_connections else 128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_5_2 = CubeSphereConv2D(128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_6 = CubeSphereConv2D(32 if skip_connections else 64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_6_2 = CubeSphereConv2D(64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_7 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_7_2 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
conv_2d_8 = CubeSphereConv2D(cso[0], 1, **{
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })


# Define the model functions.

def basic_model(x):
    x = cube_padding_1(x)
    x = relu(conv_2d_1(x))
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_2(x))
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_3(x))
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_6(x))
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_7(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_7_2(x))
    x = conv_2d_8(x)
    return x


def unet(x):
    x0 = cube_padding_1(x)
    x0 = relu(conv_2d_1(x0))
    x1 = pooling_2(x0)
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2(x1))
    x2 = pooling_2(x1)
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_3(x2))
    x2 = up_sampling_2(x2)
    x = concatenate([x2, x1], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_6(x))
    x = up_sampling_2(x)
    x = concatenate([x, x0], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_7(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_7_2(x))
    x = conv_2d_8(x)
    return x


def unet2(x):
    x0 = cube_padding_1(x)
    x0 = relu(conv_2d_1(x0))
    x0 = cube_padding_1(x0)
    x0 = relu(conv_2d_1_2(x0))
    x1 = pooling_2(x0)
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2(x1))
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2_2(x1))
    x2 = pooling_2(x1)
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_5_2(x2))
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_5(x2))
    x2 = up_sampling_2(x2)
    x = concatenate([x2, x1], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_6_2(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_6(x))
    x = up_sampling_2(x)
    x = concatenate([x, x0], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_7(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_7_2(x))
    x = conv_2d_8(x)
    return x


def unet3(x):
    x0 = cube_padding_1(x)
    x0 = relu(conv_2d_1(x0))
    x0 = cube_padding_1(x0)
    x0 = relu(conv_2d_1_2(x0))
    x1 = pooling_2(x0)
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2(x1))
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2_2(x1))
    x2 = pooling_2(x1)
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_3_2(x2))
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_3(x2))
    x3 = pooling_2(x2)
    x3 = cube_padding_1(x3)
    x3 = relu(conv_2d_4_2(x3))
    x3 = cube_padding_1(x3)
    x3 = relu(conv_2d_4(x3))
    x3 = up_sampling_2(x3)
    x = concatenate([x3, x2], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_5_2(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_5(x))
    x = up_sampling_2(x)
    x = concatenate([x, x1], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_6_2(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_6(x))
    x = up_sampling_2(x)
    x = concatenate([x, x0], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_7(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_7_2(x))
    x = conv_2d_8(x)
    return x


def complete_model(x_in):
    outputs = []
    model_function = unet2 if skip_connections else basic_model
    is_seq = isinstance(x_in, (list, tuple))
    xi = x_in[0] if is_seq else x_in
    if is_seq and has_constants:
        xi = Concatenate(axis=1)([xi, x_in[-1]])
    outputs.append(model_function(xi))
    for step in range(1, integration_steps):
        xo = outputs[step - 1]
        if is_seq and input_solar:
            xo = Reshape(generator.shape)(xo)
            xo = Concatenate(axis=2)([xo, x_in[step]])
            xo = Reshape(cs)(xo)
        if is_seq and has_constants:
            xo = Concatenate(axis=1)([xo, x_in[-1]])
        outputs.append(model_function(xo))

    return outputs


# Build the model with inputs and outputs
if not input_solar and not has_constants:
    inputs = main_input
else:
    inputs = [main_input]
    if input_solar:
        inputs = inputs + solar_inputs
    if has_constants:
        inputs = inputs + [constant_input]
model = Model(inputs=inputs, outputs=complete_model(inputs))

# No weighted loss available for cube sphere at the moment, but we can weight each integration sequence
loss_function = 'mse'
if loss_by_step is None:
    loss_by_step = [1./integration_steps] * integration_steps

# Build the DLWP model
dlwp.build_model(model, loss=loss_function, loss_weights=loss_by_step, optimizer='adam', metrics=['mae'], gpus=n_gpu)
print(dlwp.base_model.summary())


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
print('Begin training...')
# run = Run.get_context()
history = History()
early = EarlyStoppingMin(monitor='val_loss' if val_generator is not None else 'loss', min_delta=0.,
                         min_epochs=min_epochs, max_epochs=max_epochs, patience=patience,
                         restore_best_weights=True, verbose=1)
tensorboard = TensorBoard(log_dir=log_directory, batch_size=batch_size, update_freq='epoch')
save = SaveWeightsOnEpoch(weights_file=model_file + '.keras.tmp', interval=25)

if use_keras_fit:
    dlwp.fit(p_train, t_train, batch_size=batch_size, epochs=max_epochs+1, verbose=1, validation_data=val,
             callbacks=[history, RNNResetStates(), early])
else:
    dlwp.fit_generator(generator, epochs=max_epochs+1, verbose=1, validation_data=val_generator,
                       use_multiprocessing=True, workers=4, callbacks=[history, RNNResetStates(), early, save])
end_time = time.time()

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)
    print('Wrote model %s' % model_file)

# Evaluate the model
print("\nTrain time -- %s seconds --" % (end_time - start_time))
if validation_data is not None:
    score = dlwp.evaluate(*val_generator.generate([]), verbose=0)
    print('Validation loss:', score[0])
    print('Validation mean absolute error:', score[1])
