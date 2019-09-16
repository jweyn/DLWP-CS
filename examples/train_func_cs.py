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
from DLWP.model import DLWPFunctional, SeriesDataGenerator
from DLWP.util import save_model, train_test_split_ind
from keras.callbacks import History, TensorBoard

from keras.layers import Input, MaxPooling3D, UpSampling3D, AveragePooling3D, concatenate, ReLU, Reshape, Concatenate
from DLWP.custom import CubeSpherePadding2D, CubeSphereConv2D, RNNResetStates, EarlyStoppingMin, slice_layer

from keras.regularizers import l2
from keras.models import Model


#%% Parameters

# File paths and names
root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = os.path.join(root_directory, 'cfs_6h_CS48_1979-2010_z500_tau300-700.nc')
model_file = os.path.join(root_directory, 'dlwp_6h_CS48_tau-sol_T2_unet01-leaky-avg')
log_directory = os.path.join(root_directory, 'logs', 'CS48-tau-sol-T2-unet01-leaky-avg')

# NN parameters. Regularization is applied to LSTM layers by default. weight_loss indicates whether to weight the
# loss function preferentially in the mid-latitudes.
model_is_convolutional = True
min_epochs = 100
max_epochs = 200
patience = 50
batch_size = 64
lambda_ = 1.e-4
loss_by_step = None
shuffle = True
skip_connections = True

# Data parameters. Specify the input/output variables/levels and input/output time steps. DLWPFunctional requires that
# the inputs and outputs match exactly (for now). Ensure that the selections use LISTS of values (even for only 1) to
# keep dimensions correct. The number of output iterations to train on is given by integration_steps. The actual number
# of forecast steps (units of model delta t) is io_time_steps * integration_steps.
io_selection = {'varlev': ['HGT/500', 'THICK/300-700']}
io_time_steps = 2
integration_steps = 2
# Add incoming solar radiation forcing
add_solar = True

# If system memory permits, loading the predictor data can greatly increase efficiency when training on GPUs, if the
# train computation takes less time than the data loading.
load_memory = True

# Use multiple GPUs, if available
n_gpu = 1

# Force use of the keras model.fit() method. May run faster in some instances, but uses (input_time_steps +
# output_time_steps) times more memory.
use_keras_fit = False

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2003, 1, 1, 0), datetime(2006, 12, 31, 18), freq='6H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 6), datetime(2002, 12, 31, 18), freq='6H'))


#%% Open data

data = xr.open_dataset(predictor_file, chunks={'sample': batch_size})
# Fix negative latitude for solar radiation input
data.lat.load()
data.lat[:] = -1. * data.lat.values

if 'time_step' in data.dims:
    time_dim = data.dims['time_step']
else:
    time_dim = 1
n_sample = data.dims['sample']


#%% Create a model and the data generators

dlwp = DLWPFunctional(is_convolutional=model_is_convolutional, is_recurrent=False, time_dim=io_time_steps)

# Find the validation set
if isinstance(validation_set, int):
    n_sample = data.dims['sample']
    ts, val_set = train_test_split_ind(n_sample, validation_set, method='last')
    if train_set is None:
        train_set = ts
    elif isinstance(train_set, int):
        train_set = list(range(train_set))
    validation_data = data.isel(sample=val_set)
    train_data = data.isel(sample=train_set)
elif validation_set is None:
    if train_set is None:
        train_set = data.sample.values
    validation_data = None
    train_data = data.sel(sample=train_set)
else:  # we must have a list of datetimes
    if train_set is None:
        train_set = np.isin(data.sample.values, np.array(validation_set, dtype='datetime64[ns]'),
                            assume_unique=True, invert=True)
    validation_data = data.sel(sample=validation_set)
    train_data = data.sel(sample=train_set)

# Build the data generators
if load_memory or use_keras_fit:
    print('Loading data to memory...')
generator = SeriesDataGenerator(dlwp, train_data, rank=3, input_sel=io_selection, output_sel=io_selection,
                                input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                sequence=integration_steps, add_insolation=add_solar,
                                batch_size=batch_size, load=load_memory, shuffle=shuffle)
if use_keras_fit:
    p_train, t_train = generator.generate([])
if validation_data is not None:
    val_generator = SeriesDataGenerator(dlwp, validation_data, input_sel=io_selection, output_sel=io_selection,
                                        rank=3, input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                        sequence=integration_steps, add_insolation=add_solar,
                                        batch_size=batch_size, load=load_memory)
    if use_keras_fit:
        val = val_generator.generate([])
else:
    val_generator = None
    if use_keras_fit:
        val = None


#%% Compile the model structure with some generator data information

# Up-sampling convolutional network with LSTM layer
cs = generator.convolution_shape
cso = generator.output_convolution_shape
input_seq = (integration_steps > 1 and add_solar)

# Define layers. Must be defined outside of model function so we use the same weights at each integration step.
input_0 = Input(shape=cs, name='input_0')
if input_seq:
    more_inputs = [Input(shape=generator.insolation_shape, name='input_%d' % d) for d in range(1, integration_steps)]
cube_padding_1 = CubeSpherePadding2D(1, data_format='channels_first')
pooling_2 = AveragePooling3D((2, 2, 1), data_format='channels_first')
up_sampling_2 = UpSampling3D((2, 2, 1), data_format='channels_first')
relu = ReLU(negative_slope=0.1)
conv_2d_1 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(lambda_)
    })
# batch_norm_1 = BatchNormalization(axis=1)
conv_2d_1_2 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(lambda_)
    })
conv_2d_2 = CubeSphereConv2D(64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(3. * lambda_)
    })
# batch_norm_2 = BatchNormalization(axis=1)
conv_2d_3 = CubeSphereConv2D(64 if skip_connections else 128, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(lambda_)
    })
# batch_norm_3 = BatchNormalization(axis=1)
conv_2d_4 = CubeSphereConv2D(32 if skip_connections else 64, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(3. * lambda_)
    })
# batch_norm_4 = BatchNormalization(axis=1)
conv_2d_5 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(10. * lambda_)
    })
# batch_norm_5 = BatchNormalization(axis=1)
conv_2d_5_2 = CubeSphereConv2D(32, 3, **{
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first',
        # 'kernel_regularizer': l2(10. * lambda_)
    })
conv_2d_6 = CubeSphereConv2D(cso[0], 1, **{
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_first'
    })
split_1_1 = slice_layer(0, 16, axis=1)
split_1_2 = slice_layer(16, 32, axis=1)
split_2_1 = slice_layer(0, 32, axis=1)
split_2_2 = slice_layer(32, 64, axis=1)


# Define the model functions.

def basic_model(x):
    x = cube_padding_1(x)
    x = relu(conv_2d_1(x))
    # x = cube_padding_1(x)
    # x = relu(conv_2d_1_2(x))
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_2(x))
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_3(x))
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_4(x))
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_5(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_5_2(x))
    x = conv_2d_6(x)
    return x


def skip_model(x):
    x = cube_padding_1(x)
    x = relu(conv_2d_1(x))
    # x = cube_padding_1(x)
    # x = relu(conv_2d_1_2(x))
    x, x1 = split_1_1(x), split_1_2(x)
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_2(x))
    x, x2 = split_2_1(x), split_2_2(x)
    x = pooling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_3(x))
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_4(x))
    x = concatenate([x, x2], axis=1)
    x = up_sampling_2(x)
    x = cube_padding_1(x)
    x = relu(conv_2d_5(x))
    x = concatenate([x, x1], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_5_2(x))
    x = conv_2d_6(x)
    return x


def unet(x):
    x0 = cube_padding_1(x)
    x0 = relu(conv_2d_1(x0))
    # x0 = cube_padding_1(x0)
    # x0 = relu(conv_2d_1_2(x0))
    x1 = pooling_2(x0)
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2(x1))
    x2 = pooling_2(x1)
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_3(x2))
    x2 = up_sampling_2(x2)
    x = concatenate([x2, x1], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_4(x))
    x = up_sampling_2(x)
    x = concatenate([x, x0], axis=1)
    x = cube_padding_1(x)
    x = relu(conv_2d_5(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_5_2(x))
    x = conv_2d_6(x)
    return x


def complete_model(x_in):
    outputs = []
    model_function = skip_model if skip_connections else basic_model
    is_seq = isinstance(x_in, (list, tuple))
    outputs.append(model_function(x_in[0] if is_seq else x_in))
    for step in range(1, integration_steps):
        xo = outputs[step - 1]
        if is_seq:
            xo = Reshape(generator.shape)(xo)
            xo = Concatenate(axis=2)([xo, x_in[step]])
            xo = Reshape(cs)(xo)
        outputs.append(model_function(xo))

    return outputs


# Build the model with inputs and outputs
inputs = [input_0] + more_inputs if input_seq else input_0
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

if use_keras_fit:
    dlwp.fit(p_train, t_train, batch_size=batch_size, epochs=max_epochs+1, verbose=1, validation_data=val,
             callbacks=[history, RNNResetStates(), early])
else:
    dlwp.fit_generator(generator, epochs=max_epochs+1, verbose=1, validation_data=val_generator,
                       use_multiprocessing=True, callbacks=[history, RNNResetStates(), early])
end_time = time.time()

# Save the model
if model_file is not None:
    save_model(dlwp, model_file, history=history)
    print('Wrote model %s' % model_file)

# Evaluate the model
print("\nTrain time -- %s seconds --" % (end_time - start_time))
if validation_data is not None:
    score = dlwp.evaluate(*val_generator.generate([], scale_and_impute=False), verbose=0)
    print('Validation loss:', score[0])
    print('Validation mean absolute error:', score[1])
