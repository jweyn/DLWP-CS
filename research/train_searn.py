#
# Copyright (c) 2020 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Train a DLWP sequence model using SEARN method to fine-tune weights for improved stability. In this method, the
original model weights are used to make a prediction of some part of the sequence and the new model is trained on
partial prediction data and partial real data to improve recovery from bad early sequence models. See:
http://users.umiacs.umd.edu/~hal/docs/daume06searn-practice.pdf
"""

import time
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from DLWP.model import ArrayDataGeneratorWithInference, tf_data_generator
from DLWP.model.preprocessing import get_constants, prepare_data_array
from DLWP.util import save_model, load_model
from tensorflow.keras.callbacks import History, TensorBoard
from DLWP.custom import RNNResetStates, EarlyStoppingMin, SaveWeightsOnEpoch, GeneratorEpochEnd
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
# Disable warning logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Set GPU for train and inference
train_gpu = 1
inference_gpu = 2
devices = [tf.config.list_physical_devices('GPU')[i] for i in [train_gpu, inference_gpu]]
tf.config.set_visible_devices(devices, 'GPU')
# Allow memory growth
tf.config.experimental.set_memory_growth(devices[0], True)
tf.config.experimental.set_memory_growth(devices[1], True)


#%% Parameters

# File paths and names
root_directory = '/home/rhodium2/jweyn/Data'
predictor_file = os.path.join(root_directory, 'era5_2deg_3h_CS2_1979-2018_z-tau-t2_500-1000_tcwv_psi-uT42-850.nc')
model_file = os.path.join(root_directory, 'dlwp-cs-s2s_4var-tcwv_UNET2-48-searn')
log_directory = os.path.join(root_directory, 'logs', 'dlwp-cs-s2s_4var-tcwv_UNET2-searn')
input_model = '/home/disk/wave2/jweyn/Data/DLWP/dlwp_era5_6h-3_CS48_tau-sfc1000-tcwv-lsm-topo_UNET2-48-relumax'

# Optional paths to files containing constant fields to add to the inputs
constant_fields = [
    (os.path.join(root_directory, 'era5_2deg_3h_CS2_land_sea_mask.nc'), 'lsm'),
    (os.path.join(root_directory, 'era5_2deg_3h_CS2_scaled_topo.nc'), 'z')
]

# Parameters for the CNN
min_epochs = 0
max_epochs = 200
patience = 50
batch_size = 64
loss_by_step = None
shuffle = True
searn_steps = [0]

# Data parameters. Specify the input/output variables/levels and input/output time steps. DLWPFunctional requires that
# the inputs and outputs match exactly (for now). Ensure that the selections use LISTS of values (even for only 1) to
# keep dimensions correct. The number of output iterations to train on is given by integration_steps. The actual number
# of forecast steps (units of model delta t) is io_time_steps * integration_steps. The parameter data_interval
# governs what the effective delta t is; it is a multiplier for the temporal resolution of the data file.
io_selection = {'varlev': ['z/500', 'tau/300-700', 'z/1000', 't2m/0', 'tcwv/0']}
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

# Optimize the optimizer for GPU tensor cores by using mixed precision
use_mp_optimizer = True

# Validation set to use. Either an integer (number of validation samples, taken from the end), or an iterable of
# pandas datetime objects. The train set can be set to the first <integer> samples, an iterable of dates, or None to
# simply use the remaining points. Match the type of validation_set.
validation_set = list(pd.date_range(datetime(2013, 1, 1, 0), datetime(2016, 12, 31, 18), freq='3H'))
train_set = list(pd.date_range(datetime(1979, 1, 1, 0), datetime(2012, 12, 31, 18), freq='3H'))


#%% Open data

data = xr.open_dataset(predictor_file)

has_constants = not(not constant_fields)
constants = get_constants(constant_fields or None)

# Find the validation set
if train_set is None:
    train_set = np.isin(data.sample.values, np.array(validation_set, dtype='datetime64[ns]'),
                        assume_unique=True, invert=True)
validation_data = data.sel(sample=validation_set)
train_data = data.sel(sample=train_set)


#%% Load the model twice, once for inference, and once to compile and fit

print('Loading model...')
dlwp = load_model(input_model)
with tf.device('/gpu:1'):  # 1 will always be the second, specified as inference, GPU
    dlwp_inf = load_model(input_model)


#%% Build the data generators

print('Loading data to memory...')
start_time = time.time()

# Pre-process an array form of the data with the DLWP module tool, and create a generator
train_array, input_ind, output_ind, sol = prepare_data_array(train_data, input_sel=io_selection,
                                                             output_sel=io_selection, add_insolation=add_solar)
generator = ArrayDataGeneratorWithInference(dlwp_inf, searn_steps, dlwp, train_array, rank=3,
                                            input_slice=input_ind, output_slice=output_ind,
                                            input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                            sequence=integration_steps, interval=data_interval, insolation_array=sol,
                                            batch_size=batch_size, shuffle=shuffle, constants=constants,
                                            channels_last=True, drop_remainder=True)

# Create a tf_data_generator. To use this, we need the model's input and output names. Hack-y way of doing that
input_names = [re.split('[/:]', i.name)[0] for i in dlwp.model.input]
output_names = [re.split('[/:]', i.name)[0] for i in dlwp.model.output]
tf_train_data = tf_data_generator(generator, batch_size=batch_size, input_names=input_names, output_names=output_names)

# Do the same for the validation data
if validation_data is not None:
    print('Loading validation data to memory...')
    val_array, input_ind, output_ind, sol = prepare_data_array(validation_data, input_sel=io_selection,
                                                               output_sel=io_selection, add_insolation=add_solar)
    val_generator = ArrayDataGeneratorWithInference(dlwp_inf, searn_steps, dlwp, val_array, rank=3,
                                                    input_slice=input_ind, output_slice=output_ind,
                                                    input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                                    sequence=integration_steps, interval=data_interval,
                                                    insolation_array=sol, batch_size=8*batch_size, shuffle=False,
                                                    constants=constants, channels_last=True)
    tf_val_data = tf_data_generator(val_generator, input_names=input_names, output_names=output_names)
else:
    tf_val_data = None

total_time = time.time() - start_time
print('Time to load data: %d m %0.2f s' % (np.floor(total_time / 60), total_time % 60))


#%% Compile the model structure with some generator data information

# No weighted loss available for cube sphere at the moment, but we can weight each integration sequence
loss_function = 'mse'
if loss_by_step is None:
    loss_by_step = [1./integration_steps] * integration_steps

# Build the DLWP model
opt = Adam(learning_rate=1.e-4)
if use_mp_optimizer:
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
dlwp.build_model(dlwp.model, loss=loss_function, loss_weights=loss_by_step, optimizer=opt, metrics=['mae'], gpus=n_gpu)
print(dlwp.base_model.summary())


#%% Train, evaluate, and save the model

# Train and evaluate the model
start_time = time.time()
print('Begin training...')
history = History()
early = EarlyStoppingMin(monitor='val_loss' if validation_data is not None else 'loss', min_delta=0.,
                         min_epochs=min_epochs, max_epochs=max_epochs, patience=patience,
                         restore_best_weights=True, verbose=1)
tensorboard = TensorBoard(log_dir=log_directory, update_freq='epoch')
save = SaveWeightsOnEpoch(weights_file=model_file + '.keras.tmp', interval=25)

dlwp.fit_generator(tf_train_data, epochs=max_epochs + 1,
                   verbose=1, validation_data=tf_val_data,
                   callbacks=[history, RNNResetStates(), early, save, GeneratorEpochEnd(generator)])
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
