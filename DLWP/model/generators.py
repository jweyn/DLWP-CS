#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level APIs for building data generators. These produce batches of data on-the-fly for DLWP models'
fit_generator() methods.
"""

import warnings
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from ..util import delete_nan_samples, insolation, to_bool


class DataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data.
    """

    def __init__(self, model, ds, batch_size=32, shuffle=False, remove_nan=True):
        """
        Initialize a DataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attributes 'predictors' and 'targets'
        :param batch_size: int: number of samples to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        """
        self.model = model
        if not hasattr(ds, 'predictors') or not hasattr(ds, 'targets'):
            raise ValueError("dataset must have 'predictors' and 'targets' variables")
        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._is_convolutional = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._n_sample = ds.dims['sample']
        self._has_time_step = 'time_step' in ds.dims

        self.on_epoch_end()

    @property
    def shape(self):
        """
        :return: the full shape of predictors, (time_step, [variable, level,] lat, lon)
        """
        if self._has_time_step:
            return self.ds.predictors.shape[1:]
        else:
            return (1,) + self.ds.predictors.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of features in the predictor array
        """
        return int(np.prod(self.shape))

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (int(np.prod(self.shape[1:-2])),) + self.shape[-2:]
        else:
            return (int(np.prod(self.shape[:-2])),) + self.ds.predictors.shape[-2:]

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples, scale_and_impute=True):
        if len(samples) > 0:
            ds = self.ds.isel(sample=samples)
        else:
            ds = self.ds.isel(sample=slice(None))
        n_sample = ds.predictors.shape[0]
        p = ds.predictors.values.reshape((n_sample, -1))
        t = ds.targets.values.reshape((n_sample, -1))
        ds.close()
        ds = None

        # Remove samples with NaN; scale and impute
        if self._remove_nan:
            p, t = delete_nan_samples(p, t)
        if scale_and_impute:
            if self._impute_missing:
                p, t = self.model.imputer_transform(p, t)
            p, t = self.model.scaler_transform(p, t)

        # Format spatial shape for convolutions; also takes care of time axis
        if self._is_convolutional:
            p = p.reshape((n_sample,) + self.convolution_shape)
            t = t.reshape((n_sample,) + self.convolution_shape)
        elif self._keep_time_axis:
            p = p.reshape((n_sample,) + self.dense_shape)
            t = t.reshape((n_sample,) + self.dense_shape)

        return p, t

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        return int(np.ceil(self._n_sample / self._batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return: (ndarray, ndarray): predictors, targets
        """
        # Generate indexes of the batch
        if int(index) < 0:
            index = len(self) + index
        if index > len(self):
            raise IndexError
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y


class SeriesDataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data. This class expects DataSet to contain a single variable,
    'predictors', which is a continuous time sequence of weather data. The user supplies arguments to load specific
    variables/levels and the number of time steps for the inputs/outputs. It is highly recommended to use the option
    to load the data into memory if enough memory is available as the increased I/O calls for generating the correct
    data sequences will take a toll. This class also makes it possible to add model-invariant data, such as incoming
    solar radiation, to the inputs.
    """

    def __init__(self, model, ds, rank=2, input_sel=None, output_sel=None, input_time_steps=1, output_time_steps=1,
                 sequence=None, interval=1, add_insolation=False, batch_size=32, shuffle=False, remove_nan=True,
                 load='required', delay_load=False, constants=None, channels_last=False, drop_remainder=False):
        """
        Initialize a SeriesDataGenerator.

        :param model: instance of a DLWP model
        :param ds: xarray Dataset: predictor dataset. Should have attribute 'predictors'.
        :param rank: int: the number of spatial dimensions (e.g. 2 for 2-d data and convolutions)
        :param input_sel: dict: variable/level selection for input features
        :param output_sel: dict: variable/level selection for output features
        :param input_time_steps: int: number of time steps in the input features
        :param output_time_steps: int: number of time steps in the output features (recommended either 1 or the same
            as input_time_steps)
        :param sequence: int or None: if int, then the output targets is a list of sequence consecutive forecast steps.
            Note that in this mode, if add_insolation is True, the inputs are also a list of consecutive forecast steps,
            with the first step containing all of the input data and subsequent steps containing only the requisite
            insolation fields.
        :param interval: int: the number of steps to take between data samples and within input/output time steps.
            Effectively it is the model delta t multiplier for the data resolution.
        :param add_insolation: bool or str:
            if False: do not add incoming solar radiation
            if True: add insolation to the inputs. Incompatible with 3-d convolutions.
            if 'hourly': same as True
            if 'daily': add the daily max insolation without diurnal cycle
        :param batch_size: int: number of samples to take at a time from the dataset
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        :param load: str: option for loading data into memory. If it evaluates to negative, no memory loading is done.
            THIS IS LIKELY VERY SLOW.
            'full': load the full dataset. May use a lot of memory.
            'required': load only the required variables, but this also loads two separate datasets for predictors and
                targets
            'minimal': load only one copy of the data, but also loads all of the variables. This may use half as much
                memory as 'required', but only if there are no unused extra variables in the file. Note that in order
                to attempt to use numpy views to save memory, the order of variables may be different from the
                input and output selections.
        :param delay_load: if True, delay the loading of the data until the first call to generate()
        :param constants: ndarray: additional constant fields to add to each input. Must match the spatial dimensions
            (last `rank` dimensions) of the input data.
        :param channels_last: bool: if True, returns data with channels as the last dimension. May slow down processing
            of data, but may speed up GPU operations on the data.
        :param drop_remainder: bool: if True, ignore the last batch of data if it is smaller than the batch size
        """
        self.model = model
        if not hasattr(ds, 'predictors'):
            raise ValueError("dataset must have 'predictors' variable")
        assert int(rank) > 0
        assert int(input_time_steps) > 0
        assert int(output_time_steps) > 0
        assert int(batch_size) > 0
        assert int(interval) > 0
        if sequence is not None:
            assert int(sequence) > 0
        if not(not load):
            if load not in ['full', 'required', 'minimal']:
                if isinstance(load, bool):
                    load = 'required'
                else:
                    raise ValueError("'load' must be one of 'full', 'required', or 'minimal'")
        try:
            add_insolation = to_bool(add_insolation)
        except ValueError:
            pass
        assert isinstance(add_insolation, (bool, str))
        if isinstance(add_insolation, str):
            assert add_insolation in ['hourly', 'daily']
        self._add_insolation = 1 if isinstance(add_insolation, str) else int(add_insolation)
        self._daily_insolation = str(add_insolation) == 'daily'
        self._load = load
        self._is_loaded = False

        self.ds = ds
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._is_convolutional = self.model.is_convolutional
        self._keep_time_axis = self.model.is_recurrent
        self._impute_missing = self.model.impute
        self._indices = []
        self._sequence = sequence
        if self._sequence is not None:
            self._n_sample = ds.dims['sample'] - interval * (input_time_steps + output_time_steps * sequence) + 1
        else:
            self._n_sample = ds.dims['sample'] - interval * (input_time_steps + output_time_steps) + 1
        if 'time_step' in ds.dims:
            # Use -1 index because Preprocessor.data_to_samples (which generates a 'time_step' dim), assigns the
            # datetime 'sample' dim based on the initialization time, time_step=-1
            self.da = self.ds.predictors.isel(time_step=-1)
        else:
            self.da = self.ds.predictors

        self.rank = rank
        self._input_sel = input_sel or {}
        if len(self._input_sel) == 0:
            if 'varlev' in self.ds.variables.keys():
                self._input_sel = {'varlev': self.ds['varlev'].values}
            else:
                self._input_sel = {'variable': self.ds['variable'].values, 'level': self.ds['level'].values}
        self._output_sel = output_sel or {}
        if len(self._output_sel) == 0:
            if 'varlev' in self.ds.variables.keys():
                self._output_sel = {'varlev': self.ds['varlev'].values}
            else:
                self._output_sel = {'variable': self.ds['variable'].values, 'level': self.ds['level'].values}
        self._input_time_steps = input_time_steps
        self._output_time_steps = output_time_steps
        self._interval = interval
        self.drop_remainder = to_bool(drop_remainder)

        # Temporarily set DataArrays for coordinates, overwritten when data are loaded
        self.input_da = self.da.isel(sample=[0]).sel(**self._input_sel)
        self.output_da = self.da.isel(sample=[0]).sel(**self._output_sel)
        if not delay_load:
            self._load_data()

        self.on_epoch_end()

        # Pre-generate the insolation data
        if self._add_insolation:
            sol = insolation(self.da.sample.values, self.ds.lat.values, self.ds.lon.values,
                             daily=self._daily_insolation)
            self.insolation_da = xr.DataArray(sol, dims=['sample'] + ['x%d' % r for r in range(self.rank)])
            self.insolation_da['sample'] = self.da.sample.values

        # Add extra constants
        self.constants = constants
        if self.constants is not None:
            try:
                assert self.constants.shape[-self.rank:] == self.shape[-self.rank:]
            except AssertionError:
                raise ValueError('spatial dimensions of constants must be the same as input data; got %s and %s' %
                                 (self.constants.shape[-self.rank:], self.shape[-self.rank:]))

        # Transpose option
        self.channels_last = to_bool(channels_last)
        self._time_transpose = (0, 1,) + tuple(range(3, 3 + self.rank)) + (2,)
        if self._keep_time_axis:
            self._transpose = self._time_transpose
        else:
            self._transpose = (0,) + tuple(range(2, 2 + self.rank)) + (1,)

    def _load_data(self):
        if not(not self._load):
            print('SeriesDataGenerator: loading data to memory')
        if self._load == 'full':
            self.ds.load()
        if self._load == 'minimal':
            # Try to transpose the axes so we can use basic indexing to return views
            if 'varlev' in self._input_sel.keys():
                union = [s for s in self._input_sel['varlev'] if s in self._output_sel['varlev']]
                added_in = [s for s in self._input_sel['varlev'] if s not in union]
                added_out = [s for s in self._output_sel['varlev'] if s not in union]
                if len(added_in) > 0 and len(added_out) > 0:
                    warnings.warn("Found extra variables in both input and output, could not reduce to basic "
                                  "indexing. 'minimal' indexing will use much more memory than 'required'.")
                    self.da.load()
                    self.input_da = self.da.sel(**self._input_sel)
                    self.output_da = self.da.sel(**self._output_sel)
                else:
                    self.da = self.da.sel(varlev=union + added_in + added_out)
                    self.da.load()
                    self.input_da = self.da.isel(varlev=slice(0, len(union) + len(added_in)))
                    self.output_da = self.da.isel(varlev=slice(0, len(union) + len(added_out)))
            else:
                raise NotImplementedError("Check for 'minimal' data loading not implemented yet for input files with "
                                          "variable/level axes. Use 'required' to avoid excessive memory use.")
        else:
            self.input_da = self.da.sel(**self._input_sel)
            self.output_da = self.da.sel(**self._output_sel)
            if self._load == 'required':
                self.input_da.load()
                self.output_da.load()
        self._is_loaded = True

    @property
    def shape(self):
        """
        :return: the original shape of input data: (time_step, [variable, level,] lat, lon); excludes insolation
        """
        return (self._input_time_steps,) + self.input_da.shape[1:]

    @property
    def n_features(self):
        """
        :return: int: the number of input features; includes insolation
        """
        return int(np.prod(self.shape)) + int(np.prod(self.shape[-self.rank:])) \
            * self._input_time_steps * self._add_insolation

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened input features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x). Includes insolation.
        """
        if self._keep_time_axis:
            result = (self._input_time_steps,) + (int(np.prod(self.shape[1:-self.rank])) + self._add_insolation,)\
                + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s - 1] for s in self._time_transpose[1:]])
            else:
                return result
        else:
            result = (int(np.prod(self.shape[:-self.rank])) +
                      self._input_time_steps * self._add_insolation,) + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._transpose[1:]])
            else:
                return result

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x); includes insolation
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    @property
    def output_shape(self):
        """
        :return: the original shape of outputs: (time_step, [variable, level,] lat, lon)
        """
        return (self._output_time_steps,) + self.output_da.shape[1:]

    @property
    def output_n_features(self):
        """
        :return: int: the number of output features
        """
        return int(np.prod(self.output_shape))

    @property
    def output_dense_shape(self):
        """
        :return: the shape of flattened output features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.output_shape[0],) + (self.output_n_features // self.output_shape[0],)
        else:
            return (self.output_n_features,) + ()

    @property
    def output_convolution_shape(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D or ConvLSTM2D layer. If the model is
            recurrent, (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            result = (self._output_time_steps,) + (int(np.prod(self.output_shape[1:-self.rank])),) \
                + self.output_shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._time_transpose[1:]])
            else:
                return result
        else:
            result = (int(np.prod(self.output_shape[:-self.rank])),) + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._transpose[1:]])
            else:
                return result

    @property
    def output_shape_2d(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.output_convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.output_convolution_shape

    @property
    def insolation_shape(self):
        """
        :return: the shape of insolation inputs in steps 1- of an input sequence, or None if add_insolation is False.
            Note that it always includes the time step dimension. The network needs to accommodate this.
        """
        if self.channels_last:
            return tuple((self._input_time_steps,) + self.convolution_shape[:self.rank]) + (1,)
        else:
            return tuple((self._input_time_steps, 1) + self.convolution_shape[-self.rank:])

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples, scale_and_impute=True):
        if len(samples) == 0:
            samples = np.arange(self._n_sample, dtype=np.int)
        else:
            samples = np.array(samples, dtype=np.int)
        n_sample = len(samples)

        if not self._is_loaded:
            self._load_data()

        # Predictors
        p = np.concatenate([self.input_da.values[samples + n * self._interval, np.newaxis]
                            for n in range(self._input_time_steps)], axis=1)
        if self._add_insolation:
            insol = []
            if self._sequence is not None:
                for s in range(self._sequence):
                    insol.append(
                        np.concatenate(
                            [self.insolation_da.values[samples + self._interval * (self._input_time_steps * s + n),
                                                       np.newaxis, np.newaxis] for n in range(self._input_time_steps)],
                            axis=1
                        )
                    )
            else:
                insol.append(
                    np.concatenate([self.insolation_da.values[samples + n * self._interval, np.newaxis, np.newaxis]
                                    for n in range(self._input_time_steps)], axis=1)
                )
            p = np.concatenate([p, insol[0]], axis=2)
        p = p.reshape((n_sample, -1))

        # Targets, including sequence if desired
        if self._sequence is not None:
            targets = []
            for s in range(self._sequence):
                t = np.concatenate(
                    [self.output_da.values[samples + self._interval * (
                            self._input_time_steps + self._output_time_steps * s + n), np.newaxis]
                     for n in range(self._output_time_steps)],
                    axis=1
                )

                t = t.reshape((n_sample, -1))

                # Remove samples with NaN; scale and impute
                if self._remove_nan:
                    p, t = delete_nan_samples(p, t)
                if scale_and_impute:
                    if self._impute_missing:
                        p, t = self.model.imputer_transform(p, t)
                    p, t = self.model.scaler_transform(p, t)

                # Format spatial shape for convolutions; also takes care of time axis
                if self._is_convolutional:
                    cl = bool(self.channels_last)
                    self.channels_last = False
                    p = p.reshape((n_sample,) + self.convolution_shape)
                    t = t.reshape((n_sample,) + self.output_convolution_shape)
                    self.channels_last = bool(cl)
                elif self._keep_time_axis:
                    p = p.reshape((n_sample,) + self.dense_shape)
                    t = t.reshape((n_sample,) + self.output_dense_shape)

                targets.append(t)

            # Sequence of inputs (plus insolation) for predictors
            if self._add_insolation:
                p = [p] + insol[1:]
        else:
            t = np.concatenate([self.output_da.values[samples + self._interval * (self._input_time_steps + n),
                                                      np.newaxis]
                                for n in range(self._output_time_steps)], axis=1)

            t = t.reshape((n_sample, -1))

            # Remove samples with NaN; scale and impute
            if self._remove_nan:
                p, t = delete_nan_samples(p, t)
            if scale_and_impute:
                if self._impute_missing:
                    p, t = self.model.imputer_transform(p, t)
                p, t = self.model.scaler_transform(p, t)

            # Format spatial shape for convolutions; also takes care of time axis
            if self._is_convolutional:
                cl = bool(self.channels_last)
                self.channels_last = False
                p = p.reshape((n_sample,) + self.convolution_shape)
                t = t.reshape((n_sample,) + self.output_convolution_shape)
                self.channels_last = bool(cl)
            elif self._keep_time_axis:
                p = p.reshape((n_sample,) + self.dense_shape)
                t = t.reshape((n_sample,) + self.output_dense_shape)

            targets = t

        # Add constants
        if self.constants is not None:
            constants = np.repeat(np.expand_dims(self.constants, axis=0), n_sample, axis=0)
            if self._keep_time_axis:
                constants = np.expand_dims(constants, 1)
            if isinstance(p, list):
                p = p + [constants]
            else:
                p = [p, constants]

        # Transpose to channels_last if requested
        if self.channels_last:
            if isinstance(p, list):
                for s in range(len(p)):
                    try:
                        p[s] = p[s].transpose(self._transpose)
                    except ValueError:  # solar inputs retain time dimension
                        p[s] = p[s].transpose(self._time_transpose)
            else:
                p = p.transpose(self._transpose)
            if isinstance(targets, list):
                for s in range(len(targets)):
                    targets[s] = targets[s].transpose(self._transpose)
            else:
                targets = targets.transpose(self._transpose)

        return p, targets

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        if self.drop_remainder:
            return int(np.floor(self._n_sample / self._batch_size))
        else:
            return int(np.ceil(self._n_sample / self._batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return: (ndarray, ndarray): predictors, targets
        """
        # Generate indexes of the batch
        if int(index) < 0:
            index = len(self) + index
        if index > len(self):
            raise IndexError
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y


class ArrayDataGenerator(Sequence):
    """
    Based on the SeriesDataGenerator class, this generator is designed to take a single array of input data loaded
    externally (see the prepare_data_array method in .preprocessing) and manipulate views to produce each batch of
    samples. For simplicity, this class does not store the model information and does not allow for model-based
    scaling or imputing of data. It is also possible to use a netCDF4 variable as the input array for disk-based IO.
    """

    def __init__(self, model, array, rank=2, batch_size=32, input_slice=None, output_slice=None,
                 input_time_steps=1, output_time_steps=1, sequence=None, interval=1,
                 shuffle=False, remove_nan=True, insolation_array=None, constants=None, channels_last=False,
                 drop_remainder=False):
        """
        Initialize an ArrayDataGenerator.

        :param model: instance of a DLWP model, just used for some metadata
        :param array: np.array or netCDF4.variable: array of predictor data; order (time, variable, ...)
        :param rank: int: the number of spatial dimensions (e.g. 2 for 2-d data and convolutions)
        :param batch_size: int: number of samples to take at a time from the dataset
        :param input_slice: slice or array-like: variable/level selection for input features
        :param output_slice: slice or array-like: variable/level selection for output features
        :param input_time_steps: int: number of time steps in the input features
        :param output_time_steps: int: number of time steps in the output features (recommended either 1 or the same
            as input_time_steps)
        :param sequence: int or None: if int, then the output targets is a list of sequence consecutive forecast steps.
            Note that in this mode, if add_insolation is True, the inputs are also a list of consecutive forecast steps,
            with the first step containing all of the input data and subsequent steps containing only the requisite
            insolation fields.
        :param interval: int: the number of steps to take between data samples and within input/output time steps.
            Effectively it is the model delta t multiplier for the data resolution.
        :param shuffle: bool: if True, randomly select batches
        :param remove_nan: bool: if True, remove any samples with NaNs
        :param insolation_array: np.array: insolation (see DLWP.util.insolation) for the given data
        :param constants: ndarray: additional constant fields to add to each input. Must match the spatial dimensions
            (last `rank` dimensions) of the input data.
        :param channels_last: bool: if True, returns data with channels as the last dimension. May slow down processing
            of data, but may speed up GPU operations on the data.
        :param drop_remainder: bool: if True, ignore the last batch of data if it is smaller than the batch size
        """
        assert int(rank) > 0
        assert int(input_time_steps) > 0
        assert int(output_time_steps) > 0
        assert int(batch_size) > 0
        assert int(interval) > 0
        if sequence is not None:
            assert int(sequence) > 0

        self.array = array
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._remove_nan = remove_nan
        self._is_convolutional = model.is_convolutional
        self._keep_time_axis = model.is_recurrent
        self._impute_missing = model.impute
        self._indices = []
        self._sequence = sequence
        if self._sequence is not None:
            self._n_sample = array.shape[0] - interval * (input_time_steps + output_time_steps * sequence) + 1
        else:
            self._n_sample = array.shape[0] - interval * (input_time_steps + output_time_steps) + 1

        self.rank = rank
        self._input_slice = input_slice or slice(None)
        self._output_slice = output_slice or slice(None)
        self._input_time_steps = input_time_steps
        self._output_time_steps = output_time_steps
        self._interval = interval
        self.drop_remainder = to_bool(drop_remainder)

        if isinstance(self._input_slice, slice):
            self._input_size = len(range(*self._input_slice.indices(array.shape[1])))
        else:
            self._input_size = len(self._input_slice)
        if isinstance(self._output_slice, slice):
            self._output_size = len(range(*self._output_slice.indices(array.shape[1])))
        else:
            self._output_size = len(self._output_slice)

        self.on_epoch_end()

        # Add insolation
        self.insolation_array = insolation_array
        self._add_insolation = 1 if self.insolation_array is not None else 0
        assert self.insolation_array.shape[-self.rank:] == self.shape[-self.rank:], \
            "spatial dimensions of insolation must be the same as input data; got %s and %s" % \
            (self.insolation_array.shape[-self.rank:], self.shape[-self.rank:])

        # Add extra constants
        self.constants = constants
        if self.constants is not None:
            assert self.constants.shape[-self.rank:] == self.shape[-self.rank:], \
                "spatial dimensions of constants must be the same as input data; got %s and %s" % \
                (self.constants.shape[-self.rank:], self.shape[-self.rank:])

        # Transpose option
        self.channels_last = to_bool(channels_last)
        self._time_transpose = (0, 1,) + tuple(range(3, 3 + self.rank)) + (2,)
        if self._keep_time_axis:
            self._transpose = self._time_transpose
        else:
            self._transpose = (0,) + tuple(range(2, 2 + self.rank)) + (1,)

    @property
    def shape(self):
        """
        :return: the original shape of input data: (time_step, varlev, lat, lon); excludes insolation
        """
        return (self._input_time_steps, self._input_size) + self.array.shape[2:]

    @property
    def n_features(self):
        """
        :return: int: the number of input features; includes insolation
        """
        return int(np.prod(self.shape)) + int(np.prod(self.shape[-self.rank:])) \
            * self._input_time_steps * self._add_insolation

    @property
    def dense_shape(self):
        """
        :return: the shape of flattened input features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.shape[0],) + (self.n_features // self.shape[0],)
        else:
            return (self.n_features,) + ()

    @property
    def convolution_shape(self):
        """
        :return: the shape of the predictors expected by a Conv2D or ConvLSTM2D layer. If the model is recurrent,
            (time_step, channels, y, x); if not, (channels, y, x). Includes insolation.
        """
        if self._keep_time_axis:
            result = (self._input_time_steps,) + (int(np.prod(self.shape[1:-self.rank])) + self._add_insolation,)\
                + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s - 1] for s in self._time_transpose[1:]])
            else:
                return result
        else:
            result = (int(np.prod(self.shape[:-self.rank])) +
                      self._input_time_steps * self._add_insolation,) + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._transpose[1:]])
            else:
                return result

    @property
    def shape_2d(self):
        """
        :return: the shape of the predictors expected by a Conv2D layer, (channels, y, x); includes insolation
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.convolution_shape

    @property
    def output_shape(self):
        """
        :return: the original shape of outputs: (time_step, [variable, level,] lat, lon)
        """
        return (self._output_time_steps, self._output_size) + self.array.shape[2:]

    @property
    def output_n_features(self):
        """
        :return: int: the number of output features
        """
        return int(np.prod(self.output_shape))

    @property
    def output_dense_shape(self):
        """
        :return: the shape of flattened output features. If the model is recurrent, (time_step, features); otherwise,
            (features,).
        """
        if self._keep_time_axis:
            return (self.output_shape[0],) + (self.output_n_features // self.output_shape[0],)
        else:
            return (self.output_n_features,) + ()

    @property
    def output_convolution_shape(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D or ConvLSTM2D layer. If the model is
            recurrent, (time_step, channels, y, x); if not, (channels, y, x).
        """
        if self._keep_time_axis:
            result = (self._output_time_steps,) + (int(np.prod(self.output_shape[1:-self.rank])),) \
                + self.output_shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._time_transpose[1:]])
            else:
                return result
        else:
            result = (int(np.prod(self.output_shape[:-self.rank])),) + self.shape[-self.rank:]
            if self.channels_last:
                return tuple([result[s-1] for s in self._transpose[1:]])
            else:
                return result

    @property
    def output_shape_2d(self):
        """
        :return: the shape of the predictors expected to be returned by a Conv2D layer, (channels, y, x)
        """
        if self._keep_time_axis:
            self._keep_time_axis = False
            s = tuple(self.output_convolution_shape)
            self._keep_time_axis = True
            return s
        else:
            return self.output_convolution_shape

    @property
    def insolation_shape(self):
        """
        :return: the shape of insolation inputs in steps 1- of an input sequence, or None if add_insolation is False.
            Note that it always includes the time step dimension. The network needs to accommodate this.
        """
        if self.channels_last:
            return tuple((self._input_time_steps,) + self.convolution_shape[:self.rank]) + (1,)
        else:
            return tuple((self._input_time_steps, 1) + self.convolution_shape[-self.rank:])

    def on_epoch_end(self):
        self._indices = np.arange(self._n_sample)
        if self._shuffle:
            np.random.shuffle(self._indices)

    def generate(self, samples):
        if len(samples) == 0:
            samples = np.arange(self._n_sample, dtype=np.int)
        else:
            samples = np.array(samples, dtype=np.int)
        n_sample = len(samples)

        # Predictors
        p = np.concatenate([self.array[samples + n * self._interval, self._input_slice][:, np.newaxis]
                            for n in range(self._input_time_steps)], axis=1)
        if self._add_insolation:
            insol = []
            if self._sequence is not None:
                for s in range(self._sequence):
                    insol.append(
                        np.concatenate(
                            [self.insolation_array[samples + self._interval * (self._input_time_steps * s + n),
                                                   np.newaxis, np.newaxis] for n in range(self._input_time_steps)],
                            axis=1
                        )
                    )
            else:
                insol.append(
                    np.concatenate([self.insolation_array[samples + n * self._interval, np.newaxis, np.newaxis]
                                    for n in range(self._input_time_steps)], axis=1)
                )
            p = np.concatenate([p, insol[0]], axis=2)
        p = p.reshape((n_sample, -1))

        # Targets, including sequence if desired
        if self._sequence is not None:
            targets = []
            for s in range(self._sequence):
                t = np.concatenate(
                    [self.array[samples + self._interval * (self._input_time_steps + self._output_time_steps * s + n),
                                self._output_slice][:, np.newaxis]
                     for n in range(self._output_time_steps)],
                    axis=1
                )

                t = t.reshape((n_sample, -1))

                # Remove samples with NaN if requested
                if self._remove_nan:
                    p, t = delete_nan_samples(p, t)

                # Format spatial shape for convolutions; also takes care of time axis
                if self._is_convolutional:
                    cl = bool(self.channels_last)
                    self.channels_last = False
                    p = p.reshape((n_sample,) + self.convolution_shape)
                    t = t.reshape((n_sample,) + self.output_convolution_shape)
                    self.channels_last = bool(cl)
                elif self._keep_time_axis:
                    p = p.reshape((n_sample,) + self.dense_shape)
                    t = t.reshape((n_sample,) + self.output_dense_shape)

                targets.append(t)

            # Sequence of inputs (plus insolation) for predictors
            if self._add_insolation:
                p = [p] + insol[1:]
        else:
            t = np.concatenate([self.array[samples + self._interval * (self._input_time_steps + n),
                                           np.newaxis, self._output_slice]
                                for n in range(self._output_time_steps)], axis=1)

            t = t.reshape((n_sample, -1))

            # Remove samples with NaN if requested
            if self._remove_nan:
                p, t = delete_nan_samples(p, t)

            # Format spatial shape for convolutions; also takes care of time axis
            if self._is_convolutional:
                cl = bool(self.channels_last)
                self.channels_last = False
                p = p.reshape((n_sample,) + self.convolution_shape)
                t = t.reshape((n_sample,) + self.output_convolution_shape)
                self.channels_last = bool(cl)
            elif self._keep_time_axis:
                p = p.reshape((n_sample,) + self.dense_shape)
                t = t.reshape((n_sample,) + self.output_dense_shape)

            targets = t

        # Add constants
        if self.constants is not None:
            constants = np.repeat(np.expand_dims(self.constants, axis=0), n_sample, axis=0)
            if self._keep_time_axis:
                constants = np.expand_dims(constants, 1)
            if isinstance(p, list):
                p = p + [constants]
            else:
                p = [p, constants]

        # Transpose to channels_last if requested
        if self.channels_last:
            if isinstance(p, list):
                for s in range(len(p)):
                    try:
                        p[s] = p[s].transpose(self._transpose)
                    except ValueError:  # solar inputs retain time dimension
                        p[s] = p[s].transpose(self._time_transpose)
            else:
                p = p.transpose(self._transpose)
            if isinstance(targets, list):
                for s in range(len(targets)):
                    targets[s] = targets[s].transpose(self._transpose)
            else:
                targets = targets.transpose(self._transpose)

        return p, targets

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        if self.drop_remainder:
            return int(np.floor(self._n_sample / self._batch_size))
        else:
            return int(np.ceil(self._n_sample / self._batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return: (ndarray, ndarray): predictors, targets
        """
        # Generate indexes of the batch
        if int(index) < 0:
            index = len(self) + index
        if index > len(self):
            raise IndexError
        indexes = self._indices[index * self._batch_size:(index + 1) * self._batch_size]

        # Generate data
        X, y = self.generate(indexes)

        return X, y


def tf_data_generator(generator, batch_size=None, input_names=None, output_names=None):
    """
    Wraps a DLWP.model Generator class into a generator function that can be used in a TensorFlow.Data.Dataset object.

    :param generator: instance of a DLWP.model.generators class
    :param batch_size: int or None: if int, use a fixed batch size. Will cause an error if the last batch of training
        data does not have the same number of samples.
    :param input_names: list of str: optional list of names for the inputs, to match the model Input layers
    :param output_names: list of str: optional list of names for the outputs, to match the model's output layers
    :return: tensorflow.data.Dataset
    """
    # Determine structure of output data
    p, t = generator.generate([0])
    p_is_list = isinstance(p, list)
    t_is_list = isinstance(t, list)
    if p_is_list:
        if input_names is None:
            input_names = ['input_%d' % (i + 1) for i in range(len(p))]
        if len(input_names) != len(p):
            raise ValueError("mismatched length of input names relative to generated data; got %d but expected %d" %
                             (len(input_names), len(p)))
    if t_is_list:
        if output_names is None:
            output_names = ['output'] + ['output_%d' % i for i in range(1, len(t))]
        if len(output_names) != len(t):
            raise ValueError("mismatched length of input names relative to generated data; got %d but expected %d" %
                             (len(output_names), len(t)))

    # Go through I/O options: define the yielding function and the data types & shape
    if not p_is_list and not t_is_list:
        def yield_fn():
            for sample in generator:
                yield sample[0], sample[1]
        data_types = (tf.float32, tf.float32)
        data_shapes = (p.shape, t.shape)
    elif p_is_list and not t_is_list:
        def yield_fn():
            for sample in generator:
                yield {input_names[i]: d for i, d in enumerate(sample[0])}, sample[1]
        data_types = ({input_names[i]: tf.float32 for i in range(len(p))}, tf.float32)
        data_shapes = ({input_names[i]: (batch_size,) + p[i].shape[1:] for i in range(len(p))}, t.shape)
    elif not p_is_list and t_is_list:
        def yield_fn():
            for sample in generator:
                yield sample[0], {output_names[i]: d for i, d in enumerate(sample[1])}
        data_types = (tf.float32, {output_names[i]: tf.float32 for i in range(len(t))})
        data_shapes = (p.shape, {output_names[i]: (batch_size,) + t[i].shape[1:] for i in range(len(t))})
    else:
        def yield_fn():
            for sample in generator:
                yield {input_names[i]: d for i, d in enumerate(sample[0])}, \
                      {output_names[i]: d for i, d in enumerate(sample[1])}
        data_types = ({input_names[i]: tf.float32 for i in range(len(p))},
                      {output_names[i]: tf.float32 for i in range(len(t))})
        data_shapes = ({input_names[i]: (batch_size,) + p[i].shape[1:] for i in range(len(p))},
                       {output_names[i]: (batch_size,) + t[i].shape[1:] for i in range(len(t))})

    # Create a tf.data.Dataset
    del p, t
    tf_dataset = tf.data.Dataset.from_generator(yield_fn, output_types=data_types, output_shapes=data_shapes)
    return tf_dataset
