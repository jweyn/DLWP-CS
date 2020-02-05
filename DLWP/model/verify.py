#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for validating DLWP forecasts.
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import warnings


def forecast_error(forecast, valid, method='mse', axis=None, weighted=False, climatology=None):
    """
    Calculate the error of a time series model forecast.

    :param forecast: ndarray or DataArray: forecast from a DLWP model (forecast hour is first axis)
    :param valid: ndarray or DataArray: validation target data for the predictors the forecast was made on
    :param method: str: method for computing the error. Options are:
        'mse': mean squared error
        'mae': mean absolute error
        'rmse': root-mean-squared error
        'acc': anomaly correlation coefficient
        'cos': cosine similarity score
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension. Note that for cosine similarity it is recommended to explicitly
        specify the spatial axes.
    :param weighted: bool: if True, expects inputs to be DataArrays with 'lat' as one of the dimensions, and weights
        according to the latitude
    :param climatology: ndarray or DataArray: mean climatology state for computing the ACC score. Dimensions other than
        axis 0 (forecast hour) and axis 1 (time) must match that of the forecast/valid arrays. If either of the first
        two axes are included, they must be size 1.
    :return: ndarray: forecast error with forecast hour as the first dimension
    """
    assert method in ['mse', 'mae', 'rmse', 'acc', 'cos'], "'method' must be one of 'mse', 'mae', 'rmse', 'acc', 'cos'"
    if method in ['acc', 'cos'] and climatology is None:
        warnings.warn("'acc' and 'cos' error methods expect to get a climatology; using 0 instead, which may yield "
                      "unexpected results.")
        climatology = 0.
    n_f = forecast.shape[0]
    if weighted:
        weights = np.cos(np.deg2rad(valid.lat))
        weights /= weights.mean()
    else:
        weights = 1.
    if len(forecast.shape) == len(valid.shape):
        # valid provided with a forecast hour dimension 0
        if axis is None:
            axis = tuple(range(1, len(valid.shape)))
        if method == 'mse':
            return np.nanmean((valid - forecast) ** 2. * weights, axis=axis)
        elif method == 'mae':
            return np.nanmean(np.abs((valid - forecast) * weights), axis=axis)
        elif method == 'rmse':
            return np.sqrt(np.nanmean((valid - forecast) ** 2. * weights, axis=axis))
        elif method == 'acc':
            return (np.nanmean((valid - climatology) * (forecast - climatology) * weights, axis=axis)
                    / np.sqrt(np.nanmean((valid - climatology) ** 2. * weights, axis=axis) *
                              np.nanmean((forecast - climatology) ** 2. * weights, axis=axis)))
        elif method == 'cos':
            return ((forecast - climatology).dot(valid * weights, dims=axis) /
                    (np.linalg.norm((forecast - climatology) * weights, axis=axis) *
                     np.linalg.norm((valid - climatology) * weights, axis=axis)))
    else:
        # valid provided as a continuous time series without a forecast hour dimension
        n_val = valid.shape[0]
        me = []
        for f in range(n_f):
            if method == 'mse':
                me.append(np.nanmean((valid[f:] - forecast[f, :(n_val - f)]) ** 2. * weights, axis=axis))
            elif method == 'mae':
                me.append(np.nanmean(np.abs((valid[f:] - forecast[f, :(n_val - f)]) * weights), axis=axis))
            elif method == 'rmse':
                me.append(np.sqrt(np.nanmean((valid[f:] - forecast[f, :(n_val - f)]) ** 2. * weights, axis=axis)))
            elif method == 'acc':
                return (np.nanmean((valid[f:] - climatology) * (forecast[f, :(n_val - f)] - climatology), axis=axis)
                        / np.sqrt(np.nanmean((valid[f:] - climatology) ** 2., axis=axis) *
                                  np.nanmean((forecast[f, :(n_val - f)] - climatology) ** 2., axis=axis)))
            elif method == 'cos':
                return (forecast[f, :(n_val - f)] - climatology).dot(valid[f:] - climatology, dims=axis) / \
                       (np.linalg.norm(forecast[f, :(n_val - f)] - climatology, axis=axis) *
                        np.linalg.norm(valid[f:] - climatology, axis=axis))
        return np.array(me)


def persistence_error(predictors, valid, n_fhour, method='mse', axis=None, weighted=False):
    """
    Calculate the error of a persistence forecast out to n_fhour forecast hours.

    DEPRECATED as of version 0.8.4. Use forecast_error instead and create an appropriate array of persistence forecasts.

    :param predictors: ndarray or DataArray: predictor data
    :param valid: ndarray or DataArray: validation target data
    :param n_fhour: int: number of steps to take forecast out to
    :param method: str: 'mse' for mean squared error, 'mae' for mean absolute error, 'rmse' for root-mean-square
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension.
    :param weighted: bool: if True, expects inputs to be DataArrays with 'lat' as one of the dimensions, and weights
        according to the latitude
    :return: ndarray: persistence error with forecast hour as the first dimension
    """
    warnings.warn("'persistence_error' is deprecated as of version 0.8.4. Use 'forecast_error' with an "
                  "appropriate array of persistence forecasts instead.", DeprecationWarning)
    if method not in ['mse', 'mae', 'rmse']:
        raise ValueError("'method' must be 'mse', 'rmse', or 'mae'")
    n_f = valid.shape[0]
    me = []
    if weighted:
        weights = np.cos(np.deg2rad(valid.lat))
        weights /= weights.mean()
    else:
        weights = 1.
    for f in range(n_fhour):
        if method == 'mse':
            me.append(np.nanmean((valid[f:] - predictors[:(n_f - f)]) ** 2. * weights, axis=axis))
        elif method == 'mae':
            me.append(np.nanmean(np.abs((valid[f:] - predictors[:(n_f - f)]) * weights), axis=axis))
        elif method == 'rmse':
            me.append(np.sqrt(np.nanmean((valid[f:] - predictors[:(n_f - f)]) ** 2. * weights, axis=axis)))
    return np.array(me)


def climo_error(valid, n_fhour, method='mse', axis=None, weighted=False):
    """
    Calculate the error of a climatology forecast out to n_fhour forecast hours.

    :param valid: ndarray or DataArray: validation target data
    :param n_fhour: int: number of steps to take forecast out to
    :param method: str: 'mse' for mean squared error, 'mae' for mean absolute error, 'rmse' for root-mean-square
    :param axis: int, tuple, or None: take the mean of the error along this axis. Regardless of this setting, the
        forecast hour will be the first dimension.
    :param weighted: bool: if True, expects inputs to be DataArrays with 'lat' as one of the dimensions, and weights
        according to the latitude
    :return: ndarray: persistence error with forecast hour as the first dimension
    """
    if method not in ['mse', 'mae', 'rmse']:
        raise ValueError("'method' must be 'mse', 'rmse', or 'mae'")
    n_f = valid.shape[0]
    me = []
    if weighted:
        weights = np.cos(np.deg2rad(valid.lat))
        weights /= weights.mean()
    else:
        weights = 1.
    for f in range(n_fhour):
        if method == 'mse':
            me.append(np.nanmean((valid[:(n_f - f)] - np.nanmean(valid, axis=0)) ** 2. * weights, axis=axis))
        elif method == 'mae':
            me.append(np.nanmean(np.abs((valid[:(n_f - f)] - np.nanmean(valid, axis=0)) * weights), axis=axis))
        elif method == 'rmse':
            me.append(np.sqrt(np.nanmean((valid[:(n_f - f)] - np.nanmean(valid, axis=0)) ** 2. * weights, axis=axis)))
    return np.array(me)


def monthly_climo_error(da, val_set, n_fhour=None, method='mse', climo_da=None, by_day_of_year=False, return_da=False,
                        weighted=False):
    """
    Calculates a month-aware climatology error for a validation set from a DataArray of the atmospheric state.

    :param da: xarray DataArray: contains a 'time' or 'sample' dimension
    :param val_set: list: list of times for which to calculate an error
    :param n_fhour: int or None: if int, multiplies the resulting error into a list of length n_fhour
    :param method: str: method for computing the error. Options are:
        'mse': mean squared error
        'mae': mean absolute error
        'rmse': root-mean-squared error
        'acc': anomaly correlation coefficient – returns zeros
        'cos': cosine similarity score – returns zeros
    :param climo_da: xarray DataArray: if provided, contains a pre-computed monthly or daily climatology
    :param by_day_of_year: bool: of True, computes climatology by day of year instead of monthly
    :param return_da: bool: if True, also returns a DataArray of the error from climatology
    :param weighted: bool: if True, expects inputs to be DataArrays with 'lat' as one of the dimensions, and weights
        according to the latitude
    :return: (int or list[, DataArray])
    """
    assert method in ['mse', 'mae', 'rmse', 'acc', 'cos'], "'method' must be one of 'mse', 'mae', 'rmse', 'acc', 'cos'"
    time_dim = 'sample' if 'sample' in da.dims else 'time'
    parameter = 'dayofyear' if by_day_of_year else 'month'
    if climo_da is None:
        climo_da = da.groupby('%s.%s' % (time_dim, parameter)).mean(time_dim)
    anomaly = da.sel(**{time_dim: val_set}).groupby('%s.%s' % (time_dim, parameter)) - climo_da
    if weighted:
        weights = np.cos(np.deg2rad(da.lat))
        weights /= weights.mean()
    else:
        weights = 1.
    if method == 'mse':
        me = float((anomaly ** 2. * weights).mean().values)
    elif method == 'mae':
        me = float((anomaly.abs() * weights).mean().values)
    elif method == 'rmse':
        me = np.sqrt(float((anomaly ** 2. * weights).mean().values))
    elif method == 'acc':
        me = 0.
    elif method == 'cos':
        me = 0.
    if n_fhour is not None:
        me = np.array([me] * n_fhour)
    if return_da:
        return me, anomaly
    else:
        return me


def predictors_to_time_series(predictors, time_steps, has_time_dim=True, use_first_step=False, meta_ds=None):
    """
    Reshapes predictors into a continuous time series that can be used for verification methods in this module and
    matches the reshaped output of DLWP models' 'predict_timeseries' method. This is only necessary if the data are for
    a model predicting multiple time steps. Also truncates the first (time_steps - 1) samples so that the time series
    matches the effective forecast initialization time, or the last (time_steps -1) samples if use_first_step == True.

    :param predictors: ndarray: array of predictor data
    :param time_steps: int: number of time steps in the predictor data
    :param has_time_dim: bool: if True, the time step dimension is axis=1 in the predictors, otherwise, axis 1 is
        assumed to be time_steps * num_channels_or_features
    :param use_first_step: bool: if True, keeps the first time step instead of the last (useful for validation)
    :param meta_ds: xarray Dataset: if not None, add metadata to the output using the coordinates in this Dataset
    :return: ndarray or xarray DataArray: reshaped predictors
    """
    idx = 0 if use_first_step else -1
    if has_time_dim:
        result = predictors[:, idx]
    else:
        sample_dim = predictors.shape[0]
        feature_shape = predictors.shape[1:]
        predictors = predictors.reshape((sample_dim, time_steps, -1) + feature_shape[1:])
        result = predictors[:, idx]
    if meta_ds is not None:
        if 'level' in meta_ds.dims:
            result = result.reshape((meta_ds.dims['sample'], meta_ds.dims['variable'], meta_ds.dims['level'],
                                     meta_ds.dims['lat'], meta_ds.dims['lon']))
            result = xr.DataArray(result,
                                  coords=[meta_ds.sample, meta_ds.variable, meta_ds.level, meta_ds.lat, meta_ds.lon],
                                  dims=['time', 'variable', 'level', 'lat', 'lon'])
        else:
            result = xr.DataArray(result, coords=[meta_ds.sample, meta_ds.varlev, meta_ds.lat, meta_ds.lon],
                                  dims=['time', 'varlev', 'lat', 'lon'])

    return result


def add_metadata_to_forecast(forecast, f_hour, meta_ds, f_hour_timedelta_type=True, channels_last=False):
    """
    Add metadata to a forecast based on the initialization times and coordinates in meta_ds.

    :param forecast: ndarray: (forecast_hour, time, variable, lat, lon)
    :param f_hour: iterable: forecast hour coordinate values
    :param meta_ds: xarray Dataset: contains metadata for time, variable, lat, and lon
    :param f_hour_timedelta_type: bool: if True, converts f_hour dimension into a timedelta type. May not always be
        compatible with netCDF applications.
    :param channels_last: bool: if True, assumes varlev or variable/level are last dimensions
    :return: xarray.DataArray: array with metadata
    """
    nf = len(f_hour)
    if f_hour_timedelta_type:
        f_hour = np.array(f_hour).astype('timedelta64[h]')
    if nf != forecast.shape[0]:
        raise ValueError("'f_hour' coordinate must have same size as the first axis of 'forecast'")
    if 'level' in meta_ds.dims:
        if channels_last:
            dims_order = ['sample', 'lat', 'lon', 'variable', 'level']
        else:
            dims_order = ['sample', 'variable', 'level', 'lat', 'lon']
        forecast = forecast.reshape([nf] + [meta_ds.dims[d] for d in dims_order])
    else:
        if channels_last:
            dims_order = ['sample', 'lat', 'lon', 'varlev']
        else:
            dims_order = ['sample', 'varlev', 'lat', 'lon']
    forecast = xr.DataArray(
        forecast,
        coords=[f_hour] + [meta_ds[d] for d in dims_order],
        dims=['f_hour'] + ['time' if d == 'sample' else d for d in dims_order],
        name='forecast'
    )
    return forecast


def add_metadata_to_forecast_cs(forecast, f_hour, meta_ds, f_hour_timedelta_type=False, channels_last=False):
    """
    Add metadata to a forecast based on the initialization times and coordinates in meta_ds, which is on a cubed sphere.

    :param forecast: ndarray: (forecast_hour, time, variable, height, width, face)
    :param f_hour: iterable: forecast hour coordinate values
    :param meta_ds: xarray Dataset: contains metadata for time, variable, height, width, and face
    :param f_hour_timedelta_type: bool: if True, converts f_hour dimension into a timedelta type. May not always be
        compatible with netCDF applications.
    :param channels_last: bool: if True, assumes varlev or variable/level are last dimensions
    :return: xarray.DataArray: array with metadata
    """
    nf = len(f_hour)
    if f_hour_timedelta_type:
        f_hour = np.array(f_hour).astype('timedelta64[h]')
    if nf != forecast.shape[0]:
        raise ValueError("'f_hour' coordinate must have same size as the first axis of 'forecast'")
    if 'level' in meta_ds.dims:
        if channels_last:
            dims_order = ['sample', 'face', 'height', 'width', 'variable', 'level']
        else:
            dims_order = ['sample', 'variable', 'level', 'face', 'height', 'width']
        forecast = forecast.reshape([nf] + [meta_ds.dims[d] for d in dims_order])
    else:
        if channels_last:
            dims_order = ['sample', 'face', 'height', 'width', 'varlev']
        else:
            dims_order = ['sample', 'varlev', 'face', 'height', 'width']
    forecast = xr.DataArray(
        forecast,
        coords=[f_hour] + [meta_ds[d] for d in dims_order],
        dims=['f_hour'] + ['time' if d == 'sample' else d for d in dims_order],
        name='forecast'
    )
    return forecast


def verification_from_samples(ds, all_ds=None, init_times=None, forecast_steps=1, dt=6, f_hour_timedelta_type=True):
    """
    Generate a DataArray of forecast verification from a validation DataSet built using Preprocessor.data_to_samples().

    :param ds: xarray.Dataset: dataset of verification data. Time is the first dimension.
    :param all_ds: xarray.Dataset: optional Dataset containing the same variables/levels/lat/lon as val_ds but
        including more time steps for more robust handling of data at times outside of the validation selection
    :param init_times: iterable of Timestamps: optional list of verification initialization times
    :param forecast_steps: int: number of forward forecast iterations
    :param dt: int: forecast time step in hours
    :param f_hour_timedelta_type: bool: if True, converts f_hour dimension into a timedelta type. May not always be
        compatible with netCDF applications.
    :return: xarray.DataArray: verification with forecast hour as the first dimension
    """
    forecast_steps = int(forecast_steps)
    if forecast_steps < 1:
        raise ValueError("'forecast_steps' must be an integer >= 1")
    dt = int(dt)
    if dt < 1:
        raise ValueError("'dt' must be an integer >= 1")
    if init_times is None:
        init_times = ds.sample.values
    dims = [d for d in ds.predictors.dims if d.lower() not in ['time_step', 'sample', 'time']]
    f_hour = np.arange(dt, dt * forecast_steps + 1, dt)
    if f_hour_timedelta_type:
        f_hour = np.array(f_hour).astype('timedelta64[h]')
    verification = xr.DataArray(
        np.full([forecast_steps, len(init_times)] + [ds.dims[d] for d in dims], np.nan, dtype=np.float32),
        coords=[f_hour, init_times] + [ds[d] for d in dims],
        dims=['f_hour', 'time'] + dims,
        name='verification'
    )
    if all_ds is not None:
        valid_da = all_ds.targets.isel(time_step=0)
    else:
        valid_da = ds.targets.isel(time_step=0)
    for d, date in enumerate(init_times):
        verification[:, d] = valid_da.reindex(
            sample=pd.date_range(date, date + np.timedelta64(timedelta(hours=dt * (forecast_steps - 1))),
                                 freq='%sH' % int(dt)),
            method=None
        ).values
    return verification


def verification_from_series(ds, all_ds=None, init_times=None, forecast_steps=1, dt=6, f_hour_timedelta_type=True):
    """
    Generate a DataArray of forecast verification from a validation DataSet built using Preprocessor.data_to_series().

    :param ds: xarray.Dataset: dataset of verification data. Time is the first dimension.
    :param all_ds: xarray.Dataset: optional Dataset containing the same variables/levels/lat/lon as val_ds but
        including more time steps for more robust handling of data at times outside of the validation selection
    :param init_times: iterable of Timestamps: optional list of verification initialization times
    :param forecast_steps: int: number of forward forecast iterations
    :param dt: int: forecast time step in hours
    :param f_hour_timedelta_type: bool: if True, converts f_hour dimension into a timedelta type. May not always be
        compatible with netCDF applications.
    :return: xarray.DataArray: verification with forecast hour as the first dimension
    """
    forecast_steps = int(forecast_steps)
    if forecast_steps < 1:
        raise ValueError("'forecast_steps' must be an integer >= 1")
    dt = int(dt)
    if dt < 1:
        raise ValueError("'dt' must be an integer >= 1")
    if init_times is None:
        init_times = ds.sample.values
    dims = [d for d in ds.predictors.dims if d.lower() not in ['time_step', 'sample', 'time']]
    f_hour = np.arange(dt, dt * forecast_steps + 1, dt)
    if f_hour_timedelta_type:
        f_hour = np.array(f_hour).astype('timedelta64[h]')
    verification = xr.DataArray(
        np.full([forecast_steps, len(init_times)] + [ds.dims[d] for d in dims], np.nan, dtype=np.float32),
        coords=[f_hour, init_times] + [ds[d] for d in dims],
        dims=['f_hour', 'time'] + dims,
        name='verification'
    )
    if all_ds is not None:
        valid_da = all_ds.predictors
    else:
        valid_da = ds.predictors
    for d, date in enumerate(init_times):
        verification[:, d] = valid_da.reindex(
            sample=pd.date_range(date + np.timedelta64(timedelta(hours=dt)),
                                 date + np.timedelta64(timedelta(hours=dt * forecast_steps)),
                                 freq='%sH' % int(dt)),
            method=None
        ).values
    return verification
