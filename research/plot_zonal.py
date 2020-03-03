#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Plot zonal mean climatology from forecasts.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from DLWP import verify

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
forecast_file = '%s/forecast_dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax_20170710.nc' % root_directory
validation_file = '%s/era5_2deg_3h_validation_z500_t2m_ILL.nc' % root_directory

validation_set = pd.date_range('2017-07-01', '2018-07-15', freq='6H')
validation_set = np.array(validation_set, dtype='datetime64[ns]')

forecast_date = pd.Timestamp('2017-07-04')
running_mean_days = 3
variable = 'z/500'
factor = 1. / 98.1


#%% Open data

forecast_ds = xr.open_dataset(forecast_file)
verification_ds = xr.open_dataset(validation_file)
scale_mean = verification_ds['mean'].sel(varlev=variable)
scale_std = verification_ds['std'].sel(varlev=variable)

forecast_steps = forecast_ds.dims['f_hour']
verification = verify.verification_from_series(verification_ds.sel(varlev=variable), init_times=[forecast_date],
                                               forecast_steps=forecast_steps, dt=6, f_hour_timedelta_type=False,
                                               include_zero=True)
verification = (verification.sel(time=forecast_date) * scale_std + scale_mean) * factor

if 'sample' in forecast_ds.dims:
    forecast_ds = forecast_ds.rename({'sample': 'time'})
forecast = forecast_ds.forecast.sel(time=forecast_date).isel(varlev=0)
forecast.load()
forecast = forecast * factor
forecast = xr.concat([verification.isel(f_hour=[0]), forecast], dim='f_hour', coords='minimal')


#%% Running mean

forecast_mean = forecast.mean('lon').rolling(f_hour=6 * running_mean_days, min_periods=1).mean()
forecast_long_mean = forecast.mean('lon').rolling(f_hour=6 * 15, min_periods=1, center=True).mean()
verification_mean = verification.mean('lon').rolling(f_hour=6 * running_mean_days, min_periods=1).mean()
verification_long_mean = verification.mean('lon').rolling(f_hour=6 * 15, min_periods=1, center=True).mean()


#%% Colormap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


#%% Plot

fig = plt.figure(figsize=(12, 4))
colormap = truncate_colormap(plt.get_cmap('gist_ncar'), 0.05, 0.9, n=128)

ax1 = plt.subplot(1, 2, 1)
forecast_mean.T.plot.contourf(ax=ax1, levels=np.arange(490, 591, 10), extend='both', cmap='Spectral_r')
verification_long_mean.T.plot.contour(ax=ax1, levels=[560], colors='k')
forecast_long_mean.T.plot.contour(ax=ax1, levels=[560], colors='w', )
plt.xlabel('forecast months')
plt.xticks(np.arange(0, 365 * 24, 30 * 24), labels=np.arange(0, 13, 1))
plt.ylabel('latitude')
plt.title('Forecast')

ax2 = plt.subplot(1, 2, 2)
verification_mean.T.plot.contourf(ax=ax2, levels=np.arange(490, 591, 10), extend='both', cmap='Spectral_r')
verification_long_mean.T.plot.contour(ax=ax2, levels=[560], colors='k')
plt.xlabel('forecast months')
plt.xticks(np.arange(0, 365 * 24, 30 * 24), labels=np.arange(0, 13, 1))
plt.ylabel('latitude')
plt.title('Verification')

plt.savefig(forecast_date.strftime('climatology_z500_%m%d_spec.pdf'), dpi=200, bbox_inches='tight')
plt.show()
