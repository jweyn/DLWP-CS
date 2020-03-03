#
# Copyright (c) 2020 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Plot forecasts from DLWP models on a global cartopy projection. Iterates over forecast hours.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from DLWP import verify


#%% Options

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
forecast_file = '%s/forecast_dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax_20170710.nc' % root_directory
verif_file = '%s/era5_2deg_3h_1979-2018_z-tau-t2_500-1000.nc' % root_directory
climo_file = '%s/era5_2deg_3h_1979-2010_climatology_z500-z1000-t2.nc' % root_directory

init_time = pd.Timestamp('2017-07-04 00:00')

selection = {'varlev': 0}
c_selection = {'varlev': 2}
climo_sel = {'varlev': 0}
climo_c_sel = {'varlev': 1}

scale_factor = 0.01
levels = np.arange(500, 581, 5)
c_levels = np.arange(-5000, 5001, 500)
add_line = 540

validation_set = pd.date_range('2017-07-01', '2018-07-31', freq='6H')

plot_directory = '/home/gold/jweyn/Data/Plots/2017070400'
plot_prefix = 'dlwp_z500'


#%% Load data

print('Loading forecast data...')
forecast_ds = xr.open_dataset(forecast_file)
if 'sample' in forecast_ds.dims:
    forecast_ds = forecast_ds.rename({'sample': 'time'})
f_da = forecast_ds.forecast.isel(**selection).sel(time=init_time) * scale_factor
f_da_c = forecast_ds.forecast.isel(**c_selection).sel(time=init_time)

print('Loading verification...')
verif_ds = xr.open_dataset(verif_file)
verif_ds = verif_ds.sel(sample=validation_set).load()
verification = verify.verification_from_series(verif_ds, init_times=[init_time],
                                               forecast_steps=len(f_da.f_hour), dt=6, f_hour_timedelta_type=False)
v_da = verification.isel(**selection).sel(time=init_time)
v_da = (v_da * verif_ds['std'].isel(**selection) + verif_ds['mean'].isel(**selection)) * scale_factor
v_da_c = verification.isel(**c_selection).sel(time=init_time)
v_da_c = v_da_c * verif_ds['std'].isel(**c_selection) + verif_ds['mean'].isel(**c_selection)

# Cyclic longitude and data
lat = v_da.lat.values
f_da_r, _ = add_cyclic_point(f_da.values, coord=v_da.lon.values, axis=f_da.dims.index('lon'))
f_da_c_r, _ = add_cyclic_point(f_da_c.values, coord=v_da.lon.values, axis=f_da_c.dims.index('lon'))
v_da_r, _ = add_cyclic_point(v_da.values, coord=v_da.lon.values, axis=v_da.dims.index('lon'))
v_da_c_r, lon = add_cyclic_point(v_da_c.values, coord=v_da.lon.values, axis=v_da_c.dims.index('lon'))

# Climatology data
if climo_file is not None:
    print('Loading climatology...')
    climo_ds = xr.open_dataset(climo_file)
    c_da = (climo_ds.predictors.isel(**climo_sel) * climo_ds['std'].isel(**climo_sel)
            + climo_ds['mean'].isel(**climo_sel)) * scale_factor
    c_da_c = climo_ds.predictors.isel(**climo_c_sel) * climo_ds['std'].isel(**climo_c_sel) \
           + climo_ds['mean'].isel(**climo_c_sel)
    c_da_r, _ = add_cyclic_point(c_da.values, coord=c_da.lon.values, axis=c_da.dims.index('lon'))
    c_da_c_r, _ = add_cyclic_point(c_da_c.values, coord=c_da.lon.values, axis=c_da_c.dims.index('lon'))


#%% Map

proj = ccrs.NearsidePerspective(central_longitude=-90., central_latitude=50., satellite_height=2.e7)
transform = ccrs.PlateCarree()


#%% Iterate plots

os.makedirs(plot_directory, exist_ok=True)
num_subplots = 2 if climo_file is None else 3

for f, hour in enumerate(f_da.f_hour.values):
    print('Plotting %d of %d...' % (f + 1, len(f_da.f_hour)))
    valid_time = init_time + pd.Timedelta(hours=hour)

    fig = plt.figure(figsize=(5 * num_subplots, 5))

    # Forecast plot
    ax1 = plt.subplot(1, num_subplots, 1, projection=proj)
    ax1.set_global()
    ax1.coastlines(color=(0.5, 0.5, 0.5))
    ax1.gridlines(linewidth=0.4, zorder=1)
    ax1.contourf(lon, lat, f_da_r[f], levels=levels,
                 cmap='Spectral_r', extend='both', transform=transform)
    if add_line is not None:
        ax1.contour(lon, lat, f_da_r[f], levels=(add_line,),
                    colors='b', transform=transform)
    ax1.contour(lon, lat, f_da_c_r[f], levels=c_levels,
                colors='k', linewidths=0.7, transform=transform)
    ax1.set_title('%d-hour DLWP forecast' % hour)

    # Verification plot
    ax2 = plt.subplot(1, num_subplots, 2, projection=proj)
    ax2.set_global()
    ax2.coastlines(color=(0.5, 0.5, 0.5))
    ax2.gridlines(linewidth=0.4, zorder=1)
    c = ax2.contourf(lon, lat, v_da_r[f], levels=levels,
                     cmap='Spectral_r', extend='both', transform=transform)
    if add_line is not None:
        ax2.contour(lon, lat, v_da_r[f], levels=(add_line,),
                    colors='b', transform=transform)
    ax2.contour(lon, lat, v_da_c_r[f], levels=c_levels,
                colors='k', linewidths=0.7, transform=transform)
    ax2.set_title('verification')

    if climo_file is not None:
        ax3 = plt.subplot(1, num_subplots, 3, projection=proj)
        ax3.set_global()
        ax3.coastlines(color=(0.5, 0.5, 0.5))
        ax3.gridlines(linewidth=0.4, zorder=1)
        # Get index of day of year for valid time
        doy_index = np.where(climo_ds['dayofyear'] == valid_time.dayofyear)[0][0]
        c = ax3.contourf(lon, lat, c_da_r[doy_index], levels=levels,
                         cmap='Spectral_r', extend='both', transform=transform)
        if add_line is not None:
            ax3.contour(lon, lat, c_da_r[doy_index], levels=(add_line,),
                        colors='b', transform=transform)
        ax3.contour(lon, lat, c_da_c_r[doy_index], levels=c_levels,
                    colors='k', linewidths=0.7, transform=transform)
        ax3.set_title('climatology')

    fig.suptitle(pd.Timestamp.strftime(valid_time, 'Valid: %Y-%m-%d %H:%M Z'))

    # Add colorbar
    fig.subplots_adjust(top=1.00, bottom=0.00)
    cbar_ax = fig.add_axes([0.26, 0.02, 0.50, 0.06])
    cbar_ax.set_in_layout(False)
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')

    plot_file = '%s/%s_{:0>4d}.png'.format(f) % (plot_directory, plot_prefix)
    print('    %s' % plot_file)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

