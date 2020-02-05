#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Plot forecasts from DLWP models on a global cartopy projection.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


#%% Options

root_directory = '/home/gold/jweyn/Data'
forecast_file = '%s/forecast_dlwp_era5_6h-3_CS48_tau-sfc1000-lsm-topo_UNET2-relumax_test.nc' % root_directory
verif_file = '/home/disk/wave2/jweyn/Data/DLWP/era5_2deg_3h_1979-2018_z-tau-t2_500-1000.nc'

forecast_time = pd.Timestamp('2017-12-10 00:00')
forecast_hour = 642
verif_time = pd.Timestamp('2018-01-05 00:00')

selection = {'varlev': 3}
c_selection = {'varlev': 2}

scale_factor = 1.
levels = np.arange(240, 311, 5)
c_levels = np.arange(-5000, 5001, 500)
add_line = 273


#%% Load data

forecast_ds = xr.open_dataset(forecast_file)
f_da = forecast_ds.forecast.isel(**selection).sel(f_hour=forecast_hour, sample=forecast_time) * scale_factor
f_da_c = forecast_ds.forecast.isel(**c_selection).sel(f_hour=forecast_hour, sample=forecast_time)

verif_ds = xr.open_dataset(verif_file)
v_da = verif_ds.predictors.isel(**selection).sel(sample=verif_time)
v_da = (v_da * verif_ds['std'].isel(**selection) + verif_ds['mean'].isel(**selection)) * scale_factor
v_da_c = verif_ds.predictors.isel(**c_selection).sel(sample=verif_time)
v_da_c = v_da_c * verif_ds['std'].isel(**c_selection) + verif_ds['mean'].isel(**c_selection)

# Cyclic longitude and data
lat = v_da.lat.values
f_da_r, lon = add_cyclic_point(f_da.values, coord=v_da.lon.values, axis=f_da.dims.index('lon'))
f_da_c_r, lon = add_cyclic_point(f_da_c.values, coord=v_da.lon.values, axis=f_da_c.dims.index('lon'))
v_da_r, lon = add_cyclic_point(v_da.values, coord=v_da.lon.values, axis=v_da.dims.index('lon'))
v_da_c_r, lon = add_cyclic_point(v_da_c.values, coord=v_da.lon.values, axis=v_da_c.dims.index('lon'))


#%% Map

proj = ccrs.NearsidePerspective(central_longitude=-90., central_latitude=50., satellite_height=2.e7)
transform = ccrs.PlateCarree()


#%% Iterate plots

fig = plt.figure(figsize=(10, 6))

# Forecast plot
ax1 = plt.subplot(1, 2, 1, projection=proj)
ax1.set_global()
ax1.coastlines(color=(0.5, 0.5, 0.5))
ax1.gridlines(linewidth=0.4, zorder=1)
ax1.contourf(lon, lat, f_da_r, levels=levels,
             cmap='Spectral_r', extend='both', transform=transform)
if add_line is not None:
    ax1.contour(lon, lat, f_da_r, levels=(add_line,),
                colors='b', transform=transform)
ax1.contour(lon, lat, f_da_c_r, levels=c_levels,
            colors='k', linewidths=0.7, transform=transform)
ax1.set_title(pd.Timestamp.strftime(forecast_time, 'Forecast: %Y-%m-%d %H:%M Z + {:d} h'.format(forecast_hour)))

# Verification plot
ax2 = plt.subplot(1, 2, 2, projection=proj)
ax2.set_global()
ax2.coastlines(color=(0.5, 0.5, 0.5))
ax2.gridlines(linewidth=0.4, zorder=1)
c = ax2.contourf(lon, lat, v_da_r, levels=levels,
                 cmap='Spectral_r', extend='both', transform=transform)
if add_line is not None:
    ax2.contour(lon, lat, v_da_r, levels=(add_line,),
                colors='b', transform=transform)
ax2.contour(lon, lat, v_da_c_r, levels=c_levels,
            colors='k', linewidths=0.7, transform=transform)
ax2.set_title(pd.Timestamp.strftime(verif_time, 'Verification: %Y-%m-%d %H:%M Z'))

# Add colorbar
fig.subplots_adjust(bottom=0.00)
cbar_ax = fig.add_axes([0.25, 0.02, 0.50, 0.06])
cbar_ax.set_in_layout(False)
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')

plt.savefig('map.pdf', dpi=200, bbox_inches='tight')
plt.show()

