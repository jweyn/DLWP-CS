import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing
import itertools as it
import os
import netCDF4
from datetime import datetime
from ecmwfapi import ECMWFDataServer


param_codes = {
    'geopotential': '156',
    'daily_2m_temperature': '167'
}

#%% Parameters

output_directory = '/home/disk/wave2/jweyn/Data/S2S/ECMF'
overwrite = True

dates_1 = pd.date_range('2018-01-01', '2018-12-31', freq='7D')
dates_2 = pd.date_range('2018-01-04', '2018-12-31', freq='7D')
dates = dates_1.append(dates_2).sort_values()

hindcast_years = [str(year) for year in range(2014, 2018)]

variable = 'geopotential'
level = '500'
# variable = 'daily_2m_temperature'
# level = ''

output_file = '%s/%s_%s_2014-2017_from_2018.nc' % (output_directory, variable, level)

# Instantaneous fields, 24-hourly, pressure level
server_args = {
    "class": "s2",
    "dataset": "s2s",
    "expver": "prod",
    "levelist": level,
    "levtype": "pl",
    "model": "glob",
    "origin": "ecmf",
    "param": param_codes[variable],
    "step": "/".join([str(n) for n in range(0, 1105, 24)]),
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
}
# # Daily fields, surface
# server_args = {
#     "class": "s2",
#     "dataset": "s2s",
#     "expver": "prod",
#     "levtype": "sfc",
#     "model": "glob",
#     "origin": "ecmf",
#     "param": "167",
#     "step": "/".join(['%s-%s' % (n, n+24) for n in range(0, 1081, 24)]),
#     "stream": "enfo",
#     "time": "00:00:00",
#     "type": "cf",
# }

procs = 4


#%% Iterate over the dates

def retrieve(yd):
    y, d = yd
    output_file = '%s/%s_%s_%s.grib' % (output_directory, d.strftime(y + '%m%d'), variable, level)

    print('PID %s: Fetching %s' % (os.getpid(), output_file))
    args = {
        'date': d.strftime('%Y-%m-%d'),
        'target': output_file
    }
    args.update(server_args)
    if int(d.year) != int(y):
        args.update({'hdate': d.strftime(y + '-%m-%d')})
        args['stream'] = 'enfh'

    if os.path.isfile(output_file):
        if not overwrite:
            print('PID %s: file %s already exists' % (os.getpid(), output_file))
            return output_file
    try:
        server.retrieve(args)
    except:
        print('PID %s: API exception when retrieving %s' % (os.getpid(), output_file))
        print('PID %s: server args were:' % os.getpid())
        for k, v in args.items():
            print('    %s: %s' % (k, v))

    return output_file


server = ECMWFDataServer()

if procs == 1:
    all_files = []
    for year in hindcast_years:
        for date in dates:
            all_files.append(retrieve((year, date)))
else:
    pool = multiprocessing.Pool(processes=procs)
    all_files = pool.map(retrieve, it.product(hindcast_years, dates))
    pool.close()
    pool.terminate()
    pool.join()

all_files.sort()


#%% Try using the netCDF4 library to make a file that does not seg fault tempest-remap

ds = xr.open_mfdataset(all_files, concat_dim='time', engine='cfgrib', combine='nested')

nc_fid = netCDF4.Dataset(output_file, 'w')
nc_fid.createDimension('time', 0)
nc_fid.createDimension('f_hour', ds.dims['step'])
nc_fid.createDimension('lat', ds.dims['latitude'])
nc_fid.createDimension('lon', ds.dims['longitude'])

# Create spatial coordinates
nc_var = nc_fid.createVariable('lat', np.float32, 'lat')
nc_var.setncatts({
    'long_name': 'Latitude',
    'units': 'degrees_north'
})
nc_fid.variables['lat'][:] = ds['latitude'].values

nc_var = nc_fid.createVariable('lon', np.float32, 'lon')
nc_var.setncatts({
    'long_name': 'Longitude',
    'units': 'degrees_east'
})
nc_fid.variables['lon'][:] = ds['longitude'].values

# Create initialization time reference variable
nc_var = nc_fid.createVariable('time', np.float32, 'time')
time_units = 'hours since 1970-01-01 00:00:00'

nc_var.setncatts({
    'long_name': 'Initialization Time',
    'units': time_units
})
times = np.array([datetime.utcfromtimestamp(d / 1e9)
                  for d in ds['time'].values.astype(datetime)])
nc_fid.variables['time'][:] = netCDF4.date2num(times, time_units)

nc_var = nc_fid.createVariable('f_hour', np.float32, 'f_hour')
nc_var.setncatts({
    'long_name': 'Forecast Hour',
    'units': 'h'
})
nc_fid.variables['f_hour'][:] = np.arange(24. if level == '' else 0., 1105., 24.)

dims = ('time', 'f_hour', 'lat', 'lon')
chunks = (1, ds.dims['step'], ds.dims['latitude'], ds.dims['longitude'])
short_name = list(ds.data_vars.keys())[0]
data = nc_fid.createVariable(short_name, np.float32, dims, chunksizes=chunks)
data.setncatts({
    'long_name': ds[short_name].attrs['long_name'],
    'units': ds[short_name].attrs['units'],
    'standard_name': ds[short_name].attrs['standard_name'],
    '_FillValue': np.array(netCDF4.default_fillvals['f4']).astype(np.float32)
})

data[:] = ds[short_name].values

nc_fid.close()
