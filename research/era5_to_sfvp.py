from DLWP.data import ERA5Reanalysis
import xarray as xr
from windspharm.xarray import VectorWind

variables = ['u_component_of_wind', 'v_component_of_wind']
levels = [200, 850]

era = ERA5Reanalysis(root_directory='/home/disk/wave2/jweyn/Data/ERA5', file_id='era5_2deg_3h')
era.set_variables(variables)

for level in levels: 
    print('Pressure level %s' % level) 
    sfs = []
    vps = []
    era.set_levels([level]) 
    era.open() 
    print(era.Dataset) 
    for year in range(1979, 2019):
        print('Year %s' % year)
        ds = era.Dataset.sel(level=level, time=slice(str(year), str(year)))
        print('Loading data...') 
        ds.load() 
        print('Creating vector...') 
        w = VectorWind(ds.u, ds.v) 
        print('Computing streamfunction...') 
        sf, vp = w.sfvp(truncation=42)
        sfs.append(sf)
        vps.append(vp)

    print('Assigning to datasets...') 
    sf_ds = xr.Dataset({'sf': xr.concat(sfs, dim='time')}).expand_dims('level', axis=1).assign_coords(
        level=era.Dataset.level.sel(level=[level]))
    vp_ds = xr.Dataset({'vp': xr.concat(vps, dim='time')}).expand_dims('level', axis=1).assign_coords(
        level=era.Dataset.level.sel(level=[level]))
    sf_name = era.raw_files[0].replace('u_component_of_wind', 'streamfunction') 
    vp_name = era.raw_files[0].replace('u_component_of_wind', 'velocity_potential') 
    print('Writing streamfunction to %s...' % sf_name) 
    sf_ds.to_netcdf(sf_name) 
    print('Writing velocity potential to %s...' % vp_name) 
    vp_ds.to_netcdf(vp_name)