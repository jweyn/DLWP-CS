from DLWP.data import ERA5Reanalysis
from DLWP.model import Preprocessor
import xarray as xr
from DLWP.barotropic.pyspharm_transforms import TransformsEngine

data_root = '/home/disk/wave2/jweyn/Data'
variable = 'total_column_water_vapour'
level = 0
truncation = 54

era = ERA5Reanalysis(root_directory='%s/ERA5' % data_root, file_id='era5_2deg_3h')
era.set_variables([variable])

print('Pressure level %s' % level)
out = []
if level != 0:
    era.set_levels([level])
era.open()
print('Loading data...')
if level != 0:
    dataset = era.Dataset.sel(level=level).copy().load()
else:
    dataset = era.Dataset.copy().load()
print(dataset)
var_short_name = [v for v in dataset.data_vars if v not in ['latitude', 'longitude', 'level']][0]
engine = TransformsEngine(dataset.dims['longitude'], dataset.dims['latitude'], truncation=truncation)
for year in range(1979, 2019):
    print('Year %s' % year)
    ds = dataset.sel(time=slice(str(year), str(year)))
    print('Creating vector...')
    data_spec = engine.grid_to_spec(ds[var_short_name].values.transpose((1, 2, 0)).astype('float64'))
    data_trunc = engine.spec_to_grid(data_spec).transpose((2, 0, 1))
    dataset[var_short_name].loc[dict(time=slice(str(year), str(year)))] = data_trunc.astype('float32')

new_file_name = era.raw_files[0].replace(variable, variable + ('_T%d' % truncation))
print('Writing new data to %s...' % new_file_name)
dataset.to_netcdf(new_file_name)

# Make a series of predictors for the variable
era.close()
era.raw_files[0] = new_file_name
era.open()
predictor_file = '%s/DLWP/era5_2deg_3h_1979-2018_%s%d-t%d.nc' % (data_root, var_short_name, level, truncation)
pp = Preprocessor(era, predictor_file=predictor_file)
pp.data_to_series(batch_samples=50000, variables='all', levels=[level], pairwise=True,
                  scale_variables=True, overwrite=False, verbose=True)
print(pp.data)
print('Output file with no varlev coordinate...')
ds_no_coord = pp.data.load().assign_coords(varlev=[0])
ds_no_coord.to_netcdf(predictor_file + '.nocoord')
