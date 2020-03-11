#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Convert ERA5 data into preprocessed predictors/targets for the DLWP model.
"""

from DLWP.data import ERA5Reanalysis
from DLWP.data.era5 import pressure_variable_names as names, surface_variable_names
from DLWP.model import Preprocessor

variables = ['geopotential', 'geopotential', '2m_temperature', 'total_column_water_vapour']
levels = [500, 1000, 0, 0]
data_root = '/home/disk/wave2/jweyn/Data/ERA5'

era = ERA5Reanalysis(root_directory=data_root, file_id='era5_1deg_3h')
era.set_variables(list(set(variables)))
era.set_levels(list(set(levels)))
era.open()
era.Dataset.load()

names.update(surface_variable_names)
pp = Preprocessor(era, predictor_file='../Data/era5_1deg_3h_1979-2018_z-t2-tcwv_500-1000.nc')
pp.data_to_series(batch_samples=100000, variables=[names[v] for v in variables], levels=levels, pairwise=True,
                  scale_variables=False, overwrite=False, verbose=True)
print(pp.data)
pp.close()
