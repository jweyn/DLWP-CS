#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test retrieval and processing of ERA5Reanalysis data.
"""

from DLWP.data import ERA5Reanalysis

# variables = ['geopotential', 'temperature', 'relative_humidity', 'u_component_of_wind', 'v_component_of_wind',
#              '2m_temperature', 'mean_sea_level_pressure', 'land_sea_mask', 'orography',
#              'sea_surface_temperature', 'total_column_water_vapour', 'total_precipitation']
# levels = [200, 850]
variables = ['geopotential', 'temperature', 'relative_humidity', 'u_component_of_wind', 'v_component_of_wind',
             '2m_temperature', 'total_column_water_vapour', 'total_precipitation']
levels = [200, 300, 500, 700, 850, 1000]
years = [str(y) for y in range(2013, 2019)]

era = ERA5Reanalysis(root_directory='/home/disk/wave2/jweyn/Data/ERA5', file_id='era5_ensemble_2deg_3h')
era.set_variables(variables)
era.set_levels(levels)

era.retrieve(variables, levels, years=years, product='ensemble_members',
             request_kwargs={'grid': [2., 2.]}, verbose=True, delete_temporary=True)

era.open()
print(era.Dataset)
era.close()
