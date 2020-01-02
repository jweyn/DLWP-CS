#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Convert ERA5 data into preprocessed predictors/targets for the DLWP model.
"""

from DLWP.data import ERA5Reanalysis
from DLWP.model import Preprocessor
from datetime import datetime
import pandas as pd

start_date = datetime(1979, 1, 1)
end_date = datetime(2018, 12, 31)
dates = list(pd.date_range(start_date, end_date, freq='D').to_pydatetime())
variables = ['2m_temperature']
levels = [0]
data_root = '/home/disk/wave2/jweyn/Data'

era = ERA5Reanalysis(root_directory='%s/CFSR' % data_root, file_id='era5_2deg_3h')
era.set_variables(variables)
era.set_levels(levels)
era.open()

pp = Preprocessor(era, predictor_file='%s/DLWP/era5_2deg_3h_1979-2018_t2m.nc' % data_root)
pp.data_to_series(batch_samples=1000, variables='all', levels='all', pairwise=True,
                  scale_variables=True, overwrite=False, verbose=True)
print(pp.data)
pp.close()
