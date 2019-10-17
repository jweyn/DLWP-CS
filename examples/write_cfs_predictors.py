#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test conversion of CFS data into preprocessed predictors/targets for the DLWP model.
"""

from DLWP.data import CFSReanalysis
from DLWP.model import Preprocessor
from datetime import datetime
import pandas as pd
import xarray as xr

start_date = datetime(1979, 1, 1)
end_date = datetime(2010, 12, 31)
dates = list(pd.date_range(start_date, end_date, freq='D').to_pydatetime())
variables = ['HGT'] * 4 + ['TMP'] * 4 + ['R H'] * 4 + ['U GRD'] * 4 + ['V GRD'] * 4 + ['TMP2']
levels = [200, 500, 850, 1000] * 5 + [0]
data_root = '/home/disk/wave2/jweyn/Data'

cfs = CFSReanalysis(root_directory='%s/CFSR' % data_root, file_id='dlwp_')
cfs.set_dates(dates)
cfs.open()

# Merge with another set of data that includes surface data
cfs2 = CFSReanalysis(root_directory='%s/CFSR' % data_root, file_id='6h_sfc_')
cfs2.set_dates(dates)
cfs2.open()
cfs.Dataset = xr.merge([cfs.Dataset, cfs2.Dataset])

# Select northern hemisphere
# cfs.Dataset = cfs.Dataset.isel(lat=(cfs.Dataset.lat >= 0.0))

pp = Preprocessor(cfs, predictor_file='%s/DLWP/cfs_6h_1979-2010_ztruv-t2_200-500-850-1000.nc' % data_root)
pp.data_to_series(batch_samples=1000, variables=variables, levels=levels, pairwise=True,
                  scale_variables=True, overwrite=False, verbose=True)
print(pp.data)
pp.close()
