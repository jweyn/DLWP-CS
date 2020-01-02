#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Remap a pre-processed predictor file to a cubed sphere. Note that TempestRemap can be quite finicky about the
structure of the input netCDF file and throws segmentation faults for no apparent reason. It may be necessary to
remove the string coordinate variable 'varlev' in the input file.
"""

import os
from DLWP.remap import CubeSphereRemap

data_root = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/era5_2deg_3h_1979-2018_psi_850.nc.nocoord' % data_root
remap_file = '%s/era5_2deg_3h_CS_1979-2018_psi_850.nc' % data_root

csr = CubeSphereRemap()

csr.generate_offline_maps(lat=91, lon=180, res=48, inverse_lat=True)
csr.remap(predictor_file, '%s/temp.nc' % data_root, '--var', 'predictors')
csr.convert_to_faces('%s/temp.nc' % data_root, remap_file,
                     coord_file='.'.join(predictor_file.split('.')[:-1]),
                     chunking={'sample': 1, 'varlev': 1})

os.remove('%s/temp.nc' % data_root)
