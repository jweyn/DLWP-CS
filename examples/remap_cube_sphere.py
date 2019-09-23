#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test remapping of a pre-processed predictor file to a cubed sphere.
"""

import os
from DLWP.remap import CubeSphereRemap

data_root = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '%s/cfs_6h_1979-2010_z3-5-7-10_tau_sfc.nc' % data_root
remap_file = '%s/cfs_6h_CS48_1979-2010_z3-5-7-10_tau_sfc.nc' % data_root

csr = CubeSphereRemap()

csr.generate_offline_maps(lat=73, lon=144, res=48)
csr.remap(predictor_file, '%s/temp.nc' % data_root, '--var', 'predictors')
csr.convert_to_faces('%s/temp.nc' % data_root, remap_file, coord_file=predictor_file,
                     chunking={'sample': 1, 'varlev': 1})

os.remove('%s/temp.nc' % data_root)
