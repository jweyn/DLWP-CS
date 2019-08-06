#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Tools for re-mapping on cubed-sphere coordinates.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import subprocess
import warnings
from .base import _BaseRemap


def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.

    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    ds_new = ds.chunk(chunk_dict)
    for var in ds_new.data_vars:
        ds_new[var].encoding['contiguous'] = False
        ds_new[var].encoding['original_shape'] = ds_new[var].shape
        ds_new[var].encoding['chunksizes'] = tuple([c[0] for c in ds_new[var].chunks])
    return ds_new


class CubeSphereRemap(_BaseRemap):
    """
    Implement tools for remapping to and from a cubed sphere using TempestRemap executables.
    """

    def __init__(self, path_to_remapper=None):
        """
        Initialize a CubeSphereRemap object.

        :param path_to_remapper: str: path to the TempestRemap executables
        """
        super(CubeSphereRemap, self).__init__(path_to_remapper=path_to_remapper)
        self.remapper = os.path.join(self.path_to_remapper, 'ApplyOfflineMap')
        self.map = None
        self.inverse_map = None
        self._lat = None
        self._lon = None
        self._res = None
        self._map_exists = False
        self._inverse_map_exists = False

    def assign_maps(self, map_name=None, inverse_map_name=None):
        """
        Point to either or both of existing map conversion files for TempestRemap.

        :param map_name: str: path to forward remapping map
        :param inverse_map_name: str: path to inverse remapping map
        """
        if map_name is not None:
            self.map = map_name
            self._map_exists = True
        if inverse_map_name is not None:
            self.inverse_map = inverse_map_name
            self._inverse_map_exists = True

    def generate_offline_maps(self, lat, lon, res, map_name=None, inverse_map_name=None):
        """
        Generate offline maps for cubed sphere remapping.

        :param lat: int: number of points in the latitude dimension
        :param lon: int: number of points in the longitude dimension
        :param res: int: number of points on a side of each cube face
        :param map_name: str: file name of the forward map
        :param inverse_map_name: str: file name of the inverse map
        :return:
        """
        assert int(lat) > 0
        assert int(lon) > 0
        assert int(res) > 0
        self._lat = lat
        self._lon = lon
        self._res = res
        if map_name is None:
            self.map = 'map_LL%dx%d_CS%d.nc' % (self._lat, self._lon, self._res)
        else:
            self.map = map_name
        if inverse_map_name is None:
            self.inverse_map = 'map_CS%d_LL%dx%d.nc' % (self._res, self._lat, self._lon)
        else:
            self.inverse_map = None

        print('CubeSphereRemap: generating offline forward map...')
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateRLLMesh'),
                                     '--lat', str(self._lat), '--lon', str(self._lon), '--file', 'outLL.g'])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the lat-lon mesh.')
            print(e.output)
            raise
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateCSMesh'),
                                     '--res', str(self._res), '--file', 'outCS.g'])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the cube sphere mesh.')
            print(e.output)
            raise
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateOverlapMesh'),
                                     '--a', 'outLL.g', '--b', 'outCS.g', '--out', 'ov_LL_CS.g'])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the overlap mesh.')
            print(e.output)
            raise
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateOfflineMap'),
                                     '--in_mesh', 'outLL.g', '--out_mesh', 'outCS.g', '--ov_mesh', 'ov_LL_CS.g',
                                     '--in_np', '1', '--in_type', 'FV', '--out_type', 'FV', '--out_map', self.map])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the offline map.')
            print(e.output)
            raise
        self._map_exists = True

        print('CubeSphereRemap: generating offline inverse map...')
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateOverlapMesh'),
                                     '--a', 'outCS.g', '--b', 'outLL.g', '--out', 'ov_CS_LL.g'])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the overlap mesh.')
            print(e.output)
            raise
        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'GenerateOfflineMap'),
                                     '--in_mesh', 'outCS.g', '--out_mesh', 'outLL.g', '--ov_mesh', 'ov_CS_LL.g',
                                     '--in_np', '1', '--in_type', 'FV', '--out_type', 'FV', '--out_map',
                                     self.inverse_map])
        except subprocess.CalledProcessError as e:
            print('An error occurred while generating the offline map.')
            print(e.output)
            raise

        for f in ['outLL.g', 'outCS.g', 'ov_LL_CS.g', 'ov_CS_LL.g']:
            os.remove(f)

        self._inverse_map_exists = True
        print('CubeSphereRemap: succesfully generated offline maps (%s, %s)' % (self.map, self.inverse_map))

    def remap(self, input_file, output_file, *args):
        """
        Apply the forward remapping to data in an input_file, saved to output_file.

        :param input_file: str: path to input data file
        :param output_file: str: path to output data file
        :param args: str: other string arguments passed to the ApplyOfflineMap function
        """
        if not self._map_exists:
            raise ValueError("No forward map has been defined or generated; use 'generate_offline_maps' or "
                             "'assign_maps' functions first")
        elif not(os.path.exists(self.map)):
            raise FileNotFoundError(self.map)

        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'ApplyOfflineMap'),
                                     '--in_data', input_file, '--out_data', output_file, '--map', self.map]
                                    + list(args))
        except subprocess.CalledProcessError as e:
            print('An error occurred while applying the offline map.')
            print(e.output)
            raise

        print('CubeSphereRemap: successfully remapped data into %s' % output_file)

    def inverse_remap(self, input_file, output_file, *args):
        """
        Apply the forward remapping to data in an input_file, saved to output_file.

        :param input_file: str: path to input data file
        :param output_file: str: path to output data file
        :param args: str: other string arguments passed to the ApplyOfflineMap function
        """
        if not self._inverse_map_exists:
            raise ValueError("No inverse map has been defined or generated; use 'generate_offline_maps' or "
                             "'assign_maps' functions first")
        elif not (os.path.exists(self.inverse_map)):
            raise FileNotFoundError(self.inverse_map)

        try:
            subprocess.check_output([os.path.join(self.path_to_remapper, 'ApplyOfflineMap'),
                                     '--in_data', input_file, '--out_data', output_file, '--map', self.inverse_map]
                                    + list(args))
        except subprocess.CalledProcessError as e:
            print('An error occurred while applying the offline map.')
            print(e.output)
            raise

        print('CubeSphereRemap: successfully inverse remapped data into %s' % output_file)

    @staticmethod
    def convert_to_faces(input_file, output_file, coord_file=None, chunking=None):
        """
        Convert a file in cubed-sphere coordinates to contain dimensions for the face number and height/width of each
        of the six cube faces. Very useful for applying convolutions on the faces.

        :param input_file: str: input data file
        :param output_file: str: output data file
        :param coord_file: str: if not None, use this file to fill in missing coordinates that may have been removed
            by the remap() process
        :param chunking: dict: if provided, save the netCDF with this chunking ({dim: chunksize} pairs)
        :return xarray.Dataset: new Dataset object
        """
        # Open the dataset to convert
        ds = xr.open_dataset(input_file)
        print('CubeSphereRemap: loading data to memory')
        ds.load()

        # First, assign any coordinates missing from the input file from the coordinate file, if provided.
        if coord_file is not None:
            ds_coord = xr.open_dataset(coord_file)
            missing_coordinates = [c for c in ds.dims.keys() if (c not in ds.coords.keys() and c != 'ncol')]
            for coord in missing_coordinates:
                if coord not in ds_coord.coords.keys():
                    warnings.warn("coordinate '%s' missing in coordinate file; omitting" % coord)
                    continue
                ds = ds.assign_coords(**{coord: ds_coord.coords[coord]})

        # Create a multi-index dimension
        n_width = int(np.sqrt(ds.dims['ncol'] // 6))
        face_index = pd.MultiIndex.from_product((range(6), range(n_width), range(n_width)),
                                                names=('face', 'height', 'width'))

        # Assign the new coordinate and transpose
        new_dims = tuple([d for d in ds.dims.keys() if d != 'ncol']) + ('height', 'width', 'face')
        print('CubeSphereRemap: assigning new coordinates to dataset')
        ds_new = ds.assign_coords(ncol=face_index).unstack('ncol').transpose(*new_dims)

        # Export to a new file
        print('CubeSphereRemap: exporting data to file %s' % output_file)
        if chunking is not None:
            ds_new = to_chunked_dataset(ds_new, chunking)
        ds_new.to_netcdf(output_file)

        print('CubeSphereRemap: succesfully exported reformatted data')
        return ds_new

    @staticmethod
    def convert_from_faces(input_file, output_file, chunking=None):
        """
        Revert a file containing height, width, face dimensions into the default 'ncol' dimension so that it can be
        inverse remapped by the remapper.

        :param input_file: str: input data file
        :param output_file: str: output data file
        :param chunking: dict: if provided, save the netCDF with this chunking ({dim: chunksize} pairs)
        :return xarray.Dataset: new Dataset object
        """
        # Open the dataset to convert
        ds = xr.open_dataset(input_file)
        print('CubeSphereRemap: loading data to memory')
        ds.load()

        # Transpose the face dimension and stack the face, height, width
        fhw = ('face', 'height', 'width')
        new_dims = tuple([d for d in ds.dims.keys() if d not in fhw]) + fhw
        print('CubeSphereRemap: assigning new coordinates to dataset')
        ds_new = ds.transpose(*new_dims).stack(ncol=fhw).reset_index('ncol')

        # Export to new file
        print('CubeSphereRemap: exporting data to file %s' % output_file)
        if chunking is not None:
            ds_new = to_chunked_dataset(ds_new, chunking)
        ds_new.to_netcdf(output_file)

        print('CubeSphereRemap: succesfully exported reformatted data')
        return ds_new
