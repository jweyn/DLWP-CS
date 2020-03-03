"""
Plot some fields on the cube sphere. Full credit for the projections to https://github.com/SciTools/cartopy/issues/882
"""

from cartopy.crs import Projection
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


class rHEALPix(Projection):
    def __init__(self, central_longitude=0, north_square=0, south_square=0):
        proj4_params = [('proj', 'rhealpix'),
                        ('north_square', north_square),
                        ('south_square', south_square),
                        ('lon_0', central_longitude)]
        super(rHEALPix, self).__init__(proj4_params)

        # Boundary is based on units of m, with a standard spherical ellipse.
        nrth_x_pos = (north_square - 2) * 1e7
        sth_x_pos = (south_square - 2) * 1e7
        top = 5.05e6
        points = []
        points.extend([
                  [2e7, -5e6],
                  [2e7, top],
                  [nrth_x_pos + 1e7, top],
                  [nrth_x_pos + 1e7, 1.5e7],
                  [nrth_x_pos, 1.5e7],
                  [nrth_x_pos, top],
                  [-2e7, top]])
        if south_square != 0:
            points.append([-2e7, -top])
        points.extend([
                  [sth_x_pos, -5e6],
                  [sth_x_pos, -1.5e7],
                  [sth_x_pos + 1e7, -1.5e7],
                  [sth_x_pos + 1e7, -5e6],
                  ])
        self._boundary = sgeom.LineString(points[::-1])

        xs, ys = zip(*points)
        self._x_limits = min(xs), max(xs)
        self._y_limits = min(ys), max(ys)
        self._threshold = (self.x_limits[1] - self.x_limits[0]) / 1e4

    @property
    def boundary(self):
        return self._boundary

    @property
    def threshold(self):
        return self._threshold

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits


class RectangularHealpix(Projection):
    """
    Also known as rHEALPix in proj.4, this projection is an extension of the
    Healpix projection to present rectangles, rather than triangles, at the
    north and south poles.

    Parameters
    ----------
    central_longitude
    north_square: int
        The position for the north pole square. Must be one of 0, 1, 2 or 3.
        0 would have the north pole square aligned with the left-most square,
        and 3 would be aligned with the right-most.
    south_square: int
        The position for the south pole square. Must be one of 0, 1, 2 or 3.

    """

    def __init__(self, central_longitude=0, north_square=0, south_square=0):
        valid_square = [0, 1, 2, 3]
        if north_square not in valid_square:
            raise ValueError('north_square must be one of '
                             '{}'.format(valid_square))
        if south_square not in valid_square:
            raise ValueError('south_square must be one of {}'
                             ''.format(valid_square))

        proj4_params = [('proj', 'rhealpix'),
                        ('north_square', north_square),
                        ('south_square', south_square),
                        ('lon_0', central_longitude)]
        super(RectangularHealpix, self).__init__(proj4_params)

        # Boundary is based on units of m, with a standard spherical ellipse.
        # The hard-coded scale is the reason for not accepting the globe
        # keyword. The scale changes based on the size of the semi-major axis.
        top = 1.5e7
        width = 2e7
        h = width / 2
        box_h = width / 4

        points = [[width, -box_h],
                  [width, box_h],
                  [(north_square - 2) * h + h, box_h],
                  [(north_square - 2) * h + h, top],
                  [(north_square - 2) * box_h, top],
                  [(north_square - 2) * box_h, box_h],
                  [-width, box_h],
                  [-width, -box_h],
                  [(south_square - 2) * h, -box_h],
                  [(south_square - 2) * h, -top],
                  [(south_square - 2) * h + h, -top],
                  [(south_square - 2) * h + h, -box_h]]

        self._boundary = sgeom.LineString(points[::-1])

        xs, ys = zip(*points)
        self._x_limits = min(xs), max(xs)
        self._y_limits = min(ys), max(ys)
        self._threshold = (self.x_limits[1] - self.x_limits[0]) / 1e4

    @property
    def boundary(self):
        return self._boundary

    @property
    def threshold(self):
        return self._threshold

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits


#%% Example cube sphere plot

# ax = plt.axes(projection=rHEALPix())
# ax.stock_img()
# ax.coastlines()
# ax.gridlines()
# ax.set_global()
# plt.show()


#%% Plot some cube sphere data

root_directory = '/home/disk/wave2/jweyn/Data/DLWP'
predictor_file = '/home/gold/jweyn/Data/era5_2deg_3h_CS2_1979-2018_z-tau-t2_500-1000_tcwv_psi850.nc'
validation_file = '%s/era5_2deg_3h_validation_z500_t2m_ILL.nc' % root_directory
lsm_file = '/home/gold/jweyn/Data/era5_2deg_3h_CS2_land_sea_mask.nc'

data = xr.open_dataset(predictor_file)
scale = xr.open_dataset(validation_file)
lsm = xr.open_dataset(lsm_file)

field = data.predictors.sel(sample='2018-01-05 00:00', varlev='t2m/0') * scale['std'].sel(varlev='t2m/0') \
    + scale['mean'].sel(varlev='t2m/0')

square_lon = np.zeros(data.lon.shape)
square_lat = np.zeros(data.lat.shape)
n_side = data.lon.shape[1]
for f in range(4):
    square_lon[f], square_lat[f] = np.meshgrid(np.linspace(90. * f, 90. * (f + 1), n_side),
                                               np.linspace(-45, 45, n_side))
square_lon[4], square_lat[4] = np.meshgrid(np.linspace(0., 90., n_side),
                                           np.linspace(-135., -45., n_side))
square_lon[5], square_lat[5] = np.meshgrid(np.linspace(0., 90., n_side),
                                           np.linspace(45., 135., n_side))

fig = plt.figure(figsize=(10, 6))
ax = plt.gca()
for face in range(6):
    cf = ax.contourf(square_lon[face], square_lat[face], field.isel(face=face).values,
                     np.arange(240, 311, 5), cmap='Spectral_r', extend='both')
    ax.contour(square_lon[face], square_lat[face], lsm['lsm'].isel(face=face).values,
               [0.5], colors='k')

plt.colorbar(cf, label='$T_2$ (K)')
ax.set_aspect('equal', 'box')
plt.savefig('cs-example-t2m.pdf', dpi=200, bbox_inches='tight')
plt.show()
