#   Copyright 2022 Sean Patrick Santos
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Classes for bin mass grids used by JEFE's bin model."""

import numpy as np

from bin_model.math_utils import add_logs


class MassGrid:
    """
    Represent the bin model's mass grid.

    Attributes:
    d_min, d_max - Minimum/maximum particle diameter (m).
    x_min, x_max - Minimum/maximum particle size (scaled mass units).
    lx_min, lx_max - Natural logarithm of x_min/x_max, for convenience.
    num_bins - Number of model bins.
    bin_bounds - Array of size num_bins+1 containing edges of bins.
                 This array is in units of log(scaled mass), i.e. the
                 first value is lx_min and last value is lx_max.
    bin_bounds_d - Same as bin_bounds, but for diameters of particles at the
                   bin edges (m).
    bin_widths - Array of size num_bins containing widths of each bin
                 in log-units, i.e. `sum(bin_widths) == lx_max-lx_min`.
    """

    mass_grid_type_str_len = 32
    """Length of mass_grid_type string on file."""

    def __init__(self, constants, bin_bounds):
        self.constants = constants
        self.num_bins = len(bin_bounds)-1
        self.bin_bounds = bin_bounds
        bin_bounds_m = np.exp(self.bin_bounds)
        self.bin_bounds_d = constants.scaled_mass_to_diameter(bin_bounds_m)
        self.bin_widths = self.bin_bounds[1:] - self.bin_bounds[:-1]
        self._lrm, self._rm_idx = \
            self._calculate_rain_threshold_info(constants.rain_m)

    def find_bin(self, lx):
        """Find the index of the bin containing the given mass value.

        Arguments:
        lx - Natural logarithm of the mass to search for.

        Returns an integer i such that
            bin_bounds[i] <= lx < bin_bounds[i+1]
        If `lx < lx_min`, the value returned is -1. If `lx >= lx_max`, the
        value returned is `num_bins`.
        """
        for i in range(self.num_bins+1):
            if self.bin_bounds[i] >= lx:
                return i-1
        return self.num_bins

    def find_sum_bins(self, lx1, lx2, ly1, ly2):
        """Find the range of bins gaining mass from two colliding bins.

        Arguments:
        lx1, lx2 - Lower/upper bound for x bin.
        ly1, ly2 - Lower/upper bound for y bin.

        Returns a tuple `(idx, num)`, where idx is the smallest bin that will
        gain mass from collisions between particles in the two bins, and num is
        the number of bins that will gain mass.

        Note that although `idx + num - 1` is typically the index of the
        largest output bin, it can be equal to `num_bins` if some of the mass
        lies outside of the range.
        """
        tol = 1.e-10 # Tolerance for considering bin ranges non-overlapping.
        low_sum_log = add_logs(lx1, ly1)
        idx = self.find_bin(low_sum_log)
        if idx < self.num_bins and self.bin_bounds[idx+1] - low_sum_log <= tol:
            idx += 1
        high_sum_log = add_logs(lx2, ly2)
        high_idx = self.find_bin(high_sum_log)
        if high_idx >= 0 and high_sum_log - self.bin_bounds[high_idx] <= tol:
            high_idx -= 1
        num = high_idx - idx + 1
        return idx, num

    def construct_sparsity_structure(self, boundary=None):
        """Find the sparsity structure of a collision tensor using this grid.

        Arguments:
        boundary (optional) - Either 'open' or 'closed'. Default is 'open'.

        We represent the collision kernel as a tensor indexed by three bins:

         1. The bin acting as a source of mass (labeled the "x" bin).
         2. A bin colliding with the source bin (the "y" bin).
         3. The destination bin that mass is added to (the "z" bin).

        For a given x and y bin, not every particle size can be produced; only
        a small range of z bins will have nonzero collision tensor. To represent
        these ranges, the function returns a tuple `(idxs, nums, max_num)`:

         - idxs is an array of shape `(num_bins, num_bins)` that contains the
           indices of the smallest z bins for which the tensor is non-zero for
           each index of x and y bins.
         - nums is also of shape `(num_bins, num_bins)`, and contains the
           number of z bins that have nonzero tensor elements for each x and y.
         - max_num is simply the value of the maximum entry in nums, and is
           returned only for convenience.

        The outputs treat particles that are larger than the largest bin as
        belonging to an extra bin that stretches to infinity; therefore, the
        maximum possible values of both idxs and idxs + nums - 1 are num_bins,
        not num_bins - 1.

        If `boundary == 'closed'`, this behavior is modified so that there is
        no bin stretching to infinity, the excessive mass is placed in the
        largest finite bin, and the maximum values of idxs and idxs + nums - 1
        are actually num_bins - 1.

        For geometrically-spaced mass grids, note that the entries of nums are
        all 1 or 2, so max_num is 2.
        """
        if boundary is None:
            boundary = 'open'
        if boundary not in ('open', 'closed'):
            raise ValueError("invalid boundary specified: " + str(boundary))
        nb = self.num_bins
        bb = self.bin_bounds
        idxs = np.zeros((nb, nb), dtype=np.int_)
        nums = np.zeros((nb, nb), dtype=np.int_)
        for i in range(nb):
            for j in range(nb):
                idxs[i,j], nums[i,j] = self.find_sum_bins(
                    bb[i], bb[i+1], bb[j], bb[j+1]
                )
        if boundary == 'closed':
            for i in range(nb):
                for j in range(nb):
                    if idxs[i,j] == nb:
                        idxs[i,j] = nb - 1
                    elif idxs[i,j] + nums[i,j] - 1 == nb:
                        nums[i,j] -= 1
        max_num = nums.max()
        return idxs, nums, max_num

    def _calculate_rain_threshold_info(self, rain_m):
        """Return information about cloud-rain threshold in the grid.

        Arguments:
        rain_m - Nondimensionalized cloud-rain threshold mass.

        Returns a tuple containing log(rain_m) and the bin where the threshold
        is present.
        """
        bb = self.bin_bounds
        lrm = max(min(np.log(rain_m), bb[-1]), bb[0])
        rm_idx = min(max(self.find_bin(lrm), 0), self.num_bins-1)
        return (lrm, rm_idx)

    def moment_weight_vector(self, n, cloud_only=None, rain_only=None):
        """Calculate weight vector corresponding to a moment of the DSD.

        Arguments:
        n - Moment to calculate (can be any real number).
        cloud_only (optional) - Only count cloud-sized drops.
        rain_only (optional) - Only count rain-sized drops.

        The returned value is a vector such that
            diameter_scale**n * np.dot(weight_vector, dsd) / self.std_mass
        is a moment of the DSD, or if the DSD is in dimensionless units,
            np.dot(weight_vector, dsd)
        is the dimensionless DSD.
        """
        if cloud_only is None:
            cloud_only = False
        if rain_only is None:
            rain_only = False
        if cloud_only and rain_only:
            raise RuntimeError("moment cannot be both cloud-only and rain-only")
        bb = self.bin_bounds
        bw = self.bin_widths
        if n == 3:
            weight_vector = np.ones((self.num_bins,))
            if cloud_only:
                weight_vector[self._rm_idx+1:] = 0.
                weight_vector[self._rm_idx] *= \
                    (self._lrm - bb[self._rm_idx]) / bw[self._rm_idx]
            elif rain_only:
                weight_vector[:self._rm_idx] = 0.
                weight_vector[self._rm_idx] *= \
                    1. - (self._lrm - bb[self._rm_idx]) / bw[self._rm_idx]
        else:
            exponent = n / 3. - 1.
            weight_vector = np.exp(exponent * bb) / exponent
            weight_vector = (weight_vector[1:] - weight_vector[:-1]) / bw
            if cloud_only:
                weight_vector[self._rm_idx+1:] = 0.
                weight_vector[self._rm_idx] = \
                    (np.exp(exponent * self._lrm) / exponent
                     - np.exp(exponent * bb[self._rm_idx]) / exponent) / bw[self._rm_idx]
            elif rain_only:
                weight_vector[:self._rm_idx] = 0.
                weight_vector[self._rm_idx] = \
                    (np.exp(exponent * bb[self._rm_idx+1]) / exponent \
                     - np.exp(exponent * self._lrm) / exponent) / bw[self._rm_idx]
        return weight_vector

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_dimension('mass_grid_type_str_len',
                                    self.mass_grid_type_str_len)
        netcdf_file.write_characters('mass_grid_type',
                                     'Irregular',
                                     'mass_grid_type_str_len',
                                     'Type of mass grid')
        netcdf_file.write_dimension('num_bins', self.num_bins)
        netcdf_file.write_dimension('num_bin_boundaries', self.num_bins+1)
        netcdf_file.write_array('bin_bounds', self.bin_bounds,
            'f8', ('num_bin_boundaries',), '1',
            'Logarithms of nondimensionalized boundaries between'
            ' mass grid bins')

    @classmethod
    def from_netcdf(cls, netcdf_file, constants):
        """Retrieve a MassGrid object from a NetcdfFile."""
        mass_grid_type = netcdf_file.read_characters('mass_grid_type')
        if mass_grid_type == 'Irregular':
            bin_bounds = netcdf_file.read_array('bin_bounds')
            return MassGrid(constants, bin_bounds)
        if mass_grid_type == 'Geometric':
            d_min = netcdf_file.read_scalar('d_min')
            d_max = netcdf_file.read_scalar('d_max')
            num_bins = netcdf_file.read_dimension('num_bins')
            return GeometricMassGrid(constants, d_min, d_max, num_bins)
        raise RuntimeError("unrecognized mass_grid_type in file")


class GeometricMassGrid(MassGrid):
    """
    Represent a mass grid with even geometric spacing.

    Initialization arguments:
    constants - ModelConstants object.
    d_min, d_max - Minimum/maximum particle diameter (m).
    num_bins - Number of model bins

    Attributes:
    dlx - Constant width of each bin.
    """

    def __init__(self, constants, d_min, d_max, num_bins):
        self.d_min = d_min
        self.d_max = d_max
        self.x_min = constants.diameter_to_scaled_mass(d_min)
        self.x_max = constants.diameter_to_scaled_mass(d_max)
        self.lx_min = np.log(self.x_min)
        self.lx_max = np.log(self.x_max)
        self.dlx = (self.lx_max - self.lx_min) / num_bins
        bin_bounds = np.linspace(self.lx_min, self.lx_max, num_bins+1)
        super().__init__(constants, bin_bounds)

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_dimension('mass_grid_type_str_len',
                                    self.mass_grid_type_str_len)
        netcdf_file.write_characters('mass_grid_type',
                                     'Geometric',
                                     'mass_grid_type_str_len',
                                     'Type of mass grid')
        netcdf_file.write_scalar('d_min', self.d_min,
            'f8', 'm',
            'Smallest diameter particle in mass grid')
        netcdf_file.write_scalar('d_max', self.d_max,
            'f8', 'm',
            'Largest diameter particle in mass grid')
        netcdf_file.write_dimension('num_bins', self.num_bins)
