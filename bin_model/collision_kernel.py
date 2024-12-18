#   Copyright 2022-2024 Sean Patrick Santos
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

"""Classes for construction of collision kernels for use in JEFE's bin model."""

from abc import ABC, abstractmethod
import enum

import numpy as np
from scipy.integrate import dblquad

from bin_model.math_utils import add_logs, sub_logs, dilogarithm

# pylint: disable-next=too-many-locals
def beard_v(const, d):
    """Terminal velocity of a particle of the given diameter.

    Arguments:
    d - Particle diameter in meters.

    Returned value is velocity according to Beard (1976) in meters/second, at
    standard temperature and pressure.

    For diameters over 7 millimeters, a constant value is returned.
    """
    d = min(7.e-3, d)
    eta = 1.818e-5
    l = 6.62e-8
    g = 9.81
    csc = 1 + 2.51 * l / d
    sigma = 7.197e-2
    deltap = (const.rho_water - const.rho_air)
    if d < 1.9e-5:
        c1 = deltap * g / (18. * eta)
        return c1 * csc * d**2
    if d < 1.07e-3:
        c2 = 4. * const.rho_air * deltap * g \
            / (3. * eta**2)
        x = np.log(c2 * d**3)
        b = [-0.318657e1, 0.992696e0, -0.153193e-2, -0.987059e-3,
             -0.578878e-3, 0.855176e-4, -0.327815e-5]
        y = b[0] + x*(b[1] + x*(b[2] + x*(b[3]
                    + x*(b[4] + x*(b[5] + x*b[6])))))
        return eta * csc * np.exp(y) / (const.rho_air * d)
    c3 = 4 * deltap * g / (3. * sigma)
    bo = c3 * d**2
    np6 = (sigma**3 * const.rho_air**2 / (eta**4 * deltap * g))**(1./6.)
    x = np.log(bo * np6)
    b = [-0.500015e1, 0.523778e1, -0.204914e1, 0.475294,
         -0.542819e-1, 0.238449e-2]
    y = b[0] + x*(b[1] + x*(b[2] + x*(b[3] + x*(b[4] + x*b[5]))))
    nre = np6 * np.exp(y)
    return eta * nre / (const.rho_air * d)

# pylint: disable-next=too-many-locals
def sc_efficiency(d1, d2):
    """Collection efficiency calculated using the Scott and Chen formula.

    Arguments:
    d1, d2 - Particle diameters in meters.

    Returned value is collection efficiency according to Scott and Chen (1970),
    with the typo correction given by Ziv and Levin (1974).

    For collector radii below 10 microns, the efficiency is calculated as if the
    collector radius was 10 microns, with the same ratio between particle sizes.
    """
    al = 0.5e6 * max(d1, d2)
    asm = 0.5e6 * min(d1, d2)
    x = asm / al
    al = max(al, 10.)
    m = 6.
    n = 1.5
    b = (1.587 * al + 32.73 + 344. * (20. / al)**1.56 \
        * np.exp(-(al-10.)/15.) * np.sin(np.pi*(al-10.)/63.)) / al**2
    x1 = 0.
    u2 = 0.
    prev_x1 = 100.
    prev_u2 = 100.
    rtol = 1.e-10
    for _ in range(100):
        if np.abs(x1 - prev_x1) < rtol * prev_x1 and \
            np.abs(u2 - prev_u2) < rtol * prev_u2:
            break
        prev_x1 = x1
        prev_u2 = u2
        x1 = b / (1. - b * (1. + prev_u2**n)**(-1./n))
        u2 = b / (1.75 - b * (1. + prev_x1**m)**(-1./m))
    x1_term = b / (x**m + x1**m)**(1./m)
    u2_term = b / ((1.-x)**n + u2**n)**(1./n)
    return ((1. + x - x1_term - u2_term) / (1. + x)) **2

_hall_radii = np.array([
    10., 20., 30., 40., 50., 60., 70., 100., 150., 200., 300.,
])

_hall_efficiencies = np.array([
    [1.e-4, 1.e-4, 1.e-4, 0.014, 0.017, 0.019, 0.022, 0.027, 0.030, 0.033,
     0.035, 0.037, 0.038, 0.038, 0.037, 0.036, 0.035, 0.032, 0.029, 0.027], # 10
    [1.e-4, 1.e-4, 0.005, 0.016, 0.022, 0.030, 0.043, 0.052, 0.064, 0.072,
     0.079, 0.082, 0.080, 0.076, 0.067, 0.057, 0.048, 0.040, 0.033, 0.027], # 20
    [1.e-4, 0.002, 0.020, 0.040, 0.085, 0.170, 0.270, 0.400, 0.500, 0.550,
     0.580, 0.590, 0.580, 0.540, 0.510, 0.490, 0.470, 0.045, 0.470, 0.520], # 30
    [0.001, 0.070, 0.280, 0.500, 0.620, 0.680, 0.740, 0.780, 0.800, 0.800,
     0.800, 0.780, 0.770, 0.760, 0.770, 0.770, 0.780, 0.790, 0.950, 1.400], # 40
    [0.005, 0.400, 0.600, 0.700, 0.780, 0.830, 0.860, 0.880, 0.900, 0.900,
     0.900, 0.900, 0.890, 0.880, 0.880, 0.890, 0.920, 1.010, 1.300, 2.300], # 50
    [0.050, 0.430, 0.640, 0.770, 0.840, 0.870, 0.890, 0.900, 0.910, 0.910,
     0.910, 0.910, 0.910, 0.920, 0.930, 0.950, 1.000, 1.030, 1.700, 3.000], # 60
    [0.200, 0.580, 0.750, 0.840, 0.880, 0.900, 0.920, 0.940, 0.950, 0.950,
     0.950, 0.950, 0.950, 0.950, 0.970, 1.000, 1.020, 1.040, 2.300, 4.000], # 70
    [0.500, 0.790, 0.910, 0.950, 0.950, 1.000, 1.000, 1.000, 1.000, 1.000,
     1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000], # 100
    [0.770, 0.930, 0.970, 0.970, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
     1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000], # 150
    [0.870, 0.960, 0.980, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
     1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000], # 200
    [0.970, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
     1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000], # 300
])

def hall_efficiency(d1, d2):
    """Collection efficiency calculated using the Hall table.

    Arguments:
    d1, d2 - Particle diameters in meters.

    Returned value is collection efficiency interpolated from the table from
    Hall (1980). Linear interpolation (over collector radius and radius ratio)
    is used for all values within the table.

    For collector drops of radius greater than 300 microns, the returned value
    is always 1, in accordance with Hall (1980).  For collector drops of radius
    less than 10 microns, this function behaves as if the drops had a size of 10
    microns, except that the radius ratio is calculated using the true drop
    size. For radius ratios lower than 0.05, this function behaves as if the
    ratio was 0.05.
    """
    collector_d = max(d1, d2)
    # Always return unity for the largest drops.
    if collector_d > 600.e-6:
        return 1.
    r_ratio = min(d1, d2) / collector_d
    collector_r = 1.e6 * collector_d / 2.
    # Table has values for every 1/20 increment of r_ratio, so find nearest
    # value above this one.
    r_ratio_by_20 = 20. * r_ratio - 1.
    ratio_idx = min(int(r_ratio_by_20), 18)
    ratio_weight = max(r_ratio_by_20, 0.) - ratio_idx
    low_r_idx = 0
    low_r = _hall_radii[0]
    for i, r in enumerate(_hall_radii[:-1]):
        if collector_r <= r:
            break
        low_r_idx = i
        low_r = r
    r_weight = (max(collector_r, 10.) - low_r) / (_hall_radii[low_r_idx+1] - low_r)
    return (1.-r_weight) * ((1. - ratio_weight) * _hall_efficiencies[low_r_idx, ratio_idx]
                            + ratio_weight * _hall_efficiencies[low_r_idx, ratio_idx+1]) \
              + r_weight * ((1. - ratio_weight) * _hall_efficiencies[low_r_idx+1, ratio_idx]
                            + ratio_weight * _hall_efficiencies[low_r_idx+1, ratio_idx+1])


class BoundType(enum.Flag):
    """Flags for boundaries used for definite double integrals.

    In this module, there are a number of integrals over the l_x and l_y
    coordinates, which are the natural logarithms of mass coordinates x and y,
    respectively. Integration regions are decomposed into pieces where the l_x
    (outer) integral has constant bounds, and boundaries over the l_y coordinate
    are either constants, or have the form:

        log(e^{c} - e^{l_x})

    for some constant c.

    This enumeration specifies the form of the bounds for such an integral, and
    can take on four values:

        CONSTANT: Both l_y bounds are constant.
        LOWER_VARIES: The lower bound varies and the upper bound is constant.
        UPPER_VARIES: The upper bound varies and the lower bound is constant.
        BOTH_VARY: Both lower and upper bounds vary.

    The lower_varies and upper_varies methods provide a more legible way of
    querying whether each bound varies.
    """
    CONSTANT = 0
    LOWER_VARIES = enum.auto()
    UPPER_VARIES = enum.auto()
    BOTH_VARY = LOWER_VARIES | UPPER_VARIES

    def lower_varies(self):
        """Query whether the lower boundary varies for this BoundType."""
        return bool(self & BoundType.LOWER_VARIES)

    def upper_varies(self):
        """Query whether the upper boundary varies for this BoundType."""
        return bool(self & BoundType.UPPER_VARIES)


def find_corners(ly_bound, lz_bound):
    """Returns lx-coordinates of four corners of an integration region.

    Arguments:
    ly_bound - Lower and upper bounds of y bin.
    lz_bound - Lower and upper bounds of z bin.

    If we are calculating the transfer of mass from bin x to bin z through
    collisions with bin y, then we require add_logs(lx, ly) to be in the
    proper lz range. For a given y bin and z bin, the values of lx and ly
    that satisfy this condition form a sort of warped quadrilateral. This
    function returns the lx coordinates of that quadrilateral's corners, in
    the order:

        (bottom_left, top_left, bottom_right, top_right)

    This is all true if the y bin is strictly less than the z bin, but if
    the y bin and z bin overlap, then some of these corners will go to
    infinity/fail to exist, in which case `None` is returned:
     - If `ly_bound[1] >= lz_bound[0]`, then `top_left` is `None`.
     - If `ly_bound[1] >= lz_bound[1]`, then `top_right` is also `None`.
     - If `ly_bound[0] >= lz_bound[0]`, then `bottom_left` is `None`.
     - If `ly_bound[0] >= lz_bound[1]`, i.e. the entire y bin is above the z
       bin, then all returned values are `None`.
    """
    if not ly_bound[1] > ly_bound[0]:
        raise ValueError("upper y bin limit not larger than"
                         " lower y bin limit")
    if not lz_bound[1] > lz_bound[0]:
        raise ValueError("upper z bin limit not larger than"
                         " lower z bin limit")
    if lz_bound[0] > ly_bound[0]:
        bottom_left = sub_logs(lz_bound[0], ly_bound[0])
    else:
        bottom_left = None
    if lz_bound[0] > ly_bound[1]:
        top_left = sub_logs(lz_bound[0], ly_bound[1])
    else:
        top_left = None
    if lz_bound[1] > ly_bound[0]:
        bottom_right = sub_logs(lz_bound[1], ly_bound[0])
    else:
        bottom_right = None
    if lz_bound[1] > ly_bound[1]:
        top_right = sub_logs(lz_bound[1], ly_bound[1])
    else:
        top_right = None
    return (bottom_left, top_left, bottom_right, top_right)

def min_max_ly(lx_bound, y_bound_p, btype):
    """Find the bounds of y values for a particular integral.

    Arguments:
    lx_bound - Bounds for l_x.
    y_bound_p - Bound parameters for l_y.
    btype - Boundary type for l_y integrals.

    If a particular bound is an l_y value according to the btype, then the
    corresponding y_bound_p entry is returned for that bound. If the bound
    is an l_z value, the corresponding l_y min or max is returned.
    """
    if btype.lower_varies():
        min_ly = sub_logs(y_bound_p[0], lx_bound[1])
    else:
        min_ly = y_bound_p[0]
    if btype.upper_varies():
        max_ly = sub_logs(y_bound_p[1], lx_bound[0])
    else:
        max_ly = y_bound_p[1]
    return (min_ly, max_ly)

def get_lxs_and_btypes(lx_bound, ly_bound, lz_bound):
    """Find the bin x bounds and integration types for a bin set.

    Arguments:
    lx_bound - Bounds for x bin (source bin).
    ly_bound - Bounds for y bin (colliding bin).
    lz_bound - Bounds for z bin (destination bin).

    Returns a tuple `(lxs, btypes)`, where lxs is a list of integration
    bounds of length 2 to 4, and btypes is a list of boundary types of size
    one less than lxs.

    If the integration region is of size zero, lists of size zero are
    returned.
    """
    if not lx_bound[1] > lx_bound[0]:
        raise ValueError("upper x bin limit not larger than"
                         " lower x bin limit")
    (bl, tl, br, tr) = find_corners(ly_bound, lz_bound)
    # Cases where there is no region of integration.
    if (br is None) or (tl is not None and lx_bound[1] <= tl) \
       or (lx_bound[0] >= br):
        return [], []
    # Figure out whether bl or tr is smaller. If both are `None` (i.e. they
    # are at -Infinity), it doesn't matter, as the logic below will use the
    # lx1 to remove this part of the list.
    if bl is None or (tr is not None and bl <= tr):
        lxs = [tl, bl, tr, br]
        btypes = [BoundType.LOWER_VARIES,
                  BoundType.CONSTANT,
                  BoundType.UPPER_VARIES]
    else:
        lxs = [tl, tr, bl, br]
        btypes = [BoundType.LOWER_VARIES,
                  BoundType.BOTH_VARY,
                  BoundType.UPPER_VARIES]
    if lxs[0] is None or lx_bound[0] > lxs[0]:
        for _ in range(len(btypes)):
            if lxs[1] is None or lx_bound[0] > lxs[1]:
                del lxs[0]
                del btypes[0]
            else:
                lxs[0] = lx_bound[0]
                break
    if lx_bound[1] < lxs[-1]:
        for _ in range(len(btypes)):
            if lx_bound[1] >= lxs[-2]:
                lxs[-1] = lx_bound[1]
                break
            del lxs[-1]
            del btypes[-1]
    return lxs, btypes

def get_y_bound_p(ly_bound, lz_bound, btypes):
    """Find y bound parameters from y and z bin bounds and btypes.

    Arguments:
    ly_bound - Lower and upper bounds of y bin.
    lz_bound - Lower and upper bounds of z bin.
    btypes - List of boundary types for l_y integrals.

    Returns an array of shape `(len(btypes), 2)`, which contains the lower
    and upper y bound parameters for each btype in the list.
    """
    y_bound_p = np.zeros((len(btypes), 2))
    for i, btype in enumerate(btypes):
        if btype.lower_varies():
            y_bound_p[i,0] = lz_bound[0]
        else:
            y_bound_p[i,0] = ly_bound[0]
        if btype.upper_varies():
            y_bound_p[i,1] = lz_bound[1]
        else:
            y_bound_p[i,1] = ly_bound[1]
    return y_bound_p


class CollisionKernel(ABC):
    """
    Represent a collision kernel for the bin model.
    """

    collision_kernel_type_str_len = 32
    """Length of collision kernel type string written to file."""

    @abstractmethod
    def integrate_over_bins(self, basis_x, basis_y, lz_bound):
        """Integrate kernel over a relevant domain given x, y, and z bins.

        Arguments:
        basis_x - Basis function for x bin (source bin).
        basis_y - Basis function for y bin (colliding bin).
        lz_bound - Bounds for z bin (destination bin).

        This returns the value of the mass-weighted kernel integrated over the
        region where the values of `(lx, ly)` are in the given bins, and where
        collisions produce masses in the z bin.
        """

    @classmethod
    @abstractmethod
    def construct_sparsity_structure(cls, basis, grid, boundary=None):
        """Find the sparsity structure of a tensor discretizing this kernel.

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

        If `boundary == 'closed'`, this behavior is modified so that there is no
        bin stretching to infinity, the excessive mass is assumed to remain in
        the largest finite bin, and the maximum values of idxs and idxs + nums -
        1 are actually num_bins - 1.

        For most mass grids used for collision-coalescence, note that the
        entries of nums are all 1 or 2, so max_num is 2.
        """

    @abstractmethod
    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""

    @classmethod
    def from_netcdf(cls, netcdf_file, constants):
        """Retrieve a CollisionKernel object from a NetcdfFile."""
        ckern_type = netcdf_file.read_characters('collision_kernel_type')
        if ckern_type == 'Long':
            kc = netcdf_file.read_scalar('kc')
            kr = netcdf_file.read_scalar('kr')
            rain_m = netcdf_file.read_scalar('rain_m')
            return LongKernel(constants, kc=kc, kr=kr, rain_m=rain_m)
        if ckern_type == 'Hall':
            efficiency_name = netcdf_file.read_characters('efficiency_name')
            return HallKernel(constants, efficiency_name)
        raise RuntimeError("unrecognized collision_kernel_type in file")


class CoalescenceKernel(CollisionKernel):
    """
    Represent a collision kernel for the bin model.
    """

    @classmethod
    def construct_sparsity_structure(cls, basis, grid, boundary=None):
        """Find the sparsity structure of a tensor discretizing this kernel.

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

        If `boundary == 'closed'`, this behavior is modified so that there is no
        bin stretching to infinity, the excessive mass is assumed to remain in
        the largest finite bin, and the maximum values of idxs and idxs + nums -
        1 are actually num_bins - 1.

        For most mass grids used for collision-coalescence, note that the
        entries of nums are all 1 or 2, so max_num is 2.
        """
        if boundary is None:
            boundary = 'open'
        if boundary not in ('open', 'closed'):
            raise ValueError("invalid boundary specified: " + str(boundary))
        nbasis = basis.size
        idxs = np.zeros((nbasis, nbasis), dtype=np.int_)
        nums = np.zeros((nbasis, nbasis), dtype=np.int_)
        for i in range(nbasis):
            for j in range(nbasis):
                idxs[i,j], nums[i,j] = grid.find_sum_bins(
                    basis[i].lower_bound, basis[i].upper_bound,
                    basis[j].lower_bound, basis[j].upper_bound,
                )
        nb = grid.num_bins
        if boundary == 'closed':
            for i in range(nbasis):
                for j in range(nbasis):
                    if idxs[i,j] == nb:
                        idxs[i,j] = nb - 1
                    elif idxs[i,j] + nums[i,j] - 1 == nb:
                        nums[i,j] -= 1
        max_num = nums.max()
        return idxs, nums, max_num

    @abstractmethod
    def kernel_x(self, x, y):
        """Calculate kernel function as a function of log scaled mass."""

    def kernel_integral(self, basis_x, basis_y, lx_bound, y_bound_p, btype):
        r"""Computes an integral necessary for constructing the collision tensor.

        Arguments:
        basis_x, basis_y - The x and y basis functions to integrate over.
        lx_bound - Bounds for l_x. Abbreviated below as lxm, lxp.
        y_bound_p - Bound parameters for l_y. Abbreviated below as (a, b).
        btype - Boundary type for y integrals.

        If K_f is the scaled kernel function, this returns:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} e^{l_x} K_f(l_x, l_y)
          basis_x(l_x) basis_y(l_y) dl_y dl_x
        """
        tol = 1.e-12
        # For efficiency and stability, refuse to bother with extremely
        # small ranges of particle sizes.
        if lx_bound[1] - lx_bound[0] < tol:
            return 0.
        def f(ly, lx):
            x = np.exp(lx)
            y = np.exp(ly)
            return basis_x(lx) * basis_y(ly) * self.kernel_x(x, y) / y
        if btype.lower_varies():
            def g(lx):
                return sub_logs(y_bound_p[0], lx)
        else:
            g = y_bound_p[0]
        if btype.upper_varies():
            def h(lx):
                return sub_logs(y_bound_p[1], lx)
        else:
            h = y_bound_p[1]
        y, _ = dblquad(f, lx_bound[0], lx_bound[1], g, h)
        return y

    def integrate_over_bins(self, basis_x, basis_y, lz_bound):
        """Integrate kernel over a relevant domain given x, y, and z bins.

        Arguments:
        basis_x - Basis function for x bin (source bin).
        basis_y - Basis function for y bin (colliding bin).
        lz_bound - Bounds for z bin (destination bin).

        This returns the value of the mass-weighted kernel integrated over the
        region where the values of `(lx, ly)` are in the given bins, and where
        collisions produce masses in the z bin.
        """
        lx_bound = (basis_x.lower_bound, basis_x.upper_bound)
        ly_bound = (basis_y.lower_bound, basis_y.upper_bound)
        lxs, btypes = get_lxs_and_btypes(lx_bound, ly_bound, lz_bound)
        y_bound_p = get_y_bound_p(ly_bound, lz_bound, btypes)
        output = 0.
        for i, btype in enumerate(btypes):
            output += self.kernel_integral(basis_x, basis_y, lxs[i:i+2],
                                           y_bound_p[i,:], btype)
        return output


class LongKernel(CoalescenceKernel):
    """
    Implement the Long (1974) collision-coalescence kernel.

    This is a simple piecewise polynomial kernel, with separate formulas for
    cloud (diameter below d_thresh) and rain (diameter above d_thresh).
    Specifically, for masses x and y, the cloud formula is kc * (x**2 + y**2),
    and the rain formula is kr * (x + y).

    The default parameters are exactly those mentioned in the original paper,
    but these can be overridden on initialization. To avoid ambiguity, kc_cgs
    and kc_si cannot both be specified, and the same goes for kr_cgs and kr_si.

    Initialization arguments:
    constants - ModelConstants object for the model.
    kc_cgs (optional) - Cloud kernel parameter in cgs (cm^3/g^2/s) units.
    kc_si (optional) - Cloud kernel parameter in SI (m^3/kg^2/s) units.
                       Effectively a synonym for kc_cgs.
    kr_cgs (optional) - Rain kernel parameter in cgs (cm^3/g/s) units.
    kr_si (optional) - Rain kernel parameter in SI (m^3/kg/s) units.
    kc (optional) - Semi-nondimensionalized cloud kernel parameter (m^3/s).
    kr (optional) - Semi-nondimensionalized rain kernel parameter (m^3/s).
    rain_m (optional) - Mass used to transition between cloud and rain
                        formulae.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(self, constants, kc=None, kr=None, kc_cgs=None, kc_si=None,
                 kr_cgs=None, kr_si=None, rain_m=None):
        kc_arg_count = [kc is None, kc_cgs is None, kc_si is None].count(False)
        if kc_arg_count > 1:
            raise RuntimeError("tried to specify multiple kc values")
        kr_arg_count = [kr is None, kr_cgs is None, kr_si is None].count(False)
        if kr_arg_count > 1:
            raise RuntimeError("tried to specify multiple kr values")
        if kc_cgs is None:
            kc_cgs = 9.44e9
        if kc_si is None:
            kc_si = kc_cgs
        if kc is None:
            kc = kc_si * constants.std_mass**2
        self.kc = kc
        if kr_cgs is None:
            kr_cgs = 5.78e3
        if kr_si is None:
            kr_si = kr_cgs * 1.e-3
        if kr is None:
            kr = kr_si * constants.std_mass
        self.kr = kr
        if rain_m is None:
            rain_m = constants.diameter_to_scaled_mass(1.e-4)
        self.rain_m = rain_m
        self.log_rain_m = np.log(rain_m)

    def kernel_x(self, x, y):
        """Calculate kernel function as a function of log scaled mass."""
        if x < self.rain_m and y < self.rain_m:
            return self.kc * (x**2 + y**2)
        else:
            return self.kr * (x + y)

    def _integral_cloud(self, lx_bound, y_bound_p, btype):
        r"""Computes integral part of the kernel for cloud-sized particles.

        Arguments:
        lx_bound - Bounds for l_x. Abbreviated below as lxm, lxp.
        y_bound_p - Bound parameters for l_y. Abbreviated below as (a, b).
        btype - Boundary type for l_y integrals.

        The definite integral being computed is:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} (e^{2l_x-l_y} + e^{l_y}) dl_y dl_x
        """
        etoa = np.exp(y_bound_p[0])
        etob = np.exp(y_bound_p[1])
        etolxm = np.exp(lx_bound[0])
        etolxp = np.exp(lx_bound[1])
        lx_width = lx_bound[1] - lx_bound[0]
        if btype.lower_varies():
            lower = etoa * np.log((etoa - etolxp)/(etoa - etolxm)) \
                    + etoa * lx_width
        else:
            lower = etoa * lx_width - 0.5 * (etolxp**2 - etolxm**2) / etoa
        if btype.upper_varies():
            upper = etob * np.log((etob - etolxp)/(etob - etolxm)) \
                    + etob * lx_width
        else:
            upper = etob * lx_width - 0.5 * (etolxp**2 - etolxm**2) / etob
        return self.kc * (upper - lower)

    def _integral_rain(self, lx_bound, y_bound_p, btype):
        r"""Computes integral part of the kernel for rain-sized particles.

        Arguments:
        lx_bound - Bounds for l_x. Abbreviated below as lxm, lxp.
        y_bound_p - Bound parameters for l_y. Abbreviated below as (a, b).
        btype - Boundary type for l_y integrals.

        The definite integral being computed is:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} (e^{l_x-l_y} + 1) dl_y dl_x
        """
        etoa = np.exp(y_bound_p[0])
        etob = np.exp(y_bound_p[1])
        etolxm = np.exp(lx_bound[0])
        etolxp = np.exp(lx_bound[1])
        lx_width = lx_bound[1] - lx_bound[0]
        if btype.lower_varies():
            lower = y_bound_p[0] * lx_width \
                + np.log((etoa - etolxp)/(etoa - etolxm)) \
                - dilogarithm(etolxp / etoa) + dilogarithm(etolxm / etoa)
        else:
            lower = y_bound_p[0] * lx_width - (etolxp - etolxm) / etoa
        if btype.upper_varies():
            upper = y_bound_p[1] * lx_width \
                + np.log((etob - etolxp)/(etob - etolxm)) \
                - dilogarithm(etolxp / etob) + dilogarithm(etolxm / etob)
        else:
            upper = y_bound_p[1] * lx_width - (etolxp - etolxm) / etob
        return self.kr * (upper - lower)

    def kernel_integral(self, basis_x, basis_y, lx_bound, y_bound_p, btype):
        r"""Computes an integral necessary for constructing the collision tensor.

        Arguments:
        basis_x, basis_y - The x and y basis functions to integrate over.
        lx_bound - Bounds for l_x. Abbreviated below as lxm, lxp.
        y_bound_p - Bound parameters for l_y. Abbreviated below as (a, b).
        btype - Boundary type for y integrals.

        If K_f is the scaled kernel function, this returns:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} e^{l_x} K_f(l_x, l_y)
          (l_x-lx0)**deg_lx (l_y-ly0)**deg_ly dl_y dl_x
        """
        if basis_x.degree == 0 and basis_y.degree == 0:
            return self.kernel_integral_deg0(lx_bound, y_bound_p, btype)
        return super().kernel_integral(basis_x, basis_y, lx_bound, y_bound_p, btype)

    def kernel_integral_deg0(self, lx_bound, y_bound_p, btype):
        r"""Computes an integral necessary for constructing the collision tensor.

        Arguments:
        lx_bound - Bounds for l_x. Abbreviated below as lxm, lxp.
        y_bound_p - Bound parameters for l_y. Abbreviated below as (a, b).
        btype - Boundary type for y integrals.

        If K_f is the scaled kernel function, this returns:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} e^{l_x} K_f(l_x, l_y) dl_y dl_x
        """
        # Note that this function also checks that the btype is in bounds.
        min_ly, max_ly = min_max_ly(lx_bound, y_bound_p, btype)
        # Fuzz factor allowing a bin to be considered pure cloud/rain even if
        # there is a tiny overlap with the other category.
        tol = 1.e-10
        # Check for at least one pure rain bin.
        if min_ly + tol >= self.log_rain_m \
           or lx_bound[0] + tol >= self.log_rain_m:
            return self._integral_rain(lx_bound, y_bound_p, btype)
        # Check for both pure cloud bins.
        if max_ly - tol <= self.log_rain_m \
           and lx_bound[1] - tol <= self.log_rain_m:
            return self._integral_cloud(lx_bound, y_bound_p, btype)
        # Handle if x bin has both rain and cloud with recursive call.
        if lx_bound[0] + tol < self.log_rain_m < lx_bound[1] - tol:
            cloud_part = self._integral_cloud((lx_bound[0], self.log_rain_m),
                                              y_bound_p, btype)
            rain_part = self._integral_rain((self.log_rain_m, lx_bound[1]),
                                            y_bound_p, btype)
            return cloud_part + rain_part
        # At this point, it is guaranteed that the y bin spans both categories
        # while the x bin does not.
        # Handle any part of the x range that uses rain formula only.
        if btype.lower_varies() \
           and y_bound_p[0] > add_logs(lx_bound[0], self.log_rain_m):
            lx_low = sub_logs(y_bound_p[0], self.log_rain_m)
            start = self._integral_rain((lx_bound[0], lx_low), y_bound_p,
                                        btype=btype)
        else:
            lx_low = lx_bound[0]
            start = 0.
        # Handle any part of the x range that uses cloud formula only.
        if btype.upper_varies() \
           and y_bound_p[1] < add_logs(lx_bound[1], self.log_rain_m):
            lx_high = sub_logs(y_bound_p[1], self.log_rain_m)
            start += self._integral_cloud((lx_high, lx_bound[1]), y_bound_p,
                                          btype=btype)
        else:
            lx_high = lx_bound[1]
        cloud_part = self._integral_cloud((lx_low, lx_high),
                                          (y_bound_p[0], self.log_rain_m),
                                          btype=(btype
                                                 & BoundType.LOWER_VARIES))
        rain_part = self._integral_rain((lx_low, lx_high),
                                        (self.log_rain_m, y_bound_p[1]),
                                        btype=(btype
                                               & BoundType.UPPER_VARIES))
        return start + cloud_part + rain_part

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_dimension('collision_kernel_type_str_len',
                                    self.collision_kernel_type_str_len)
        netcdf_file.write_characters('collision_kernel_type',
                                     'Long',
                                     'collision_kernel_type_str_len',
                                     'Type of collision kernel')
        netcdf_file.write_scalar('kc', self.kc,
            'f8', 'm^3/s',
            "Semi-nondimensionalized Long kernel cloud parameter")
        netcdf_file.write_scalar('kr', self.kr,
            'f8', 'm^3/s',
            "Semi-nondimensionalized Long kernel rain parameter")
        netcdf_file.write_scalar('rain_m', self.rain_m,
            'f8', 'kg',
            "Cloud-rain threshold mass")


def make_golovin_kernel(constants, b=6.e3):
    """Create a Golovin kernel.

    Arguments:
    b (optional) - The constant scaling factor to multiply the kernel by.
                   Defaults to 6000 sec^-1.

    This is implemented by returning a LongKernel with the rain threshold
    diameter set extremely low (to 10^-3 microns).
    """
    return LongKernel(constants, rain_m=1.e-30, kr_si=b)


class HallKernel(CoalescenceKernel):
    """
    Implement Hall-like collision-coalescence kernel.

    Initialization arguments:
    constants - ModelConstants object for the model.
    efficiency_name - Name of collection efficiency formula to use.
                      Can only be 'ScottChen'.
    """

    efficiency_name_len = 32
    """Maximum length of collection efficiency formula name."""

    def __init__(self, constants, efficiency_name='Hall'):
        self.constants = constants
        self.efficiency_name = efficiency_name
        if efficiency_name == 'ScottChen':
            self.efficiency = sc_efficiency
        elif efficiency_name == 'Hall':
            self.efficiency = hall_efficiency
        else:
            raise ValueError("bad value for efficiency_name: "
                             + efficiency_name)

    def kernel_d(self, d1, d2):
        """Calculate kernel function as a function of particle diameters."""
        const = self.constants
        v_diff = np.abs(beard_v(const, d1) - beard_v(const, d2))
        eff = self.efficiency(d1, d2)
        return 0.25 * np.pi * (d1 + d2)**2 * eff * v_diff

    def kernel_x(self, x, y):
        """Calculate kernel function as a function of log scaled mass."""
        const = self.constants
        d1 = const.scaled_mass_to_diameter(x)
        d2 = const.scaled_mass_to_diameter(y)
        return self.kernel_d(d1, d2)

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_dimension('collision_kernel_type_str_len',
                                    self.collision_kernel_type_str_len)
        netcdf_file.write_characters('collision_kernel_type',
                                     'Hall',
                                     'collision_kernel_type_str_len',
                                     'Type of collision kernel')
        netcdf_file.write_dimension('efficiency_name_len',
                                    self.efficiency_name_len)
        netcdf_file.write_characters('efficiency_name',
                                     self.efficiency_name,
                                     'efficiency_name_len',
                                     'Collection efficiency formula name')
