"""Types related to a simple single-moment 1D microphysics scheme.

Classes:
LongKernel
Kernel

Utility functions:
add_logs
sub_logs
dilogarithm
"""

import numpy as np
from scipy.special import spence

def add_logs(x, y):
    """Returns log(exp(x)+exp(y))."""
    return x + np.log(1. + np.exp(y-x))

def sub_logs(x, y):
    """Returns log(exp(x)-exp(y))."""
    assert y < x, "y >= x in sub_logs"
    return x + np.log(1. - np.exp(y-x))

def dilogarithm(x):
    """Returns the dilogarithm of the input."""
    return spence(1. - x)

class ModelConstants:
    """
    Define relevant constants and scalings for the model.

    Parameters:
    rho_water - Density of water (kg/m^3).
    rho_air - Density of air (kg/m^3).
    std_diameter - Diameter (m) of a particle of "typical" size, used
                   internally to non-dimensionalize particle sizes.
    rain_d - Diameter (m) of threshold diameter defining the distinction
             between cloud and rain particles.

    Attributes:
    std_mass - Mass in kg corresponding to a scaled mass of 1.

    Methods:
    diameter_to_scaled_mass - Convert diameter in meters to scaled mass.
    scaled_mass_to_diameter - Convert scaled mass to diameter in meters.
    """

    def __init__(self, rho_water, rho_air, std_diameter, rain_d):
        self.rho_water = rho_water
        self.rho_air = rho_air
        self.std_diameter = std_diameter
        self.std_mass = rho_water * np.pi/6. * std_diameter**3
        self.rain_d = rain_d
        self.rain_m = self.diameter_to_scaled_mass(rain_d)

    def diameter_to_scaled_mass(self, d):
        """Convert diameter in meters to non-dimensionalized particle size."""
        return (d / self.std_diameter)**3

    def scaled_mass_to_diameter(self, x):
        """Convert non-dimensionalized particle size to diameter in meters."""
        return self.std_diameter * x**(1./3.)


class Kernel:
    """
    Represent a collision kernel for the bin model.

    Methods:
    integrate_over_bins
    kernel_integral

    Utility methods:
    find_corners
    min_max_ly
    get_lxs_and_btypes
    get_ly_bounds
    """

    def find_corners(self, ly1, ly2, lz1, lz2):
        """Returns lx-coordinates of four corners of an integration region.

        Arguments:
        ly1, ly2 - Lower and upper bounds of y bin.
        lz1, lz2 - Lower and upper bounds of z bin.

        If we are calculating the transfer of mass from bin x to bin z through
        collisions with bin y, then we require add_logs(lx, ly) to be in the
        proper lz range. For a given y bin and z bin, the values of lx and ly
        that satisfy this condition form a sort of warped quadrilateral. This
        function returns the lx coordinates of that quadrilateral's corners, in
        the order:
        
            (bottom_left, top_left, bottom_right, top_right)

        This is all true if `ly2 < lz1`, but if the y bin and z bin overlap,
        then some of these corners will go to infinity/fail to exist, in which
        case `None` is returned:
         - If `ly2 >= lz1`, then `top_left` is `None`.
         - If `ly2 >= lz2`, then `top_right` is also `None`.
         - If `ly1 >= lz1`, then `bottom_left` is `None`.
         - If `ly1 >= lz2`, i.e. the entire y bin is above the z bin, then all
           returned values are `None`.
        """
        assert ly2 > ly1, "upper y bin limit not larger than lower y bin limit"
        assert lz2 > lz1, "upper z bin limit not larger than lower z bin limit"
        if lz1 > ly1:
            bottom_left = sub_logs(lz1, ly1)
        else:
            bottom_left = None
        if lz1 > ly2:
            top_left = sub_logs(lz1, ly2)
        else:
            top_left = None
        if lz2 > ly1:
            bottom_right = sub_logs(lz2, ly1)
        else:
            bottom_right = None
        if lz2 > ly2:
            top_right = sub_logs(lz2, ly2)
        else:
            top_right = None
        return (bottom_left, top_left, bottom_right, top_right)

    def min_max_ly(self, a, b, lxm, lxp, btype):
        """Find the bounds of y values for a particular integral.

        Arguments:
        a - Lower bound parameter for l_y.
        b - Upper bound parameter for l_y.
        lxm - Lower bound for l_x.
        lxp - Upper bound for l_x.
        btype - Boundary type for l_y integrals.

        If a particular bound is an l_y value according to the btype, then the
        corresponding value a or b is returned for that bound. If the bound is
        an l_z value, the corresponding l_y min or max is returned.
        """
        assert 0 <= btype < 4, "invalid btype in min_max_ly"
        if btype % 2 == 0:
            min_ly = a
        else:
            min_ly = sub_logs(a, lxp)
        if btype < 2:
            max_ly = b
        else:
            max_ly = sub_logs(b, lxm)
        return (min_ly, max_ly)

    def get_lxs_and_btypes(self, lx1, lx2, ly1, ly2, lz1, lz2):
        """Find the bin x bounds and integration types for a bin set.

        Arguments:
        lx1, lx2 - Bounds for x bin (source bin).
        ly1, ly2 - Bounds for y bin (colliding bin).
        lz1, lz2 - Bounds for z bin (destination bin).

        Returns a tuple `(lxs, btypes)`, where lxs is a list of integration
        bounds of length 2 to 4, and btypes is a list of boundary types of size
        one less than lxs.

        If the integration region is of size zero, lists of size zero are
        returned.
        """
        assert lx2 > lx1, "upper x bin limit not larger than lower x bin limit"
        (bl, tl, br, tr) = self.find_corners(ly1, ly2, lz1, lz2)
        # Cases where there is no region of integration.
        if (br is None) or (tl is not None and lx2 <= tl) or (lx1 >= br):
            return [], []
        # Figure out whether bl or tr is smaller. If both are `None` (i.e. they
        # are at -Infinity), it doesn't matter, as the logic below will use the
        # lx1 to remove this part of the list.
        if bl is None or (tr is not None and bl <= tr):
            lxs = [tl, bl, tr, br]
            btypes = [1, 0, 2]
        else:
            lxs = [tl, tr, bl, br]
            btypes = [1, 3, 2]
        if lxs[0] is None or lx1 > lxs[0]:
            for i in range(len(btypes)):
                if lxs[1] is None or lx1 > lxs[1]:
                    del lxs[0]
                    del btypes[0]
                else:
                    lxs[0] = lx1
                    break
        if lx2 < lxs[-1]:
            for i in range(len(btypes)):
                if lx2 >= lxs[-2]:
                    lxs[-1] = lx2
                    break
                else:
                    del lxs[-1]
                    del btypes[-1]
        return lxs, btypes

    def get_ly_bounds(self, ly1, ly2, lz1, lz2, btypes):
        """Find a and b bound parameters from y and z bin bounds and btypes.

        Arguments:
        ly1, ly2 - Lower and upper bounds of y bin.
        lz1, lz2 - Lower and upper bounds of z bin.
        btypes - List of boundary types for l_y integrals.

        Returns a tuple `(avals, bvals)`, where avals is a list of lower bound
        parameters and bvals is a list of upper bound parameters.
        """
        avals = []
        bvals = []
        for btype in btypes:
            if btype % 2 == 0:
                avals.append(ly1)
            else:
                avals.append(lz1)
            if btype < 2:
                bvals.append(ly2)
            else:
                bvals.append(lz2)
        return avals, bvals

    def kernel_integral(self, a, b, lxm, lxp, btype):
        """Computes an integral necessary for constructing the kernel tensor.

        If K_f is the scaled kernel function, this returns:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} e^{l_x} K_f(l_x, l_y) dl_y dl_x

        Arguments:
        a - Lower bound parameter for l_y.
        b - Upper bound parameter for l_y.
        lxm - Lower bound for l_x.
        lxp - Upper bound for l_x.
        btype - Boundary type for y integrals.

        Boundary functions can be either constant or of the form
            p(c) = log(e^{c} - e^{l_x})
        This is determined by btype:
         - If btype = 0, g(a) = a and h(b) = b.
         - If btype = 1, g(a) = p(a) and h(b) = b.
         - If btype = 2, g(a) = a and h(b) = p(b).
         - If btype = 3, g(a) = p(a) and h(b) = p(b).

        This docstring is attached to an unimplemented function on the base
        Kernel class. Child classes should override this with an actual
        implementation.
        """
        raise NotImplementedError

    def integrate_over_bins(self, lx1, lx2, ly1, ly2, lz1, lz2):
        """Integrate kernel over a relevant domain given x, y, and z bins.

        Arguments:
        lx1, lx2 - Bounds for x bin (source bin).
        ly1, ly2 - Bounds for y bin (colliding bin).
        lz1, lz2 - Bounds for z bin (destination bin).

        This returns the value of the mass-weighted kernel integrated over the
        region where the values of `(lx, ly)` are in the given bins, and where
        collisions produce masses in the z bin.
        """
        lxs, btypes = self.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        avals, bvals = self.get_ly_bounds(ly1, ly2, lz1, lz2, btypes)
        output = 0.
        for i in range(len(btypes)):
            output += self.kernel_integral(avals[i], bvals[i],
                                           lxs[i], lxs[i+1], btypes[i])
        return output


class LongKernel(Kernel):
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
    """

    def __init__(self, constants, kc_cgs=None, kc_si=None, kr_cgs=None,
                 kr_si=None, rain_m=None):
        assert (kc_cgs is None) or (kc_si is None)
        assert (kr_cgs is None) or (kr_si is None)
        if kc_cgs is None:
            kc_cgs = 9.44e9
        if kc_si is None:
            kc_si = kc_cgs
        self.kc = kc_si * constants.rho_air * constants.std_mass**2
        if kr_cgs is None:
            kr_cgs = 5.78e3
        if kr_si is None:
            kr_si = kr_cgs * 1.e-3
        self.kr = kr_si * constants.rho_air * constants.std_mass
        if rain_m is None:
            self.log_rain_m = np.log(constants.rain_m)
        else:
            self.log_rain_m = np.log(rain_m)

    def _integral_cloud(self, a, b, lxm, lxp, btype):
        """Computes integral part of the kernel for cloud-sized particles.

        The definite integral being computed is:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} (e^{2l_x-l_y} + e^{l_y}) dl_y dl_x

        Arguments:
        a - Lower bound parameter for l_y.
        b - Upper bound parameter for l_y.
        lxm - Lower bound for l_x.
        lxp - Upper bound for l_x.
        btype - Boundary type for l_y integrals.

        Boundary functions can be either constant or of the form
            p(c) = log(e^{c} - e^{l_x})
        This is determined by btype:
         - If btype = 0, g(a) = a and h(b) = b.
         - If btype = 1, g(a) = p(a) and h(b) = b.
         - If btype = 2, g(a) = a and h(b) = p(b).
         - If btype = 3, g(a) = p(a) and h(b) = p(b).
        """
        assert 0 <= btype < 4, "invalid btype in _integral_cloud"
        etoa = np.exp(a)
        etob = np.exp(b)
        etolxm = np.exp(lxm)
        etolxp = np.exp(lxp)
        if btype < 2:
            upper = etob * (lxp-lxm) - 0.5 * (etolxp**2 - etolxm**2) / etob
        else:
            upper = etob * np.log((etob - etolxp)/(etob - etolxm)) \
                    + etob * (lxp-lxm)
        if btype % 2 == 0:
            lower = etoa * (lxp-lxm) - 0.5 * (etolxp**2 - etolxm**2) / etoa
        else:
            lower = etoa * np.log((etoa - etolxp)/(etoa - etolxm)) \
                    + etoa * (lxp-lxm)
        return upper - lower

    def _integral_rain(self, a, b, lxm, lxp, btype):
        """Computes integral part of the kernel for rain-sized particles.

        The definite integral being computed is:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} (e^{l_x-l_y} + 1) dl_y dl_x

        Arguments:
        a - Lower bound parameter for l_y.
        b - Upper bound parameter for l_y.
        lxm - Lower bound for l_x.
        lxp - Upper bound for l_x.
        btype - Boundary type for y integrals.

        Boundary functions can be either constant or of the form
            p(c) = log(e^{c} - e^{l_x})
        This is determined by btype:
         - If btype = 0, g(a) = a and h(b) = b.
         - If btype = 1, g(a) = p(a) and h(b) = b.
         - If btype = 2, g(a) = a and h(b) = p(b).
         - If btype = 3, g(a) = p(a) and h(b) = p(b).
        """
        assert 0 <= btype < 4, "invalid btype in _integral_rain"
        etoa = np.exp(a)
        etob = np.exp(b)
        etolxm = np.exp(lxm)
        etolxp = np.exp(lxp)
        if btype < 2:
            upper = b * (lxp-lxm) - (etolxp - etolxm) / etob
        else:
            upper = b * (lxp-lxm) + np.log((etob - etolxp)/(etob - etolxm)) \
                    - dilogarithm(etolxp / etob) + dilogarithm(etolxm / etob)
        if btype % 2 == 0:
            lower = a * (lxp-lxm) - (etolxp - etolxm) / etoa
        else:
            lower = a * (lxp-lxm) + np.log((etoa - etolxp)/(etoa - etolxm)) \
                    - dilogarithm(etolxp / etoa) + dilogarithm(etolxm / etoa)
        return upper - lower

    def kernel_integral(self, a, b, lxm, lxp, btype):
        """Computes an integral necessary for constructing the kernel tensor.

        If K_f is the scaled kernel function, this returns:

        \int_{lxm}^{lxp} \int_{g(a)}^{h(b)} e^{l_x} K_f(l_x, l_y) dl_y dl_x

        Arguments:
        a - Lower bound parameter for l_y.
        b - Upper bound parameter for l_y.
        lxm - Lower bound for l_x.
        lxp - Upper bound for l_x.
        btype - Boundary type for y integrals.

        Boundary functions can be either constant or of the form
            p(c) = log(e^{c} - e^{l_x})
        This is determined by btype:
         - If btype = 0, g(a) = a and h(b) = b.
         - If btype = 1, g(a) = p(a) and h(b) = b.
         - If btype = 2, g(a) = a and h(b) = p(b).
         - If btype = 3, g(a) = p(a) and h(b) = p(b).
        """
        min_ly, max_ly = self.min_max_ly(a, b, lxm, lxp, btype)
        # Fuzz factor allowing a bin to be considered pure cloud/rain even if
        # there is a tiny overlap with the other category.
        tol = 1.e-10
        # Check for at least one pure rain bin.
        if min_ly + tol >= self.log_rain_m or lxm + tol >= self.log_rain_m:
            return self.kr * self._integral_rain(a, b, lxm, lxp, btype)
        # Check for both pure cloud bins.
        elif max_ly - tol <= self.log_rain_m and lxp - tol <= self.log_rain_m:
            return self.kc * self._integral_cloud(a, b, lxm, lxp, btype)
        # Handle if x bin has both rain and cloud with recursive call.
        if lxm + tol < self.log_rain_m < lxp - tol:
            cloud_part = self.kernel_integral(a, b, lxm, self.log_rain_m,
                                              btype)
            rain_part = self.kernel_integral(a, b, self.log_rain_m, lxp,
                                             btype)
            return cloud_part + rain_part
        # At this point, it is guaranteed that the y bin spans both categories
        # while the x bin does not. Handle this with recursive call.
        if btype == 0:
            # Can simply split up the parts in this case.
            cloud_part = self.kernel_integral(a, self.log_rain_m, lxm, lxp,
                                              btype=0)
            rain_part = self.kernel_integral(self.log_rain_m, b, lxm, lxp,
                                             btype=0)
            return cloud_part + rain_part
        if btype == 1:
            # Handle any part of the x range that uses rain formula only.
            if a > add_logs(lxm, self.log_rain_m):
                lx_low = sub_logs(a, self.log_rain_m)
                start = self.kernel_integral(a, b, lxm, lx_low, btype=1)
            else:
                lx_low = lxm
                start = 0.
            cloud_part = self.kernel_integral(a, self.log_rain_m, lx_low, lxp,
                                              btype=1)
            rain_part = self.kernel_integral(self.log_rain_m, b, lx_low, lxp,
                                             btype=0)
            return start + cloud_part + rain_part
        if btype == 2:
            # Handle any part of the x range that uses cloud formula only.
            if b < add_logs(lxp, self.log_rain_m):
                lx_high = sub_logs(b, self.log_rain_m)
                start = self.kernel_integral(a, b, lx_high, lxp, btype=2)
            else:
                lx_high = lxp
                start = 0.
            cloud_part = self.kernel_integral(a, self.log_rain_m, lxm, lx_high,
                                              btype=0)
            rain_part = self.kernel_integral(self.log_rain_m, b, lxm, lx_high,
                                             btype=2)
            return start + cloud_part + rain_part
        if btype == 3:
            # Handle any part of the x range that uses rain formula only.
            if a > add_logs(lxm, self.log_rain_m):
                lx_low = sub_logs(a, self.log_rain_m)
                start = self.kernel_integral(a, b, lxm, lx_low, btype=3)
            else:
                lx_low = lxm
                start = 0.
            # Handle any part of the x range that uses cloud formula only.
            if b < add_logs(lxp, self.log_rain_m):
                lx_high = sub_logs(b, self.log_rain_m)
                start += self.kernel_integral(a, b, lx_high, lxp, btype=3)
            else:
                lx_high = lxp
            cloud_part = self.kernel_integral(a, self.log_rain_m,
                                              lx_low, lx_high, btype=1)
            rain_part = self.kernel_integral(self.log_rain_m, b,
                                             lx_low, lx_high, btype=2)
            return start + cloud_part + rain_part


class Grid:
    """
    Bin model grid.
    """
