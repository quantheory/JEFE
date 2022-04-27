"""Types related to a simple single-moment 1D microphysics scheme.

Classes:
Kernel
LongKernel
MassGrid
GeometricMassGrid
KernelTensor

Utility functions:
add_logs
sub_logs
dilogarithm
lower_gamma_deriv
gamma_dist_d
gamma_dist_lam_deriv
gamma_dist_nu_deriv
"""

import numpy as np
from scipy.special import spence, gamma, gammainc, digamma

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

def lower_gamma_deriv(s, x, atol=1.e-14, rtol=1.e-14):
    """Derivative of lower incomplete gamma function with respect to s.

    Arguments:
    s - Shape parameter
    x - Integral bound
    atol (optional) - Absolute error tolerance (defaults to 1.e-14).
    rtol (optional) - Relative error tolerance (defaults to 1.e-14).

    This function finds the derivative of the lower incomplete gamma function
    with respect to its shape parameter, using the series representation given
    by Sun and Qin (2017). It attempts to produce a result y such that the
    error is approximately rtol*y + atol or less.

    This function is only valid for real x > 0 and real s. x and s may be
    numpy arrays of equal shape, or one may be an array and the other a scalar.
    """
    if isinstance(x, np.ndarray):
        out = np.zeros(x.shape)
        if isinstance(s, np.ndarray):
            assert x.shape == s.shape, "shapes of x and s do not match"
            for i in range(len(out.flat)):
                out.flat[i] = lower_gamma_deriv(s.flat[i], x.flat[i],
                                                atol=atol, rtol=rtol)
        else:
            for i in range(len(out.flat)):
                out.flat[i] = lower_gamma_deriv(s, x.flat[i], atol, rtol)
        return out
    if isinstance(s, np.ndarray):
        out = np.zeros(s.shape)
        for i in range(len(out.flat)):
            out.flat[i] = lower_gamma_deriv(s.flat[i], x, atol, rtol)
        return out
    assert x > 0., "x must be positive"
    assert atol > 0., "atol must be positive"
    assert rtol > 0., "rtol must be positive"
    l = np.log(x)
    if s > 2.:
        out = gammainc(s-1., x) * gamma(s-1.)
        out -= l * x**(s-1.) * np.exp(-x)
        out += (s-1.)*lower_gamma_deriv(s-1., x)
        return out
    if x < 2.:
        y = 0.
        k = 0
        sign = 1.
        mag = x**s
        recip_s_plus_k = 1./(s + k)
        err_lim = 1.e100
        while x > k or err_lim > atol + abs(y)*rtol:
            term = sign * mag * recip_s_plus_k * (l - recip_s_plus_k)
            y += term
            # Set up for next iteration.
            k += 1
            sign *= -1.
            mag *= x / k
            recip_s_plus_k = 1./(s + k)
            err_lim = mag * recip_s_plus_k * max(l, recip_s_plus_k)
        return y
    else:
        dgamma = digamma(s) * gamma(s)
        # Numerators and denominators in sequence of continued fraction
        # convergents, with their derivatives.
        num_vec = np.zeros((4,))
        den_vec = np.zeros((4,))
        # First sequence entries are 0/1 and (x^s e^-x)/(x-s+1).
        den_vec[0] = 1.
        y_prev = dgamma
        num_vec[2] = x**s * np.exp(-x)
        num_vec[3] = l * num_vec[2]
        den_vec[2] = x - s + 1.
        den_vec[3] = -1.
        y = dgamma - (den_vec[2] * num_vec[3] - num_vec[2] * den_vec[3]) \
                / den_vec[2]**2
        # Recursion matrix
        Y = np.zeros((4, 4))
        Y[0, 2] = 1.
        Y[1, 3] = 1.
        k = 0
        while x > k or abs(y - y_prev) > atol + abs(y)*rtol:
            k = k + 1
            y_prev = y
            Y[2,0] = k * (s - k)
            Y[2,2] = 2*k + 1. + x - s
            Y[3,0] = k
            Y[3,1] = Y[2,0]
            Y[3,2] = -1.
            Y[3,3] = Y[2,2]
            num_vec = Y @ num_vec
            den_vec = Y @ den_vec
            if np.any(num_vec > 1.e100) or np.any(den_vec > 1.e100):
                num_vec /= 1.e100
                den_vec /= 1.e100
            y = dgamma - num_vec[3]/den_vec[2] \
                    + (num_vec[2]/den_vec[2]) * (den_vec[3]/den_vec[2])
        return y

def gamma_dist_d(grid, lam, nu):
    """Convert a gamma distribution over diameter to a mass-weighted DSD.

    Arguments:
    grid - The MassGrid used to discretize the distribution.
    lam - The gamma distribution scale parameter.
    nu - The gamma distribution shape parameter.

    The returned value is a simple 1-D Python array with length grid.num_bins.
    Normalization is such that, over an infinite grid, integrating the result
    (i.e. taking the dot product with grid.bin_widths) would yield 1. However,
    if much of the gamma distribution mass is outside of the given grid, the
    result would be somewhat less, so the user should check this.
    """
    gamma_integrals = gammainc(nu+3., lam*grid.bin_bounds_d)
    return (gamma_integrals[1:] - gamma_integrals[:-1]) / grid.bin_widths

def gamma_dist_d_lam_deriv(grid, lam, nu):
    """Derivative of gamma_dist_d with respect to lam."""
    bbd = grid.bin_bounds_d
    bin_func = bbd * (lam*bbd)**(nu+2) * np.exp(-lam*bbd) / gamma(nu+3)
    return (bin_func[1:] - bin_func[:-1]) / grid.bin_widths

def gamma_dist_d_nu_deriv(grid, lam, nu):
    """Derivative of gamma_dist_d with respect to nu."""
    bbd = grid.bin_bounds_d
    gnu3 = gamma(nu+3.)
    bin_func = lower_gamma_deriv(nu+3., lam*grid.bin_bounds_d) / gnu3
    bin_func -= gammainc(nu+3., lam*grid.bin_bounds_d) * digamma(nu+3.)
    return (bin_func[1:] - bin_func[:-1]) / grid.bin_widths

class ModelConstants:
    """
    Define relevant constants and scalings for the model.

    Initialization arguments:
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
            rain_m = constants.diameter_to_scaled_mass(1.e-4)
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


class MassGrid:
    """
    Represent the bin model's mass grid.

    Attributes:
    d_min, d_max - Minimum/maximum particle diameter (m).
    x_min, x_max - Minimum/maximum particle size (scaled mass units).
    lx_min, lx_max - Natural ogarithm of x_min/x_max, for convenience.
    num_bins - Number of model bins.
    bin_bounds - Array of size num_bins+1 containing edges of bins.
                 This array is in units of log(scaled mass), i.e. the
                 first value is lx_min and last value is lx_max.
    bin_bounds_d - Same as bin_bounds, but for diameters of particles at the
                   bin edges (m).
    bin_widths - Array of size num_bins containing widths of each bin
                 in log-units, i.e. `sum(bin_widths) == lx_max-lx_min`.

    Methods:
    find_bin
    get_sum_bins
    """
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
        """Find the sparsity structure of a kernel tensor using this grid.

        Arguments:
        boundary (optional) - Either 'open' or 'closed'. Default is 'open'.

        We represent the kernel as a tensor indexed by three bins:

         1. The bin acting as a source of mass (labeled the "x" bin).
         2. A bin colliding with the source bin (the "y" bin).
         3. The destination bin that mass is added to (the "z" bin).

        For a given x and y bin, not every particle size can be produced; only
        a small range of z bins will have nonzero kernel tensor. To represent
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
        assert boundary in ('open', 'closed'), \
            "invalid boundary specified: " + str(boundary)
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
        self.num_bins = num_bins
        self.x_min = constants.diameter_to_scaled_mass(d_min)
        self.x_max = constants.diameter_to_scaled_mass(d_max)
        self.lx_min = np.log(self.x_min)
        self.lx_max = np.log(self.x_max)
        self.dlx = (self.lx_max - self.lx_min) / self.num_bins
        self.bin_bounds = np.linspace(self.lx_min, self.lx_max, num_bins+1)
        bin_bounds_m = np.exp(self.bin_bounds)
        self.bin_bounds_d = constants.scaled_mass_to_diameter(bin_bounds_m)
        self.bin_widths = self.bin_bounds[1:] - self.bin_bounds[:-1]


class KernelTensor():
    """
    Represent a collision kernel evaluated on a particular mass grid.

    Initialization arguments:
    kernel - A Kernel object representing the collision kernel.
    grid - A MassGrid object defining the bins.
    scaling (optional) - If present, the kernel is divided by this scaling
                         factor.
    boundary (optional) - Upper boundary condition. If 'open', then particles
                          that are created larger than the largest bin size
                          "fall out" of the box. If 'closed', these particles
                          are placed in the largest bin. Defaults to 'open'.

    Attributes:
    grid - Stored reference to corresponding grid.
    scaling - Effect of the kernel has been scaled down by this amount.
    boundary - Upper boundary condition for this kernel.
    idxs, nums, max_num - Outputs of `MassGrid.construct_sparsity_structure`.
    data - Data corresponding to nonzero elements of the tensor kernel.
           This is represented by an array of shape:
               (num_bins, num_bins, max_num)
    """
    def __init__(self, kernel, grid, scaling=None, boundary=None):
        self.grid = grid
        if scaling is None:
            scaling = 1.
        self.scaling = scaling
        if boundary is None:
            boundary = 'open'
        self.boundary = boundary
        idxs, nums, max_num = \
            grid.construct_sparsity_structure(boundary=boundary)
        self.idxs = idxs
        self.nums = nums
        self.max_num = max_num
        nb = grid.num_bins
        bb = grid.bin_bounds
        self.data = np.zeros((nb, nb, max_num))
        if boundary == 'closed':
            high_bin = nb - 1
        else:
            high_bin = nb
        for i in range(nb):
            for j in range(nb):
                idx = idxs[i,j]
                for k in range(nums[i,j]):
                    zidx = idx + k
                    if zidx == high_bin:
                        top_bound = np.inf
                    else:
                        top_bound = bb[zidx+1]
                    self.data[i,j,k] = kernel.integrate_over_bins(
                        bb[i], bb[i+1], bb[j], bb[j+1], bb[zidx], top_bound)
        self.data /= scaling

    def calc_rate(self, f, out_flux=None, derivative=False):
        """Calculate rate of change of f due to collision-coalescence.

        Arguments:
        f - Representation of DSD on this grid.
        out_flux (optional) - Whether to force output of mass leaving the box.

        ``f'' must be an array of total size num_bins or num_bins+1, either a
        1-D array, a row vector, or column vector. If of size num_bins+1, the
        last bin is assumed to hold the mass that has been removed from the
        grid cell, which is ignored.

        If out_flux is not specified, the output is an array with the same
        shape as f, containing the rate of change of f over time due to
        collision-coalescence. If f is of size num_bins+1, the final element of
        the output is the amount of mass that leaves the box due to collision-
        coalescence (i.e. due to becoming too large).

        If out_flux is specified and True, then the output is the same as if f
        had been of shape (nb+1,), and if it is False, then the output is the
        same as if f was of shape (nb,).
        """
        nb = self.grid.num_bins
        f_len = len(f.flat)
        assert nb <= f_len < nb + 2, "invalid f length: "+str(f_len)
        if out_flux is None:
            out_flux = f_len == nb + 1
            out_len = f_len
            out_shape = f.shape
        else:
            out_len = nb + 1 if out_flux else nb
            out_shape = (out_len,)
        f = np.reshape(f, (f_len, 1))
        f_outer = np.dot(f, np.transpose(f))
        output = np.zeros((out_len,))
        if derivative:
            rate_deriv = np.zeros((out_len, out_len))
        for i in range(nb):
            for j in range(nb):
                idx = self.idxs[i,j]
                fprod = f_outer[i,j]
                for k in range(self.nums[i,j]):
                    dfdt_term = fprod * self.data[i,j,k]
                    output[i] -= dfdt_term
                    if out_flux or idx+k < nb:
                        output[idx+k] += dfdt_term
                    if derivative:
                        deriv_i = self.data[i,j,k] * f[j]
                        deriv_j = self.data[i,j,k] * f[i]
                        rate_deriv[i,i] -= deriv_i
                        rate_deriv[i,j] -= deriv_j
                        if out_flux or idx+k < nb:
                            rate_deriv[idx+k,i] += deriv_i
                            rate_deriv[idx+k,j] += deriv_j
        output[:nb] /= self.grid.bin_widths
        output = np.reshape(output, out_shape)
        if derivative:
            for i in range(out_len):
                rate_deriv[:nb,i] /= self.grid.bin_widths
            return output, rate_deriv
        else:
            return output
