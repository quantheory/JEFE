"""Types related to a simple single-moment 1D microphysics scheme.

Utility functions:
add_logs
sub_logs
dilogarithm
lower_gamma_deriv
gamma_dist_d
gamma_dist_lam_deriv
gamma_dist_nu_deriv

Classes:
Kernel
LongKernel
MassGrid
GeometricMassGrid
KernelTensor
ModelStateDescriptor
ModelState
RK45Integrator
LogTransform
"""

import numpy as np
import scipy.linalg as la
from scipy.special import spence, gamma, gammainc, digamma
from scipy.integrate import solve_ivp

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
    with respect to its shape parameter. This is done with a series if x is
    small, and using Legendre's continued fraction when x is large. It attempts
    to produce a result y such that the error is approximately rtol*y + atol or
    less.

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
    mass_conc_scale (optional) - Mass concentration scale used for
                                 nondimensionalization.
    time_scale (optional) - Time scale used for nondimensionalization.

    Attributes:
    rho_water
    rho_air
    std_diameter
    std_mass - Mass in kg corresponding to a scaled mass of 1.
    rain_d
    rain_m - `rain_d` converted to scaled mass.
    mass_conc_scale
    time_scale

    Methods:
    diameter_to_scaled_mass - Convert diameter in meters to scaled mass.
    scaled_mass_to_diameter - Convert scaled mass to diameter in meters.
    """

    def __init__(self, rho_water, rho_air, std_diameter, rain_d,
                 mass_conc_scale=None, time_scale=None):
        if mass_conc_scale is None:
            mass_conc_scale = 1.
        if time_scale is None:
            time_scale = 1.
        self.rho_water = rho_water
        self.rho_air = rho_air
        self.std_diameter = std_diameter
        self.std_mass = rho_water * np.pi/6. * std_diameter**3
        self.rain_d = rain_d
        self.rain_m = self.diameter_to_scaled_mass(rain_d)
        self.mass_conc_scale = mass_conc_scale
        self.time_scale = time_scale

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
        self.kc = kc_si * constants.std_mass**2
        if kr_cgs is None:
            kr_cgs = 5.78e3
        if kr_si is None:
            kr_si = kr_cgs * 1.e-3
        self.kr = kr_si * constants.std_mass
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

    def moment_weight_vector(self, n, cloud_only=None, rain_only=None):
        """Calculate weight vector corresponding to a moment of the DSD.

        Arguments:
        n - Moment to calculate (can be any real number).
        cloud_only (optional) - Only count cloud-sized drops.
        rain_only (optional) - Only count rain-sized drops.

        The returned value is a vector such that
            std_diameter**n * np.dot(weight_vector, dsd) / self.std_mass
        is a moment of the DSD, or if the DSD is in dimensionless units,
            np.dot(weight_vector, dsd)
        is the dimensionless DSD.
        """
        if cloud_only is None:
            cloud_only is False
        if rain_only is None:
            rain_only is False
        assert not (cloud_only and rain_only), \
            "moments cannot be both cloud-only and rain-only"
        const = self.constants
        nb = self.num_bins
        bb = self.bin_bounds
        bw = self.bin_widths
        if cloud_only or rain_only:
            log_thresh = np.log(const.rain_m)
            thresh_idx = self.find_bin(log_thresh)
            if 0 <= thresh_idx < nb:
                thresh_frac = (log_thresh - bb[thresh_idx]) / bw[thresh_idx]
            elif thresh_idx < 0:
                thresh_idx = 0
                thresh_frac = 0.
            else:
                thresh_idx = nb-1
                thresh_frac = 1.
        if n == 3:
            weight_vector = bw.copy()
            if cloud_only:
                weight_vector[thresh_idx+1:] = 0.
                weight_vector[thresh_idx] *= thresh_frac
            elif rain_only:
                weight_vector[:thresh_idx] = 0.
                weight_vector[thresh_idx] *= 1. - thresh_frac
        else:
            exponent = n / 3. - 1.
            weight_vector = np.exp(exponent * bb) / exponent
            weight_vector = weight_vector[1:] - weight_vector[:-1]
            if cloud_only:
                weight_vector[thresh_idx+1:] = 0.
                weight_vector[thresh_idx] = \
                    np.exp(exponent * log_thresh) / exponent \
                    - np.exp(exponent * bb[thresh_idx]) / exponent
            elif rain_only:
                weight_vector[:thresh_idx] = 0.
                weight_vector[thresh_idx] = \
                    np.exp(exponent * bb[thresh_idx+1]) / exponent \
                    - np.exp(exponent * log_thresh) / exponent
        return weight_vector


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
        self.constants = constants
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

    Methods:
    calc_rate
    """
    def __init__(self, kernel, grid, boundary=None):
        self.grid = grid
        self.const = grid.constants
        self.scaling = self.const.std_mass \
            / (self.const.mass_conc_scale * self.const.time_scale)
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
        self.data /= self.scaling

    def calc_rate(self, f, out_flux=None, derivative=False):
        """Calculate rate of change of f due to collision-coalescence.

        Arguments:
        f - Representation of DSD on this grid.
        out_flux (optional) - Whether to force output of mass leaving the box.
        derivative (optional) - Whether to return the Jacobian of the rate
                                calculation as well.

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

        If derivative is True, the return value is a tuple, where the first
        element is the output described above, and the second element is a
        square matrix with the same size (on each size) as the first output.
        This matrix contains the Jacobian of this output with respect to the
        DSD (+ fallout if included in the output).
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


class ModelStateDescriptor:
    """
    Describe the state variables contained in a ModelState.

    Initialization arguments:
    constants - A ModelConstants object.
    mass_grid - A MassGrid object defining the bins.
    fallout (optional) - Whether to include a variable for mass that has fallen
                         out of the model box. Defaults to True.
    dsd_deriv_names (optional) - List of strings naming variables with respect
                                 to which DSD derivatives are prognosed.
    dsd_deriv_scales (optional) - List of scales for the derivatives of the
                                  named variable. These scales will be applied
                                  in addition to dsd_scale. They default to 1.

    Attributes:
    constants - ModelConstants object used by this model.
    mass_grid - The grid of the DSD used by this model.
    dsd_scale - Factor used internally to scale the DSD.
    dsd_deriv_num - Number of variables with respect to which the derivative of
                    the DSD is tracked.
    dsd_deriv_names - Names of variables with tracked derivatives.
    dsd_deriv_scales - Scales of variables with tracked derivatives.
                       These scales are applied on top of the dsd_scale.
    perturbed_variables - A list of tuples, with each tuple containing a weight
        vector, a transform, and a scale, in that order.
    perturbation_rate - A covariance matrix representing the error introduced
        to the perturbed variables per second.

    Methods:
    state_len
    construct_raw
    dsd_loc
    fallout_loc
    dsd_deriv_loc
    fallout_deriv_loc
    perturb_cov_loc
    dsd_raw
    fallout_raw
    dsd_deriv_raw
    fallout_deriv_raw
    perturb_cov_raw
    """
    def __init__(self, constants, mass_grid, dsd_scale=None,
                 dsd_deriv_names=None, dsd_deriv_scales=None,
                 perturbed_variables=None, perturbation_rate=None,
                 correction_time=None):
        self.constants = constants
        self.mass_grid = mass_grid
        self.dsd_scale = constants.mass_conc_scale
        if dsd_deriv_names is not None:
            self.dsd_deriv_num = len(dsd_deriv_names)
            assert len(set(dsd_deriv_names)) == self.dsd_deriv_num, \
                "duplicate derivatives found in list"
            self.dsd_deriv_names = dsd_deriv_names
            if dsd_deriv_scales is None:
                dsd_deriv_scales = np.ones((self.dsd_deriv_num,))
            assert len(dsd_deriv_scales) == self.dsd_deriv_num, \
                "dsd_deriv_scales length does not match dsd_deriv_names"
            self.dsd_deriv_scales = dsd_deriv_scales
        else:
            assert dsd_deriv_scales is None, \
                "cannot specify dsd_deriv_scales without dsd_deriv_names"
            self.dsd_deriv_num = 0
            self.dsd_deriv_names = []
            self.dsd_deriv_scales = np.zeros((0,))
        if perturbed_variables is not None:
            pn = len(perturbed_variables)
            nb = mass_grid.num_bins
            self.perturb_num = pn
            self.perturb_wvs = np.zeros((pn, nb))
            for i in range(pn):
                self.perturb_wvs[i,:] = perturbed_variables[i][0]
            self.perturb_transforms = [t[1] for t in perturbed_variables]
            self.perturb_scales = np.array([t[2] for t in perturbed_variables])
            self.perturbation_rate = np.zeros((pn, pn))
            if perturbation_rate is not None:
                assert perturbation_rate.shape == (pn, pn), \
                    "perturbation_rate is the wrong shape, should be " \
                    + str((pn, pn))
                for i in range(pn):
                    for j in range(pn):
                        self.perturbation_rate[i,j] = perturbation_rate[i,j] \
                            / perturbed_variables[i][2] \
                            / perturbed_variables[j][2] * constants.time_scale
            if correction_time is not None:
                self.correction_time = correction_time / constants.time_scale
            else:
                assert pn == self.dsd_deriv_num + 1, \
                    "must specify correction time unless perturb_num is " \
                    "equal to dsd_deriv_num+1"
                self.correction_time = None
        else:
            assert perturbation_rate is None, \
                "cannot specify perturbation_rate without perturbed_variables"
            assert correction_time is None, \
                "cannot specify correction_time without perturbed_variables"
            self.perturb_num = 0

    def state_len(self):
        """Return the length of the state vector."""
        idx, num = self.perturb_cov_loc()
        return idx + num

    def construct_raw(self, dsd, fallout=None, dsd_deriv=None,
                      fallout_deriv=None, perturb_cov=None):
        """Construct raw state vector from individual variables.

        Arguments:
        dsd - Drop size distribution.
        fallout (optional) - Amount of third moment that has fallen out of the
                             model. If not specified, defaults to zero.
        dsd_deriv - DSD derivatives. Mandatory if dsd_deriv_num is not zero.
        fallout_deriv - Fallout derivatives. If not specified, defaults to
                        zero.
        perturb_cov - Covariance matrix for Gaussian perturbation. If not
                      specified, defaults to zero.

        Returns a 1-D array of size given by state_len().
        """
        raw = np.zeros((self.state_len(),))
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        pn = self.perturb_num
        assert len(dsd) == nb, "dsd of wrong size for this descriptor's grid"
        idx, num = self.dsd_loc()
        raw[idx:idx+num] = dsd / self.dsd_scale
        if fallout is None:
            fallout = 0.
        raw[self.fallout_loc()] = fallout / self.dsd_scale
        if ddn > 0:
            assert dsd_deriv is not None, \
                "dsd_deriv input is required, but missing"
            assert (dsd_deriv.shape == (ddn, nb)), \
                "dsd_deriv input is the wrong shape"
            if fallout_deriv is None:
                fallout_deriv = np.zeros((ddn,))
            assert len(fallout_deriv) == ddn, \
                "fallout_deriv input is wrong length"
            for i in range(ddn):
                idx, num = self.dsd_deriv_loc(self.dsd_deriv_names[i])
                raw[idx:idx+num] = dsd_deriv[i,:] / self.dsd_deriv_scales[i] \
                                    / self.dsd_scale
                idx = self.fallout_deriv_loc(self.dsd_deriv_names[i])
                raw[idx] = fallout_deriv[i] / self.dsd_deriv_scales[i] \
                                    / self.dsd_scale
        else:
            assert dsd_deriv is None or len(dsd_deriv.flat) == 0, \
                "no dsd derivatives should be specified for this descriptor"
            assert fallout_deriv is None or len(fallout_deriv) == 0, \
                "no fallout derivatives should be specified " \
                "for this descriptor"
        if pn > 0:
            if perturb_cov is not None:
                assert (perturb_cov.shape == (pn, pn)), \
                    "perturb_cov input is the wrong shape"
                perturb_cov = perturb_cov.copy()
                for i in range(pn):
                    for j in range(pn):
                        perturb_cov[i,j] /= \
                            self.perturb_scales[i] * self.perturb_scales[j]
                idx, num = self.perturb_cov_loc()
                raw[idx:idx+num] = np.reshape(perturb_cov, (num,))
        else:
            assert perturb_cov is None, \
                "no perturbation covariance should be specified " \
                "for this descriptor"
        return raw

    def dsd_loc(self, with_fallout=None):
        """Return location of the DSD data in the state vector.

        Arguments:
        with_fallout (optional) - Include fallout at the end of DSD data.
                                  Defaults to False.

        Returns a tuple (idx, num), where idx is the location of the first DSD
        entry and num is the number of entries.
        """
        if with_fallout is None:
            with_fallout = False
        add = 1 if with_fallout else 0
        return (0, self.mass_grid.num_bins + add)

    def fallout_loc(self):
        """Return index of fallout scalar in the state vector."""
        idx, num = self.dsd_loc()
        return idx + num

    def dsd_deriv_loc(self, var_name=None, with_fallout=None):
        """Return location of DSD derivative data in the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.
        with_fallout (optional) - Include fallout derivative at the end of DSD
                                  derivative data. Defaults to False.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.

        Returns a tuple (idx, num), where idx is the location of the first
        entry and num is the number of entries. If all derivative information
        is returned, idx is a list of integers, while num is a scalar that is
        the size of each contiguous block (since all will be the same size).
        """
        if with_fallout is None:
            with_fallout = False
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        st_idx, st_num = self.dsd_loc(with_fallout=True)
        start = st_idx + st_num
        num = nb+1 if with_fallout else nb
        if ddn == 0:
            return [start], 0
        if var_name is None:
            return [start + i*(nb+1) for i in range(ddn)], num
        else:
            idx = self.dsd_deriv_names.index(var_name)
            return start + idx*(nb+1), num

    def fallout_deriv_loc(self, var_name=None):
        """Return location of fallout derivative data in the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.
        """
        idx, num = self.dsd_deriv_loc(var_name, with_fallout=False)
        if var_name is None:
            return [i+num for i in idx]
        else:
            return idx+num

    def perturb_cov_loc(self):
        """Return location of perturbation covariance matrix.

        Returns a tuple (idx, num), where idx is the location of the first
        element and num is the number of elements.
        """
        idx, num = self.dsd_deriv_loc(with_fallout=True)
        pn = self.perturb_num
        return idx[-1] + num, pn*pn

    def dsd_raw(self, raw, with_fallout=None):
        """Return raw DSD data from the state vector.

        Arguments:
        raw - Raw state vector.
        with_fallout (optional) - Include fallout at the end of DSD data, if
                                  present. Defaults to False.
        """
        idx, num = self.dsd_loc(with_fallout)
        return raw[idx:idx+num]

    def fallout_raw(self, raw):
        """Return raw fallout data from the state vector."""
        return raw[self.fallout_loc()]

    def dsd_deriv_raw(self, raw, var_name=None, with_fallout=None):
        """Return raw DSD derivative data from the state vector.

        Arguments:
        raw - Raw state vector.
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 2D array of size
        `(dsd_deriv_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        idx, num = self.dsd_deriv_loc(var_name, with_fallout)
        if var_name is None:
            output = np.zeros((ddn, num))
            for i in range(ddn):
                output[i,:] = raw[idx[i]:idx[i]+num]
            return output
        else:
            return raw[idx:idx+num]

    def fallout_deriv_raw(self, raw, var_name=None):
        """Return raw fallout derivative data from the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.
        """
        ddn = self.dsd_deriv_num
        idx = self.fallout_deriv_loc(var_name)
        if var_name is None:
            output = np.zeros((ddn,))
            for i in range(ddn):
                output[i] = raw[idx[i]]
            return output
        else:
            return raw[idx]

    def perturb_cov_raw(self, raw):
        """Return raw perturbation covariance matrix from the state vector."""
        idx, num = self.perturb_cov_loc()
        pn = self.perturb_num
        return np.reshape(raw[idx:idx+num], (pn, pn))


class ModelState:
    """
    Describe a state of the model at a moment in time.

    Initialization arguments:
    desc - The ModelStateDescriptor object corresponding to this object.
    dsd - A 1-D vector representing the mass-weighted drop size distribution.
          This should define the amount of 3rd moment with respect to diameter
          in each bin, using units consistent with the constants object.

    Attributes:
    constants - ModelConstants object used by this model.
    mass_grid - The grid of the DSD used by this model.
    desc - ModelStateDescriptor associated with this state.
    raw - A single large vector used to handle the state (mainly for use in
          time integration).

    Methods:
    dsd
    dsd_moment
    fallout
    dsd_deriv
    fallout_deriv
    perturb_cov
    dsd_time_deriv_raw
    time_derivative_raw
    linear_func_raw
    linear_func_rate_raw
    rain_prod_breakdown
    """
    def __init__(self, desc, raw):
        self.constants = desc.constants
        self.mass_grid = desc.mass_grid
        self.desc = desc
        self.raw = raw

    def dsd(self):
        """Droplet size distribution associated with this state.

        The units are those of M3 for the distribution."""
        return self.desc.dsd_raw(self.raw) * self.desc.dsd_scale

    def dsd_moment(self, n, cloud_only=None, rain_only=None):
        """Calculate a moment of the DSD.

        Arguments:
        n - Moment to calculate (can be any real number).
        cloud_only (optional) - Only count cloud-sized drops.
        rain_only (optional) - Only count rain-sized drops.
        """
        const = self.constants
        grid = self.mass_grid
        m3_dsd = self.dsd() / const.std_mass
        weight_vector = grid.moment_weight_vector(n, cloud_only, rain_only)
        return const.std_diameter**(n) * np.dot(weight_vector, m3_dsd)

    def fallout(self):
        """Return amount of third moment that has fallen out of the model."""
        return self.desc.fallout_raw(self.raw) * self.desc.dsd_scale

    def dsd_deriv(self, var_name=None):
        """Return derivatives of the DSD with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 2D array of size
        `(dsd_deriv_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        dsd_deriv = self.desc.dsd_deriv_raw(self.raw, var_name).copy()
        if var_name is None:
            for i in range(self.desc.dsd_deriv_num):
                dsd_deriv[i,:] *= self.desc.dsd_deriv_scales[i]
        else:
            idx = self.desc.dsd_deriv_names.index(var_name)
            dsd_deriv *= self.desc.dsd_deriv_scales[idx]
        dsd_deriv *= self.desc.dsd_scale
        return dsd_deriv

    def fallout_deriv(self, var_name=None):
        """Return derivative of fallout with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 1D array of length dsd_deriv_num is
        returned, with all derivatives in it. If var_name is provided, a
        single scalar is returned for that derivative.
        """
        if var_name is None:
            output = self.desc.fallout_deriv_raw(self.raw)
            for i in range(self.desc.dsd_deriv_num):
                output[i] *= self.desc.dsd_deriv_scales[i]
            return output * self.desc.dsd_scale
        else:
            idx = self.desc.dsd_deriv_names.index(var_name)
            return self.desc.fallout_deriv_raw(self.raw) \
                * self.desc.dsd_scale * self.desc.dsd_deriv_scales[idx]

    def perturb_cov(self):
        """Return perturbation covariance matrix."""
        output = self.desc.perturb_cov_raw(self.raw).copy()
        pn = self.desc.perturb_num
        pscales = self.desc.perturb_scales
        for i in range(pn):
            for j in range(pn):
                output[i,j] *= pscales[i] * pscales[j]
        return output

    def dsd_time_deriv_raw(self, proc_tens):
        """Time derivative of the raw dsd using the given process tensors.

        Arguments:
        proc_tens - List of process tensors, the rates of which sum to give the
                    time derivative.
        """
        dsd_raw = self.desc.dsd_raw(self.raw, with_fallout=True)
        dfdt = np.zeros(dsd_raw.shape)
        for pt in proc_tens:
            dfdt += pt.calc_rate(dsd_raw, out_flux=True)
        return dfdt

    def time_derivative_raw(self, proc_tens):
        """Time derivative of the state using the given process tensors.

        Arguments:
        proc_tens - List of process tensors, the rates of which sum to give the
                    time derivative.
        """
        desc = self.desc
        nb = self.mass_grid.num_bins
        ddn = desc.dsd_deriv_num
        pn = desc.perturb_num
        dfdt = np.zeros((len(self.raw),))
        dsd_raw = desc.dsd_raw(self.raw)
        dsd_deriv_raw = desc.dsd_deriv_raw(self.raw, with_fallout=True)
        didx, dnum = desc.dsd_loc(with_fallout=True)
        dridxs, drnum = desc.dsd_deriv_loc(with_fallout=True)
        if pn > 0:
            double_time_deriv = np.zeros((nb+1))
        for pt in proc_tens:
            if ddn > 0:
                rate, derivative = pt.calc_rate(dsd_raw, out_flux=True,
                                                derivative=True)
                dfdt[didx:didx+dnum] += rate
                for i in range(ddn):
                    dfdt[dridxs[i]:dridxs[i]+drnum] += \
                        derivative @ dsd_deriv_raw[i,:]
                if pn > 0:
                    double_time_deriv += derivative @ rate
            else:
                dfdt[didx:didx+dnum] += pt.calc_rate(dsd_raw, out_flux=True)
        if pn > 0:
            pcidx, pcnum = desc.perturb_cov_loc()
            ddsddt = desc.dsd_raw(dfdt)
            ddsddt_deriv = np.zeros((ddn+1,nb))
            ddsddt_deriv[0,:] = double_time_deriv[:nb]
            ddsddt_deriv[1:,:] = desc.dsd_deriv_raw(dfdt)
            perturb_cov_raw = desc.perturb_cov_raw(self.raw)
            lfs = np.zeros((pn,))
            lf_jac = np.zeros((pn, ddn+1))
            lf_rates = np.zeros((pn,))
            lf_rate_jac = np.zeros((pn, ddn+1))
            for i in range(pn):
                wv = self.desc.perturb_wvs[i]
                lfs[i], lf_jac[i,:] = \
                    self.linear_func_raw(wv, derivative=True,
                                         dfdt=ddsddt)
                lf_rates[i], lf_rate_jac[i,:] = \
                    self.linear_func_rate_raw(wv, ddsddt,
                                              dfdt_deriv=ddsddt_deriv)
            transform_mat = np.zeros((pn, pn))
            transform_mat2 = np.zeros((pn,))
            for i in range(pn):
                transform = desc.perturb_transforms[i]
                transform_mat[i,i] = transform.derivative(lfs[i])
                transform_mat2[i] = \
                    transform.second_over_first_derivative(lfs[i])
            zeta_to_v = transform_mat @ lf_jac
            jacobian = transform_mat @ lf_rate_jac @ la.pinv(zeta_to_v)
            jacobian += np.diag(lf_rates * transform_mat2)
            if self.desc.correction_time is None:
                perturb_cov_projected = perturb_cov_raw
            else:
                projection = la.inv(zeta_to_v.T @ desc.perturbation_rate
                                        @ zeta_to_v)
                projection = zeta_to_v @ projection @ zeta_to_v.T \
                                @ desc.perturbation_rate
                perturb_cov_projected = projection @ perturb_cov_raw \
                                            @ projection.T
            cov_rate = jacobian @ perturb_cov_projected
            cov_rate += cov_rate.T
            cov_rate += desc.perturbation_rate
            if self.desc.correction_time is not None:
                cov_rate += (perturb_cov_projected - perturb_cov_raw) \
                                / self.desc.correction_time
            dfdt[pcidx:pcidx+pcnum] = np.reshape(cov_rate, (pcnum,))
        return dfdt

    def linear_func_raw(self, weight_vector, derivative=None, dfdt=None):
        """Calculate a linear functional of the DSD (nondimensionalized units).

        Arguments:
        weight_vector - Weighting function defining the integral over the DSD.
                        (E.g. obtained from `MassGrid.moment_weight_vector`.)
        derivative (optional) - If True, return derivative information as well.
                                Defaults to False.
        dfdt (optional) - The raw derivative of the DSD with respect to time.
                          If included and derivative is True, the time
                          derivative of the functional will be prepended to the
                          derivative output.

        If derivative is False, this returns a scalar representing the
        nondimensional moment or other linear functional requested. If
        derivative is True, returns a tuple where the first element is the
        requested functional and the second element is the gradient of the
        functional with to the variables listed in self.desc.

        If dfdt is specified and derivative is True, the 0-th element of the
        gradient will be the derivative with respect to time.
        """
        if derivative is None:
            derivative = False
        dsd_raw = self.desc.dsd_raw(self.raw)
        nb = self.mass_grid.num_bins
        if derivative:
            ddn = self.desc.dsd_deriv_num
            offset = 1 if dfdt is not None else 0
            dsd_deriv_raw = np.zeros((ddn+offset, nb))
            if dfdt is not None:
                dsd_deriv_raw[0,:] = dfdt
            dsd_deriv_raw[offset:,:] = self.desc.dsd_deriv_raw(self.raw)
            grad = dsd_deriv_raw @ weight_vector
            return np.dot(dsd_raw, weight_vector), grad
        else:
            return np.dot(dsd_raw, weight_vector)

    def linear_func_rate_raw(self, weight_vector, dfdt, dfdt_deriv=None):
        """Calculate rate of change of a linear functional of the DSD.

        Arguments:
        weight_vector - Weighting function defining the integral over the DSD.
                        (E.g. obtained from `MassGrid.moment_weight_vector`.)
        dfdt - The raw derivative of the DSD with respect to time.
        dfdt_deriv (optional) - If not None, return derivative information as
                                well. Defaults to None.

        If dfdt_deriv is None, this returns a scalar representing the time
        derivative of the nondimensional moment or other linear functional
        requested. If dfdt_deriv contains DSD derivative information, returns a
        tuple where the first element is the requested time derivative and the
        second element is the gradient of the derivative with respect to the
        variables for which the DSD derivative is provided.

        If dfdt_deriv is not None, it should be an array of shape
            (ddn, num_bins)
        where ddn is the number of derivatives that will be returned in the
        second argument.
        """
        if dfdt_deriv is None:
            return np.dot(dfdt, weight_vector)
        else:
            return np.dot(dfdt, weight_vector), dfdt_deriv @ weight_vector

    def rain_prod_breakdown(self, ktens, cloud_vector, derivative=None):
        """Calculate autoconversion and accretion rates.

        Arguments:
        ktens - Collision KernelTensor.
        cloud_vector - A vector of values between 0 and 1, representing the
                       percentage of mass in a bin that should be considered
                       cloud rather than rain.
        derivative (optional) - If True, returns Jacobian information.
                                Defaults to False.

        If derivative is False, the return value is an array of length 2
        containing only the autoconversion and accretion rates. If derivative
        is True, the return value is a tuple, with the first entry containing
        the process rates and the second entry containing the Jacobian of those
        rates with respect to time and the dsd_deriv variables listed in desc.
        """
        if derivative is None:
            derivative = False
        rate_scale = self.constants.mass_conc_scale / self.constants.time_scale
        grid = self.mass_grid
        nb = grid.num_bins
        m3_vector = grid.moment_weight_vector(3)
        dsd_raw = self.desc.dsd_raw(self.raw)
        total_inter = ktens.calc_rate(dsd_raw, out_flux=True,
                                      derivative=derivative)
        if derivative:
            save_deriv = total_inter[1]
            total_inter = total_inter[0]
            ddn = self.desc.dsd_deriv_num
            dsd_deriv_raw = np.zeros((ddn+1, nb+1))
            dsd_deriv_raw[0,:] = total_inter
            dsd_deriv_raw[1:,:] = self.desc.dsd_deriv_raw(self.raw,
                                                          with_fallout=True)
            total_deriv = save_deriv @ dsd_deriv_raw.T
        cloud_dsd_raw = dsd_raw * cloud_vector
        cloud_inter = ktens.calc_rate(cloud_dsd_raw, out_flux=True,
                                      derivative=derivative)
        if derivative:
            cloud_dsd_deriv = np.transpose(dsd_deriv_raw).copy()
            for i in range(3):
                cloud_dsd_deriv[:nb,i] *= cloud_vector
                cloud_dsd_deriv[nb,i] = 0.
            cloud_deriv = cloud_inter[1] @ cloud_dsd_deriv
            cloud_inter = cloud_inter[0]
        rain_vector = 1. - cloud_vector
        auto = np.dot(cloud_inter[:nb]*rain_vector, m3_vector) \
            + cloud_inter[nb]
        auto *= rate_scale
        no_csc_or_auto = total_inter - cloud_inter
        accr = np.dot(-no_csc_or_auto[:nb]*cloud_vector, m3_vector)
        accr *= rate_scale
        rates = np.array([auto, accr])
        if derivative:
            rate_deriv = np.zeros((2, ddn+1))
            rate_deriv[0,:] = (m3_vector * rain_vector) @ cloud_deriv[:nb,:] \
                + cloud_deriv[nb,:]
            no_soa_deriv = total_deriv - cloud_deriv
            rate_deriv[1,:] = -(m3_vector * cloud_vector) @ no_soa_deriv[:nb,:]
            rate_deriv *= rate_scale
            rate_deriv[:,0] /= self.constants.time_scale
            for i in range(ddn):
                rate_deriv[:,1+i] *= self.desc.dsd_deriv_scales[i]
            return rates, rate_deriv
        else:
            return rates


class RK45Integrator:
    """
    Integrate a model state in time using the SciPy RK45 implementation.

    Initialization arguments:
    dt - Max time step at which to calculate the results.

    Methods:
    integrate_raw
    integrate
    """
    def __init__(self, constants, dt):
        self.constants = constants
        self.dt = dt / constants.time_scale

    def integrate_raw(self, t_len, state, proc_tens):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        dt = self.dt
        tol = dt * 1.e-10
        num_step = int(t_len / dt)
        if t_len - (num_step * dt) > tol:
            num_step = num_step + 1
        times = np.linspace(0., t_len, num_step+1)
        raw_len = len(state.raw)
        rate_fun = lambda t, raw: \
            ModelState(state.desc, raw).time_derivative_raw(proc_tens)
        solbunch = solve_ivp(rate_fun, (times[0], times[-1]), state.raw,
                             method='RK45', t_eval=times, max_step=self.dt)
        output = np.transpose(solbunch.y)
        return times, output

    def integrate(self, t_len, state, proc_tens):
        tscale = self.constants.time_scale
        desc = state.desc
        times, raws = self.integrate_raw(t_len / tscale, state, proc_tens)
        states = []
        for i in range(len(times)):
            states.append(ModelState(desc,raws[i,:]))
        return times * tscale, states


class LogTransform:
    """
    Transform a variable using the natural logarithm.

    Methods:
    transform
    derivative
    second_over_first_derivative
    """
    def transform(self, x):
        return np.log(x)

    def derivative(self, x):
        return 1./x

    def second_over_first_derivative(self, x):
        return -1./x
