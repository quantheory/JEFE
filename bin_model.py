"""Types related to a simple single-moment 1D microphysics scheme.

Classes:
LongKernel
Kernel

Utility functions:
add_logs
sub_logs
"""

import numpy as np
from scipy.special import spence

def add_logs(x, y):
    """Returns log(exp(x)+exp(y))."""
    return x + np.log(1. + np.exp(y-x))

def sub_logs(x, y):
    """Returns log(exp(x)-exp(y))."""
    assert y < x, "y < x in sub_logs"
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

    Attributes:
    std_mass - Mass in kg corresponding to a scaled mass of 1.

    Methods:
    diameter_to_scaled_mass - Convert diameter in meters to scaled mass.
    scaled_mass_to_diameter - Convert scaled mass to diameter in meters.
    """

    def __init__(self, rho_water, rho_air, std_diameter):
        self.rho_water = rho_water
        self.rho_air = rho_air
        self.std_diameter = std_diameter
        self.std_mass = rho_water * np.pi/6. * std_diameter**3

    def diameter_to_scaled_mass(self, d):
        """Convert diameter in meters to non-dimensionalized particle size."""
        return (d / self.std_diameter)**3

    def scaled_mass_to_diameter(self, x):
        """Convert non-dimensionalized particle size to diameter in meters."""
        return self.std_diameter * x**(1./3.)


class Kernel:
    """
    Represent a collision kernel for the bin model.

    Attributes:
    scale - Characteristic magnitude of kernel.
    """

    scale = None


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
                 kr_si=None):
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

    def _calculate_edges(self, ly1, ly2, lz1, lz2):
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
        bottom_left = sub_logs(lz1, ly1)
        top_left = sub_logs(lz1, ly2)
        bottom_right = sub_logs(lz2, ly1)
        top_right = sub_logs(lz2, ly2)
        return (bottom_left, top_left, bottom_right, top_right)
