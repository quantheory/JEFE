"""Math utility functions used by the JEFE bin model."""

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

    For purposes of this function, the lower incomplete gamma function is
    defined as:

    $$\gamma(s,x) = \int_0^x t^{s-1} e^{-t} dt$$

    This function finds the derivative of the lower incomplete gamma function
    with respect to its shape parameter "s". This is done with a series if x is
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
