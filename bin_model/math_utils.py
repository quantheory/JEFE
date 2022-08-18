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

"""Math utility functions used by the JEFE bin model."""

import numpy as np
from scipy.special import spence, gamma, gammainc, digamma

def add_logs(x, y):
    """Returns log(exp(x)+exp(y))."""
    return x + np.log(1. + np.exp(y-x))

def sub_logs(x, y):
    """Returns log(exp(x)-exp(y))."""
    if not y < x:
        raise ValueError("second argument not less than first in sub_logs"
                         f" (first {x} vs. second {y})")
    return x + np.log(1. - np.exp(y-x))

def dilogarithm(x):
    """Returns the dilogarithm of the input."""
    return spence(1. - x)

def lower_gamma_deriv(s, x, atol=1.e-14, rtol=1.e-14):
    r"""Derivative of lower incomplete gamma function with respect to s.

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
    # The following are written the way that they are so that NaN values will
    # also cause ValueError to be raised.
    if not np.all(np.isfinite(s)):
        raise ValueError(f"s must be finite but is {s}")
    if not np.all(x > 0.):
        raise ValueError(f"x must be positive but is {x}")
    if not atol > 0.: # pylint: disable=unneeded-not
        raise ValueError(f"atol must be positive but is {atol}")
    if not rtol > 0.: # pylint: disable=unneeded-not
        raise ValueError(f"rtol must be positive but is {rtol}")
    if isinstance(x, np.ndarray):
        out = np.zeros(x.shape)
        if isinstance(s, np.ndarray):
            if s.shape != x.shape:
                raise ValueError("shapes of s and x are incompatible"
                                 f" ({s.shape} vs. {x.shape})")
            for i, x_i in enumerate(x.flat):
                out.flat[i] = _lower_gamma_deriv_scalar(s.flat[i], x_i,
                                                        atol, rtol)
        else:
            for i, x_i in enumerate(x.flat):
                out.flat[i] = _lower_gamma_deriv_scalar(s, x_i, atol, rtol)
        return out
    if isinstance(s, np.ndarray):
        out = np.zeros(s.shape)
        for i, s_i in enumerate(s.flat):
            out.flat[i] = _lower_gamma_deriv_scalar(s_i, x, atol, rtol)
        return out
    return _lower_gamma_deriv_scalar(s, x, atol, rtol)


def _lower_gamma_deriv_scalar(s, x, atol, rtol):
    # All methods need the log of x.
    l = np.log(x)
    # Using the fact that the lower incomplete gamma function satisfies
    #     gamma(s+1,x) = s * gamma(s, x) - x**s * exp(-x)
    # we can take the derivative of each side with respect to s to get an
    # equation for lowering the value of s.
    add = 0.
    fac = 1.
    new_s = s
    while new_s > 2.:
        new_s -= 1.
        add += fac * (gammainc(new_s, x) * gamma(new_s)
                      - l * x**new_s * np.exp(-x))
        fac *= new_s
    # Calculate the derivative for the lowered value of s.
    # The formulas in these two cases are taken from certain cases in the
    # (considerably more sophisticated) Boost library calculation described
    # here:
    #
    # https://www.boost.org/doc/libs/1_80_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
    if x < 2.:
        y = _lower_gamma_deriv_low_x(new_s, x, l, atol, rtol)
    else:
        y = _lower_gamma_deriv_high_x(new_s, x, l, atol, rtol)
    # Correct the result to obtain the answer for the original value of s.
    return add + fac*y

def _lower_gamma_deriv_low_x(s, x, l, atol, rtol):
    # This function uses the series representation:
    #     x^s \sum_{k=0}^{\infty} \frac{(-x)^k}{k!(s+k)} [\log(x)-\frac{1}{s+k}]
    # which is simply the derivative with respect to s of the power series
    #     gamma(s, x) = x^s \sum_{k=0}^{\infty} \frac{(-x)^k}{k!(s+k)}
    y = 0.
    k = 0
    fac = x**s
    recip_s_plus_k = 1./(s + k)
    err_lim = 1.e100
    while x > k or err_lim > atol + abs(y)*rtol:
        term = fac * recip_s_plus_k * (l - recip_s_plus_k)
        y += term
        # Set up for next iteration.
        k += 1
        fac *= -x / k
        recip_s_plus_k = 1./(s + k)
        err_lim = abs(fac) * recip_s_plus_k * max(l, recip_s_plus_k)
    return y

def _lower_gamma_deriv_high_x(s, x, l, atol, rtol):
    # This function uses Legendre's continued fraction for the *upper*
    # incomplete gamma function:
    #     Gamma(s, x) = \frac{a_0}{b_0 + \frac{a_1}{b_1 + \dots}}
    #     a_0 = x^s e^{-x}
    #     a_k = k(s-k) for k >= 1
    #     b_k = 2k + 1 + x - s for k >= 0
    # The numerators (n_k) and denominators (d_k) of the convergents of any such
    # continued fraction satisfy the recursion relation:
    #     n_k = a_k n_{k-2} + b_k n_{k-1}
    #     d_k = a_k d_{k-2} + b_k d_{k-1}
    # Taking the derivative of each equation with respect to s, we obtain a set
    # of recursion relations for n_k and d(n_k)/ds in terms of previous values,
    # and the same for d_k. Then an approximation to the derivative of the
    # upper incomplete gamma function is:
    #     [d_k d(n_k)/ds - n_k d(d_k)/ds] / d_k^2

    # Calculate derivative of the complete Gamma function; the lower incomplete
    # gamma derivative is this minus the upper incomplete gamma derivative.
    dgamma = digamma(s) * gamma(s)
    # num_vec contains the values
    #      (n_{k-1}, d(n_{k-1})/ds, n_k, d(n_k)/ds)
    # and den_vec is defined similarly using d_k.
    num_vec = np.zeros((4,))
    den_vec = np.zeros((4,))
    # First sequence entries are n_{-1}=0, d_{-1}=1, n_0 = (x^s e^-x), and
    # d_0 = (x-s+1).
    den_vec[0] = 1.
    num_vec[2] = x**s * np.exp(-x)
    num_vec[3] = l * num_vec[2]
    den_vec[2] = x - s + 1.
    den_vec[3] = -1.
    y_prev = dgamma
    y = dgamma - (den_vec[2] * num_vec[3] - num_vec[2] * den_vec[3]) \
        / den_vec[2]**2
    # Matrix applied at each step to perform the recursion.
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
        # The numerators and denominators can easily grow quite large, but we
        # can divide them by a common factor without affecting the result.
        if np.any(num_vec > 1.e100) or np.any(den_vec > 1.e100):
            num_vec /= 1.e100
            den_vec /= 1.e100
        # This rearrangement of the formula has the benefit of avoiding overflow
        # errors, though it is probably not the optimal choice.
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
    gnu3 = gamma(nu+3.)
    bin_func = lower_gamma_deriv(nu+3., lam*grid.bin_bounds_d) / gnu3
    bin_func -= gammainc(nu+3., lam*grid.bin_bounds_d) * digamma(nu+3.)
    return (bin_func[1:] - bin_func[:-1]) / grid.bin_widths
