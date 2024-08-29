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

"""Test code for math_utils functions."""

import unittest

from scipy.integrate import quad

# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.math_utils import *
from bin_model import ModelConstants, GeometricMassGrid


class TestLogUtilities(unittest.TestCase):
    """
    Test fast implementations of log(exp(x) + exp(y)) and log(exp(x) - exp(y)).
    """

    def test_add_logs(self):
        """Check add_logs on two values."""
        self.assertAlmostEqual(add_logs(1., 2.),
                               np.log(np.exp(1.) + np.exp(2.)))

    def test_sub_logs(self):
        """Check sub_logs on two values."""
        self.assertAlmostEqual(sub_logs(2., 1.),
                               np.log(np.exp(2.) - np.exp(1.)))

    def test_sub_logs_invalid(self):
        """Check sub_logs raises ValueError if second argument isn't smaller."""
        # Log of a negative number is not a single real value.
        with self.assertRaises(ValueError):
            sub_logs(1., 2.)
        # Log of zero is undefined.
        with self.assertRaises(ValueError):
            sub_logs(2., 2.)


class TestDilogarithm(unittest.TestCase):
    """
    Test implementation of the dilogarithm function."""

    def test_dilogarithm(self):
        """Check that dilogarithm is correct where an analytic result is known.

        This is mainly just to check that the dilogarithm used follows the
        expected convention, since in some libraries the reflection
        (`dilogarithm(1-x)`) is implemented instead.
        """
        self.assertAlmostEqual(dilogarithm(1.),
                               np.pi**2 / 6.)


class TestLowerGammaDeriv(unittest.TestCase):
    """
    Tests of lower incomplete gamma function derivative.
    """

    @staticmethod
    def low_gam(s, x):
        """Lower incomplete gamma function using SciPy."""
        return gammainc(s, x) * gamma(s)

    def test_lower_gamma_deriv(self):
        """Check lower_gamma_deriv using finite differences.

        Richardson extrapolation is used to obtain a second-order finite
        difference approximation.
        """
        s = 1.
        x = 2.5
        perturb = 1.e-6
        # Finite difference + Richardson extrapolation.
        expected = (4. * self.low_gam(s+perturb, x)
                    - self.low_gam(s+2.*perturb, x)
                    - 3. * self.low_gam(s, x)) / (2.*perturb)
        actual = lower_gamma_deriv(s, x)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_lower_gamma_deriv_raises_for_nonfinite_s(self):
        """Check lower_gamma_deriv raises ValueError for non-finite s values."""
        with self.assertRaises(ValueError):
            lower_gamma_deriv(np.nan, 1.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(np.inf, 1.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(-np.inf, 1.)

    def test_lower_gamma_deriv_raises_for_nonpositive_x(self):
        """Check lower_gamma_deriv raises ValueError for nonpositive x."""
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 0.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., -1.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., np.nan)

    def test_lower_gamma_deriv_raises_for_nonpositive_atol(self):
        """Check lower_gamma_deriv raises ValueError for nonpositive atol."""
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., atol=0.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., atol=-1.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., atol=np.nan)

    def test_lower_gamma_deriv_raises_for_nonpositive_rtol(self):
        """Check lower_gamma_deriv raises ValueError for nonpositive rtol."""
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., rtol=0.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., rtol=-1.)
        with self.assertRaises(ValueError):
            lower_gamma_deriv(1., 1., rtol=np.nan)

    def test_lower_gamma_deriv_array_inputs(self):
        """Check lower_gamma_deriv works if one or both arguments are arrays."""
        s = np.array([3., 1.3])
        x = np.array([2., 3.5])
        expected = np.array([lower_gamma_deriv(s[0], x[0]),
                             lower_gamma_deriv(s[0], x[1])])
        actual = lower_gamma_deriv(s[0], x)
        for i in range(2):
            self.assertAlmostEqual(actual[i], expected[i])
        expected = np.array([lower_gamma_deriv(s[0], x[0]),
                             lower_gamma_deriv(s[1], x[0])])
        actual = lower_gamma_deriv(s, x[0])
        for i in range(2):
            self.assertAlmostEqual(actual[i], expected[i])
        expected = np.array([lower_gamma_deriv(s[0], x[0]),
                             lower_gamma_deriv(s[1], x[1])])
        actual = lower_gamma_deriv(s, x)
        for i in range(2):
            self.assertAlmostEqual(actual[i], expected[i])

    def test_lower_gamma_deriv_mismatched_shape_raises(self):
        """Check lower_gamma_deriv raises ValueError for incompatible shapes."""
        s = np.array([3., 1.3])
        x = np.array([2., 3.5, 6.])
        with self.assertRaises(ValueError):
            lower_gamma_deriv(s, x)

    def test_lower_gamma_deriv_large_x_s(self):
        """Check lower_gamma_deriv with large x and s values.

        Richardson extrapolation is used to obtain a second-order finite
        difference approximation.
        """
        s = 8.
        x = 25.535422793161228
        actual = lower_gamma_deriv(s, x)
        perturb = 1.e-6
        # Finite difference + Richardson extrapolation.
        expected = (4. * self.low_gam(s+perturb, x)
                    - self.low_gam(s+2.*perturb, x)
                    - 3. * self.low_gam(s, x)) / (2.*perturb)
        self.assertAlmostEqual(actual / expected, 1.)


class TestGammaDistD(unittest.TestCase):
    """
    Tests of gamma_dist_d and related functions.

    The functions tested by this class use a gamma distribution over the
    *diameter*, not the particle *mass*.
    """

    def setUp(self):
        """Set up a MassGrid on which to represent the gamma distribution."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=6)

    def test_gamma_dist_d(self):
        """Check gamma_dist_d analytic solution against numerical quadrature."""
        grid = self.grid
        nbin = grid.num_bins
        bbd = grid.bin_bounds_d
        bwidth = grid.bin_widths
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d(grid.bin_bounds_d, lam, nu)
        self.assertEqual(len(actual), nbin)
        def gamma_test(x):
            """Gamma distribution for use in numerical quadrature."""
            # Note that we add 3 to nu here to get mass-weighted distribution.
            return lam**(nu+3.) * x**(nu+2.) * np.exp(-lam*x) / gamma(nu+3)
        expected = np.zeros((nbin,))
        for i in range(nbin):
            expected[i], _ = quad(gamma_test, bbd[i], bbd[i+1])
        max_expected = np.abs(expected).max()
        for i in range(nbin):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_lam_deriv(self):
        """Check gamma_dist_d_lam against finite differences.

        Richardson extrapolation is used to obtain a second-order finite
        difference approximation.
        """
        bbd = self.grid.bin_bounds_d
        nbin = self.grid.num_bins
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d_lam_deriv(bbd, lam, nu)
        self.assertEqual(len(actual), nbin)
        perturb = 1.
        # Finite difference + Richardson extrapolation.
        expected = (4. * gamma_dist_d(bbd, lam+perturb, nu)
                    - gamma_dist_d(bbd, lam+2.*perturb, nu)
                    - 3. * gamma_dist_d(bbd, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nbin):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_nu_deriv(self):
        """Check gamma_dist_d_nu against finite differences.

        Richardson extrapolation is used to obtain a second-order finite
        difference approximation.
        """
        bbd = self.grid.bin_bounds_d
        nbin = self.grid.num_bins
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d_nu_deriv(bbd, lam, nu)
        self.assertEqual(len(actual), nbin)
        perturb = 1.e-6
        # Finite difference + Richardson extrapolation.
        expected = (4. * gamma_dist_d(bbd, lam, nu+perturb)
                    - gamma_dist_d(bbd, lam, nu+2.*perturb)
                    - 3. * gamma_dist_d(bbd, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nbin):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_nu_deriv_small_lam(self):
        """Check gamma_dist_d_nu against finite differences with small lambda.

        Richardson extrapolation is used to obtain a second-order finite
        difference approximation.
        """
        bbd = self.grid.bin_bounds_d
        nbin = self.grid.num_bins
        nu = 5.
        lam = nu / 1.5e-7
        actual = gamma_dist_d_nu_deriv(bbd, lam, nu)
        self.assertEqual(len(actual), nbin)
        perturb = 1.e-3
        # Finite difference + Richardson extrapolation.
        expected = (4. * gamma_dist_d(bbd, lam, nu+perturb)
                    - gamma_dist_d(bbd, lam, nu+2.*perturb)
                    - 3. * gamma_dist_d(bbd, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nbin):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected,
                                   places=5)
