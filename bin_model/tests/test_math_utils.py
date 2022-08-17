"""Test code for math_utils functions."""

import unittest

from scipy.integrate import quad

from bin_model.math_utils import *
from bin_model import ModelConstants, GeometricMassGrid


class TestLogUtilities(unittest.TestCase):
    """
    Test fast implementations of log(exp(x) + exp(y)) and log(exp(x) - exp(y)).
    """

    def test_add_logs(self):
        self.assertAlmostEqual(add_logs(1., 2.),
                               np.log(np.exp(1.) + np.exp(2.)))

    def test_sub_logs(self):
        self.assertAlmostEqual(sub_logs(2., 1.),
                               np.log(np.exp(2.) - np.exp(1.)))

    def test_sub_logs_invalid(self):
        # Log of a negative number is not a valid real.
        with self.assertRaises(AssertionError):
            sub_logs(1., 2.)


class TestDilogarithm(unittest.TestCase):
    """
    Test implementation of the dilogarithm function.

    This is mainly just to check that the dilogarithm used follows the expected
    convention, since in some libraries the reflection (`dilogarithm(1-x)`) is
    implemented instead.
    """

    def test_dilogarithm(self):
        self.assertAlmostEqual(dilogarithm(1.),
                               np.pi**2 / 6.)


class TestLowerGammaDeriv(unittest.TestCase):
    """
    Tests of lower incomplete gamma function derivative.
    """

    def test_lower_gamma_deriv(self):
        s = 1.
        x = 2.5
        low_gam = lambda s, x: gammainc(s, x) * gamma(s)
        perturb = 1.e-6
        # Richardson extrapolation.
        expected = (4. * low_gam(s+perturb, x)
                    - low_gam(s+2.*perturb, x)
                    - 3. * low_gam(s, x)) / (2.*perturb)
        actual = lower_gamma_deriv(s, x)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_lower_gamma_deriv_raises_for_negative_x(self):
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., 0.)
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., -1.)

    def test_lower_gamma_deriv_raises_for_negative_atol(self):
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., 1., atol=0.)
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., 1., atol=-1.)

    def test_lower_gamma_deriv_raises_for_negative_rtol(self):
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., 1., rtol=0.)
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(1., 1., rtol=-1.)

    def test_lower_gamma_deriv_array_inputs(self):
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
        s = np.array([3., 1.3])
        x = np.array([2., 3.5, 6.])
        with self.assertRaises(AssertionError):
            lower_gamma_deriv(s, x)

    def test_lower_gamma_deriv_large_x(self):
        s = 8.
        x = 25.535422793161228
        actual = lower_gamma_deriv(s, x)
        low_gam = lambda s, x: gammainc(s, x) * gamma(s)
        perturb = 1.e-6
        # Richardson extrapolation.
        expected = (4. * low_gam(s+perturb, x)
                    - low_gam(s+2.*perturb, x)
                    - 3. * low_gam(s, x)) / (2.*perturb)
        self.assertAlmostEqual(lower_gamma_deriv(s, x) / expected, 1.)


class TestGammaDistD(unittest.TestCase):
    """
    Tests of gamma_dist_d and related functions.

    The functions tested by this class use a gamma distribution over the
    *diameter*, not the particle *mass*.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=6)

    def test_gamma_dist_d(self):
        grid = self.grid
        nb = grid.num_bins
        bbd = grid.bin_bounds_d
        bw = grid.bin_widths
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d(grid, lam, nu)
        self.assertEqual(len(actual), nb)
        # Note that we add 3 to nu here to get mass weighted distribution.
        f = lambda x: lam**(nu+3.) * x**(nu+2.) * np.exp(-lam*x) / gamma(nu+3)
        expected = np.zeros((nb,))
        for i in range(nb):
            y, _ = quad(f, bbd[i], bbd[i+1])
            expected[i] = y / bw[i]
        max_expected = np.abs(expected).max()
        for i in range(nb):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_lam_deriv(self):
        grid = self.grid
        nb = grid.num_bins
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d_lam_deriv(grid, lam, nu)
        self.assertEqual(len(actual), nb)
        perturb = 1.
        # Richardson extrapolation.
        expected = (4. * gamma_dist_d(grid, lam+perturb, nu)
                    - gamma_dist_d(grid, lam+2.*perturb, nu)
                    - 3. * gamma_dist_d(grid, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nb):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_nu_deriv(self):
        grid = self.grid
        nb = grid.num_bins
        nu = 5.
        lam = nu / 1.5e-6
        actual = gamma_dist_d_nu_deriv(grid, lam, nu)
        self.assertEqual(len(actual), nb)
        perturb = 1.e-6
        # Richardson extrapolation.
        expected = (4. * gamma_dist_d(grid, lam, nu+perturb)
                    - gamma_dist_d(grid, lam, nu+2.*perturb)
                    - 3. * gamma_dist_d(grid, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nb):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected)

    def test_gamma_dist_d_nu_deriv_small_lam(self):
        grid = self.grid
        nb = grid.num_bins
        nu = 5.
        lam = nu / 1.5e-7
        actual = gamma_dist_d_nu_deriv(grid, lam, nu)
        self.assertEqual(len(actual), nb)
        perturb = 1.e-3
        # Richardson extrapolation.
        expected = (4. * gamma_dist_d(grid, lam, nu+perturb)
                    - gamma_dist_d(grid, lam, nu+2.*perturb)
                    - 3. * gamma_dist_d(grid, lam, nu)) \
                        / (2.*perturb)
        max_expected = np.abs(expected).max()
        for i in range(nb):
            self.assertAlmostEqual(actual[i] / max_expected,
                                   expected[i] / max_expected,
                                   places=5)
