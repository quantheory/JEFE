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

# pylint: disable=protected-access, too-many-public-methods

"""Tests for kernel module."""

import unittest

from scipy.integrate import dblquad

from bin_model.math_utils import add_logs, sub_logs, dilogarithm
from bin_model.constants import ModelConstants
# pylint: disable-next=wildcard-import
from bin_model.kernel import *


class TestBeardV(unittest.TestCase):
    """
    Test beard_v function.
    """
    def setUp(self):
        """Set up constants for evaluating beard_v."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)

    def test_low_beard_v(self):
        """Check a beard_v value at the low end of the size range."""
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 1.e-5),
                               0.0030440)

    def test_medium_beard_v(self):
        """Check a beard_v value in the middle of the size range."""
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 6.e-4),
                               2.4455837)

    def test_high_beard_v(self):
        """Check a beard_v value at the high end of the size range."""
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 2.e-3),
                               6.5141471)

    def test_very_high_beard_v(self):
        """Check a beard_v value above the high end of the range.

        For diameters above 7 mm, this function returns a constant.
        """
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 7.e-3),
                               beard_v(const, 7.001e-3))
        self.assertAlmostEqual(beard_v(const, 7.e-3),
                               beard_v(const, 8.e-3))


class TestSCEfficiency(unittest.TestCase):
    """
    Test sc_efficiency function.
    """

    def test_sc_efficiency(self):
        """Test the Scott-Chen collection efficiency formula."""
        d1 = 50.e-6
        x = 0.7
        d2 = x * d1
        self.assertAlmostEqual(sc_efficiency(d1, d2),
                               (0.8 / (1. + x))**2,
                               places=2)
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))

    def test_sc_efficiency_low_diameter(self):
        """Test the Scott-Chen formula's extrapolation to small radii."""
        d1 = 5.e-6
        x = 0.7
        d2 = x * d1
        self.assertAlmostEqual(sc_efficiency(d1, d2),
                               sc_efficiency(20.e-6, x * 20.e-6))
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))


class TestKernel(unittest.TestCase):
    """
    Test utility methods on Kernel objects.

    Kernel is really supposed to be more of an abstract base class. If measures
    are taken to prevent it from being constructed in the usual way, this
    testing strategy will need to change.
    """

    def setUp(self):
        self.kernel = Kernel()

    def test_find_corners_invalid(self):
        kernel = self.kernel
        ly1 = 1.
        ly2 = 0.
        lz1 = 2.
        lz2 = 3.
        with self.assertRaises(ValueError):
            actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        ly1 = 0.
        ly2 = 1.
        lz1 = 3.
        lz2 = 2.
        with self.assertRaises(ValueError):
            actual = kernel.find_corners((ly1, ly2), (lz1, lz2))

    def test_find_corners_bins_separated(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), sub_logs(lz1, ly2),
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_bins_overlap(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 0.
        ly2 = 2.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_large_y2(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 3.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), None)
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 0.
        ly2 = 4.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), None)
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_large_y1(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 2.
        lz1 = 0.
        lz2 = 3.
        expected = (None, None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 1.
        ly2 = 2.
        lz1 = 0.
        lz2 = 3.
        expected = (None, None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_empty_set(self):
        kernel = self.kernel
        ly1 = 1.
        ly2 = 3.
        lz1 = 0.
        lz2 = 1.
        expected = (None, None,
                    None, None)
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 2.
        ly2 = 3.
        lz1 = 0.
        lz2 = 1.
        expected = (None, None,
                    None, None)
        actual = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertAlmostEqual(actual, expected)

    def test_min_max_ly_asserts_valid_btype(self):
        kernel = self.kernel
        a = 2.
        b = 3.
        lxm = 0.
        lxp = 1.
        with self.assertRaises(ValueError):
            kernel.min_max_ly((lxm, lxp), (a, b), btype=-1)
        with self.assertRaises(ValueError):
            kernel.min_max_ly((lxm, lxp), (a, b), btype=4)

    def test_min_max_ly(self):
        kernel = self.kernel
        a = 2.
        b = 3.
        lxm = 0.
        lxp = 1.
        expected = (a, b)
        actual = kernel.min_max_ly((lxm, lxp), (a, b), btype=0)
        self.assertEqual(actual, expected)
        expected = (sub_logs(a, lxp), b)
        actual = kernel.min_max_ly((lxm, lxp), (a, b), btype=1)
        self.assertEqual(actual, expected)
        expected = (a, sub_logs(b, lxm))
        actual = kernel.min_max_ly((lxm, lxp), (a, b), btype=2)
        self.assertEqual(actual, expected)
        expected = (sub_logs(a, lxp), sub_logs(b, lxm))
        actual = kernel.min_max_ly((lxm, lxp), (a, b), btype=3)
        self.assertEqual(actual, expected)

    def test_get_lxs_and_btypes_assertions(self):
        # Test assertions for bins with bounds in the wrong order.
        kernel = self.kernel
        lx1 = 0.
        lx2 = -1.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        with self.assertRaises(ValueError):
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        lx1 = 0.
        lx2 = 1.
        ly1 = 0.
        ly2 = -1.
        with self.assertRaises(ValueError):
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        ly1 = 0.
        ly2 = 1.
        lz1 = 3.
        lz2 = 2.
        with self.assertRaises(ValueError):
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))

    def test_get_lxs_and_btypes_wide_lx(self):
        # The case where ly and lz are separated, and lx has a wide enough
        # range that its bounds are irrelevant.
        kernel = self.kernel
        lx1 = -20.
        lx2 = 20.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [tl, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.01
        lz2 = 1.02
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [tl, tr, bl, br])
        self.assertListEqual(actual_btypes, [1, 3, 2])

    def test_get_lxs_and_btypes_no_overlap(self):
        # Cases where coalescence between the x and y bins will never produce
        # output in the z bin.
        kernel = self.kernel
        lx1 = 19.
        lx2 = 20.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [])
        self.assertListEqual(actual_btypes, [])
        lx1 = -20.
        lx2 = -19.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [])
        self.assertListEqual(actual_btypes, [])

    def test_get_lxs_and_btypes_lower_lx(self):
        # Cases where the lower lx bound affects the integration region.
        kernel = self.kernel
        # First test where x bound cuts off left region
        lx1 = 0.5
        lx2 = 20.
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.01
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # x bound large enough to cut off middle region
        lx1 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, tr, br])
        self.assertListEqual(actual_btypes, [0, 2])
        # x bound large enough to cut off right region
        lx1 = 1.7
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, br])
        self.assertListEqual(actual_btypes, [2])

    def test_get_lxs_and_btypes_upper_lx(self):
        # Cases where the upper lx bound affects the integration region.
        kernel = self.kernel
        # First test where x bound cuts off right region
        lx1 = -20
        lx2 = 1.7
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.01
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [tl, bl, tr, lx2])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # x small enough to cut off middle region
        lx2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [tl, bl, lx2])
        self.assertListEqual(actual_btypes, [1, 0])

    def test_get_lxs_and_btypes_missing_corners(self):
        # Cases where some of the corners don't exist (go to -Infinity).
        kernel = self.kernel
        # No top left corner.
        lx1 = -20.
        lx2 = 20.
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # No bottom left corner.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, tr, br])
        self.assertListEqual(actual_btypes, [0, 2])
        # No top right corner.
        ly1 = 0.
        ly2 = 2.
        lz1 = 1.
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, bl, br])
        self.assertListEqual(actual_btypes, [3, 2])
        # No bottom left or top right corner.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, br])
        self.assertListEqual(actual_btypes, [2])
        # No bottom left or top right corner, upper x bound matters.
        lx2 = 0.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, lx2])
        self.assertListEqual(actual_btypes, [2])
        # No region at all.
        ly1 = 1.
        ly2 = 2.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [])
        self.assertListEqual(actual_btypes, [])

    def test_get_lxs_and_btypes_infinite_lz_upper(self):
        # Test specifying an infinite upper bound for lz.
        kernel = self.kernel
        lx1 = -20.
        lx2 = 20.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = np.inf
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [tl, bl, lx2])
        self.assertListEqual(actual_btypes, [1, 0])
        # Test specifying an infinite upper bound for lz when the lower bound
        # is the same as ly's.
        kernel = self.kernel
        lx1 = -2.
        lx2 = -1.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = np.inf
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes((lx1, lx2), (ly1, ly2), (lz1, lz2))
        bl, tl, br, tr = kernel.find_corners((ly1, ly2), (lz1, lz2))
        self.assertListEqual(actual_lxs, [lx1, lx2])
        self.assertListEqual(actual_btypes, [0])

    def test_get_y_bound_p(self):
        kernel = self.kernel
        btypes = [0, 1, 2, 3]
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        y_bound_p = kernel.get_y_bound_p((ly1, ly2), (lz1, lz2), btypes)
        self.assertListEqual(list(y_bound_p[:,0]), [0., 2., 0., 2.])
        self.assertListEqual(list(y_bound_p[:,1]), [1., 1., 3., 3.])

    def test_integrate_over_bins_raises(self):
        kernel = self.kernel
        lx1 = 0.
        lx2 = 1.
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.
        lz2 = 2.
        with self.assertRaises(NotImplementedError):
            kernel.integrate_over_bins((lx1, lx2), (ly1, ly2), (lz1, lz2))


def reference_long_cloud(a, b, lxm, lxp):
    r"""Reference implementation of double integral used for Long's kernel.

    The definite integral being computed is:

    \int_{lxm}^{lxp} \int_{a}^{log(e^b-e^x)} (e^{2x - y} + e^{y}) dy dx
    """
    return np.exp(b) * np.log((np.exp(b)-np.exp(lxp))/(np.exp(b)-np.exp(lxm)))\
        + np.exp(-a) / 2. * (np.exp(2.*lxp)-np.exp(2.*lxm))\
        + (np.exp(b)-np.exp(a))*(lxp-lxm)

def reference_long_rain(a, b, lxm, lxp):
    r"""Reference implementation of double integral used for Long's kernel.

    The definite integral being computed is:

    \int_{lxm}^{lxp} \int_{a}^{log(e^b-e^x)} (e^{x - y} + 1) dy dx
    """
    return np.log((np.exp(b)-np.exp(lxp))/(np.exp(b)-np.exp(lxm)))\
        + np.exp(-a) * (np.exp(lxp)-np.exp(lxm))\
        - dilogarithm(np.exp(lxp-b)) + dilogarithm(np.exp(lxm-b))\
        + (b-a)*(lxp-lxm)

class TestReferenceLong(unittest.TestCase):
    """
    Test reference functions that will be used to test LongKernel.
    """

    def test_reference_long_cloud(self):
        integrand = lambda y, x: np.exp(2.*x - y) + np.exp(y)
        a = 0.
        b = 1.
        lxm = -1.
        lxp = 0.
        hfun = lambda x: np.log(np.exp(b)-np.exp(x))
        expected, err = dblquad(integrand, lxm, lxp, a, hfun, epsabs=1.e-13)
        actual = reference_long_cloud(a, b, lxm, lxp)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_reference_long_rain(self):
        integrand = lambda y, x: np.exp(x - y) + 1.
        a = 0.
        b = 1.
        lxm = -1.
        lxp = 0.
        hfun = lambda x: np.log(np.exp(b)-np.exp(x))
        expected, err = dblquad(integrand, lxm, lxp, a, hfun, epsabs=1.e-13)
        actual = reference_long_rain(a, b, lxm, lxp)
        self.assertAlmostEqual(actual, expected, places=12)


class TestLongKernel(unittest.TestCase):
    """
    Tests of all LongKernel methods.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)

    def test_fail_if_two_kcs(self):
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc_cgs=1., kc_si=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc_cgs=1., kc=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc=1., kc_si=1.)

    def test_fail_if_two_krs(self):
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr_cgs=1., kr_si=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr_cgs=1., kr=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr=1., kr_si=1.)

    def test_kc(self):
        kernel = LongKernel(self.constants, kc=3.)
        self.assertEqual(kernel.kc, 3.)

    def test_kr(self):
        kernel = LongKernel(self.constants, kr=3.)
        self.assertEqual(kernel.kr, 3.)

    def test_kc_cgs(self):
        kernel = LongKernel(self.constants, kc_cgs=3.)
        expected = 3. * self.constants.std_mass**2
        self.assertAlmostEqual(kernel.kc, expected, places=25)

    def test_kc_si(self):
        kernel1 = LongKernel(self.constants, kc_cgs=3.)
        kernel2 = LongKernel(self.constants, kc_si=3.)
        self.assertEqual(kernel1.kc, kernel2.kc)

    def test_kc_default(self):
        kernel1 = LongKernel(self.constants)
        kernel2 = LongKernel(self.constants, kc_cgs=9.44e9)
        self.assertAlmostEqual(kernel1.kc, kernel2.kc, places=15)

    def test_kr_si(self):
        kernel = LongKernel(self.constants, kr_si=3.)
        expected = 3. * self.constants.std_mass
        self.assertAlmostEqual(kernel.kr, expected, places=15)

    def test_kr_cgs(self):
        kernel1 = LongKernel(self.constants, kr_cgs=3.)
        kernel2 = LongKernel(self.constants, kr_si=3.e-3)
        self.assertAlmostEqual(kernel1.kr, kernel2.kr, places=15)

    def test_kr_default(self):
        kernel1 = LongKernel(self.constants)
        kernel2 = LongKernel(self.constants, kr_cgs=5.78e3)
        self.assertAlmostEqual(kernel1.kr, kernel2.kr, places=15)

    def test_log_rain_m(self):
        kernel = LongKernel(self.constants, rain_m=0.8)
        self.assertAlmostEqual(kernel.log_rain_m, np.log(0.8))

    def test_log_rain_m_default(self):
        constants = ModelConstants(rho_water=1000.,
                                   rho_air=1.2,
                                   std_diameter=1.e-4,
                                   rain_d=8.e-5)
        kernel = LongKernel(constants)
        # Original Long kernel value.
        self.assertAlmostEqual(kernel.log_rain_m, 0.)

    def test_integral_cloud_btype_bounds(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        with self.assertRaises(ValueError):
            actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=-1)
        with self.assertRaises(ValueError):
            actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=4)

    def test_integral_cloud_type_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = reference_long_cloud(a, 1., lxm, lxp) \
                    - reference_long_cloud(b, 1., lxm, lxp)
        expected *= kernel.kc
        actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_cloud(b, a, lxm, lxp)
        expected *= kernel.kc
        actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=1)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_cloud(a, b, lxm, lxp)
        expected *= kernel.kc
        actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=2)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_cloud(1., b, lxm, lxp) \
                    - reference_long_cloud(1., a, lxm, lxp)
        expected *= kernel.kc
        actual = kernel._integral_cloud((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_btype_bounds(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        with self.assertRaises(ValueError):
            actual = kernel._integral_rain((lxm, lxp), (a, b), btype=-1)
        with self.assertRaises(ValueError):
            actual = kernel._integral_rain((lxm, lxp), (a, b), btype=4)

    def test_integral_rain_type_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = reference_long_rain(a, 1., lxm, lxp) \
                    - reference_long_rain(b, 1., lxm, lxp)
        expected *= kernel.kr
        actual = kernel._integral_rain((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_rain(b, a, lxm, lxp)
        expected *= kernel.kr
        actual = kernel._integral_rain((lxm, lxp), (a, b), btype=1)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_rain(a, b, lxm, lxp)
        expected *= kernel.kr
        actual = kernel._integral_rain((lxm, lxp), (a, b), btype=2)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_rain(1., b, lxm, lxp) \
                    - reference_long_rain(1., a, lxm, lxp)
        expected *= kernel.kr
        actual = kernel._integral_rain((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_kernel_integral_asserts_valid_btype(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        with self.assertRaises(ValueError):
            kernel.kernel_integral((lxm, lxp), (a, b), btype=-1)
        with self.assertRaises(ValueError):
            kernel.kernel_integral((lxm, lxp), (a, b), btype=4)

    def test_kernel_integral_cloud(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = kernel._integral_cloud((lxm, lxp), (a, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_cloud_btype_3(self):
        kernel = LongKernel(self.constants)
        a = add_logs(0., -2.)
        b = add_logs(0., -1.)
        lxm = -1.
        lxp = 0.
        expected = kernel._integral_cloud((lxm, lxp), (a, b), btype=3)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_rain(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = 0.
        lxp = 1.
        expected = kernel._integral_rain((lxm, lxp), (a, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)
        kernel = LongKernel(self.constants)
        a = 0.
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = kernel._integral_rain((lxm, lxp), (a, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)
        kernel = LongKernel(self.constants)
        a = 0.
        b = 1.
        lxm = 0.
        lxp = 1.
        expected = kernel._integral_rain((lxm, lxp), (a, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_lx_spans_rain_m(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 1.
        expected = \
            kernel._integral_cloud((lxm, kernel.log_rain_m), (a, b), btype=0)
        expected += \
            kernel._integral_rain((kernel.log_rain_m, lxp), (a, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = \
            kernel._integral_cloud((lxm, lxp), (a, kernel.log_rain_m), btype=0)
        expected += \
            kernel._integral_rain((lxm, lxp), (kernel.log_rain_m, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_1_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1., 0.)
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = \
            kernel._integral_cloud((lxm, lxp), (a, kernel.log_rain_m), btype=1)
        expected += \
            kernel._integral_rain((lxm, lxp), (kernel.log_rain_m, b), btype=0)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=1)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_1_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1., 0.1)
        b = 1.
        lxm = -1.
        lxp = 0.
        cross_x = sub_logs(a, kernel.log_rain_m)
        expected = \
            kernel._integral_cloud((cross_x, lxp), (a, kernel.log_rain_m),
                                   btype=1)
        expected += \
            kernel._integral_rain((cross_x, lxp), (kernel.log_rain_m, b),
                                  btype=0)
        expected += \
            kernel._integral_rain((lxm, cross_x), (a, b), btype=1)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=1)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_2_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        expected = \
            kernel._integral_cloud((lxm, lxp), (a, kernel.log_rain_m), btype=0)
        expected += \
            kernel._integral_rain((lxm, lxp), (kernel.log_rain_m, b), btype=2)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=2)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_2_crossing(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(b, kernel.log_rain_m)
        expected = \
            kernel._integral_cloud((lxm, cross_x), (a, kernel.log_rain_m),
                                   btype=0)
        expected += \
            kernel._integral_cloud((cross_x, lxp), (a, b), btype=2)
        expected += \
            kernel._integral_rain((lxm, cross_x), (kernel.log_rain_m, b),
                                  btype=2)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=2)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.)
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        expected = \
            kernel._integral_cloud((lxm, lxp), (a, kernel.log_rain_m), btype=1)
        expected += \
            kernel._integral_rain((lxm, lxp), (kernel.log_rain_m, b), btype=2)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_bot_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.05)
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(a, kernel.log_rain_m)
        expected = \
            kernel._integral_cloud((cross_x, lxp),
                                   (a, kernel.log_rain_m), btype=1)
        expected += \
            kernel._integral_rain((cross_x, lxp),
                                  (kernel.log_rain_m, b), btype=2)
        expected += \
            kernel._integral_rain((lxm, cross_x),
                                  (a, b), btype=3)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_top_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.)
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(b, kernel.log_rain_m)
        expected = \
            kernel._integral_cloud((lxm, cross_x),
                                   (a, kernel.log_rain_m), btype=1)
        expected += \
            kernel._integral_cloud((cross_x, lxp), (a, b), btype=3)
        expected += \
            kernel._integral_rain((lxm, cross_x),
                                  (kernel.log_rain_m, b), btype=2)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_both_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.05)
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        x_low = sub_logs(a, kernel.log_rain_m)
        x_high = sub_logs(b, kernel.log_rain_m)
        expected = \
            kernel._integral_cloud((x_low, x_high),
                                   (a, kernel.log_rain_m), btype=1)
        expected += \
            kernel._integral_cloud((x_high, lxp), (a, b), btype=3)
        expected += \
            kernel._integral_rain((x_low, x_high),
                                  (kernel.log_rain_m, b), btype=2)
        expected += \
            kernel._integral_rain((lxm, x_low),
                                  (a, b), btype=3)
        actual = kernel.kernel_integral((lxm, lxp), (a, b), btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_integrate_over_bins(self):
        # This is just to test that the parent class's implementation does not
        # raise an error.
        kernel = LongKernel(self.constants)
        lx1 = -1.
        lx2 = 0.
        ly1 = -1.
        ly2 = 0.
        lz1 = 0.
        lz2 = 1.
        actual = kernel.integrate_over_bins((lx1, lx2), (ly1, ly2), (lz1, lz2))
        expected = 3.1444320613930285e-09
        self.assertAlmostEqual(actual, expected, places=20)


class TestHallKernel(unittest.TestCase):
    """
    Test HallKernel methods.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        self.kernel = HallKernel(self.constants, 'ScottChen')

    def test_bad_efficiency_string_raises_error(self):
        with self.assertRaises(ValueError):
            HallKernel(self.constants, 'nonsense')

    def test_kernel_d(self):
        const = self.constants
        d1 = 10.e-6
        d2 = 100.e-6
        actual = self.kernel.kernel_d(d1, d2)
        expected = np.abs(beard_v(const, d1) - beard_v(const, d2))
        expected *= sc_efficiency(d1, d2)
        expected *= 0.25 * np.pi * (d1 + d2)**2
        self.assertAlmostEqual(actual / expected, 1.)
        self.assertEqual(actual, self.kernel.kernel_d(d2, d1))

    def test_kernel_lx(self):
        const = self.constants
        d1 = 10.e-6
        d2 = 100.e-6
        x1 = const.diameter_to_scaled_mass(d1)
        x2 = const.diameter_to_scaled_mass(d2)
        lx1 = np.log(x1)
        lx2 = np.log(x2)
        actual = self.kernel.kernel_lx(lx1, lx2)
        expected = self.kernel.kernel_d(d1, d2) / (x1 * x2)
        self.assertAlmostEqual(actual / expected, 1.)
        self.assertEqual(actual, self.kernel.kernel_lx(lx2, lx1))

    def test_kernel_integral_asserts_valid_btype(self):
        kernel = self.kernel
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        with self.assertRaises(ValueError):
            kernel.kernel_integral((lxm, lxp), (a, b), btype=-1)
        with self.assertRaises(ValueError):
            kernel.kernel_integral((lxm, lxp), (a, b), btype=4)

    def test_kernel_integral(self):
        a = -1.
        afun = lambda lx: sub_logs(a, lx)
        b = 0.
        bfun = lambda lx: sub_logs(b, lx)
        lxm = -2.5
        lxp = -2.
        f = lambda ly, lx: np.exp(lx) * self.kernel.kernel_lx(lx, ly)
        btype = 0
        actual = self.kernel.kernel_integral((lxm, lxp), (-1., 0.), btype)
        expected, _ = dblquad(f, lxm, lxp, a, b)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 1
        actual = self.kernel.kernel_integral((lxm, lxp), (-1., 0.), btype)
        expected, _ = dblquad(f, lxm, lxp, afun, b)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 2
        actual = self.kernel.kernel_integral((lxm, lxp), (-1., 0.), btype)
        expected, _ = dblquad(f, lxm, lxp, a, bfun)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 3
        actual = self.kernel.kernel_integral((lxm, lxp), (-1., 0.), btype)
        expected, _ = dblquad(f, lxm, lxp, afun, bfun)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_kernel_integral_skips_close_x_bounds(self):
        lx1 = -3.
        lx2 = lx1 + 1.e-14
        actual = self.kernel.kernel_integral((lx1, lx2), (-1., 0.), btype=0)
        self.assertEqual(actual, 0.)
