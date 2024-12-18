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

# pylint: disable=protected-access, too-many-public-methods

"""Tests for collision_kernel module."""

import unittest

from scipy.integrate import dblquad

from bin_model.math_utils import add_logs, sub_logs, dilogarithm
from bin_model.basis import PolynomialOnInterval, \
    make_piecewise_polynomial_basis
from bin_model.constants import ModelConstants
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.collision_kernel import *
from bin_model.mass_grid import GeometricMassGrid

from .array_assert import ArrayTestCase

# Because this module requires specifying a large number of BoundType enums,
# give them short names here.
CNST = BoundType.CONSTANT
LOVA = BoundType.LOWER_VARIES
UPVA = BoundType.UPPER_VARIES
BOVA = BoundType.BOTH_VARY

class TestBeardV(unittest.TestCase):
    """
    Test beard_v function.
    """
    def setUp(self):
        """Set up constants for evaluating beard_v."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
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
        # pylint: disable-next=arguments-out-of-order
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))

    def test_sc_efficiency_low_diameter(self):
        """Test the Scott-Chen formula's extrapolation to small radii."""
        d1 = 5.e-6
        x = 0.7
        d2 = x * d1
        self.assertAlmostEqual(sc_efficiency(d1, d2),
                               sc_efficiency(20.e-6, x * 20.e-6))
        # pylint: disable-next=arguments-out-of-order
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))


class TestHallEfficiency(unittest.TestCase):
    """
    Test hall_efficiency function.
    """

    def _check_hall_efficiency_value(self, radius, ratio, expected):
        actual = hall_efficiency(2.e-6*radius, 2.e-6*radius*ratio)
        self.assertAlmostEqual(actual / expected, 1.)
        actual = hall_efficiency(2.e-6*radius*ratio, 2.e-6*radius)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_hall_efficiency(self):
        """Test the Hall efficiency for some exact table values."""
        self._check_hall_efficiency_value(300., 0.05, 0.97)
        self._check_hall_efficiency_value(300., 1., 1.)
        self._check_hall_efficiency_value(10., 0.05, 0.0001)
        self._check_hall_efficiency_value(10., 1., 0.027)
        self._check_hall_efficiency_value(60., 0.5, 0.91)
        self._check_hall_efficiency_value(30., 0.35, 0.27)
        self._check_hall_efficiency_value(50., 0.85, 0.92)

    def test_hall_efficiency_large(self):
        """Test the Hall efficiency for large values of collector radius."""
        self._check_hall_efficiency_value(301., 0.01, 1.)
        self._check_hall_efficiency_value(301., 0.05, 1.)
        self._check_hall_efficiency_value(301., 0.5, 1.)
        self._check_hall_efficiency_value(301., 1., 1.)

    def test_hall_efficiency_interp(self):
        """Test the Hall efficiency for interpolated values."""
        self._check_hall_efficiency_value(25., 0.675,
                                          (0.58+0.54+0.08+0.076)*0.25)
        self._check_hall_efficiency_value(62., 0.27,
                                          0.48*0.84+0.32*0.87+0.12*0.88+0.08*0.90)
        self._check_hall_efficiency_value(300., 0.07, 0.982)
        self._check_hall_efficiency_value(80., 0.05, 0.3)
        self._check_hall_efficiency_value(10., 0.725, 0.0375)
        self._check_hall_efficiency_value(65., 1., 3.5)

    def test_hall_efficiency_low_ratio(self):
        """Test the Hall efficiency for low radius ratios."""
        self._check_hall_efficiency_value(100., 0.02, 0.5)
        self._check_hall_efficiency_value(80., 0.02, 0.3)

    def test_hall_efficiency_small(self):
        """Test the Hall efficiency for small values of collector radius."""
        self._check_hall_efficiency_value(5., 0.2, 0.014)
        self._check_hall_efficiency_value(5., 0.725, 0.0375)
        self._check_hall_efficiency_value(5., 0.02, 0.0001)


class TestRegionUtils(unittest.TestCase):
    """
    Test methods that map out and decompose integration regions.
    """

    # pylint: disable-next=too-many-arguments
    def lxs_btypes_assert(self, lx_bound, ly_bound, lz_bound, expected_lxs,
                          expected_btypes):
        """Check lxs and btypes for given bounds.

        Calls get_lxs_and_btypes and compares the result to the supplied
        expected_lxs and expected_btypes lists.

        Since expected_lxs will typically include the output of find_corners,
        the strings "bl", "tl", "br", and "tr" can be part of the list, and if
        present will be substituted with the appropriate find_corners output.

        """
        bl, tl, br, tr = find_corners(ly_bound, lz_bound)
        replace_dict = {"bl": bl, "tl": tl, "br": br, "tr": tr}
        expected_lxs = [replace_dict[lx] if lx in replace_dict else lx
                        for lx in expected_lxs]
        actual_lxs, actual_btypes = \
            get_lxs_and_btypes(lx_bound, ly_bound, lz_bound)
        self.assertListEqual(actual_lxs, expected_lxs)
        self.assertListEqual(actual_btypes, expected_btypes)

    def test_find_corners_invalid(self):
        """Check find_corners raises an error given bad y or z bound values."""
        bad_bound = (1., 0.)
        good_bound = (2., 3.)
        with self.assertRaises(ValueError):
            find_corners(bad_bound, good_bound)
        with self.assertRaises(ValueError):
            find_corners(good_bound, bad_bound)

    def test_find_corners_bins_separated(self):
        """Check find_corners works when y and z bins do not overlap."""
        ly_bound = (0., 1.)
        lz_bound = (2., 3.)
        expected = (sub_logs(lz_bound[0], ly_bound[0]),
                    sub_logs(lz_bound[0], ly_bound[1]),
                    sub_logs(lz_bound[1], ly_bound[0]),
                    sub_logs(lz_bound[1], ly_bound[1]))
        actual = find_corners(ly_bound, lz_bound)
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_bins_overlap(self):
        """Check find_corners works when y and z bins overlap."""
        lz_bound = (1., 3.)
        for ly_bound in ((0., 1.), (0., 2.)):
            expected = (sub_logs(lz_bound[0], ly_bound[0]),
                        None,
                        sub_logs(lz_bound[1], ly_bound[0]),
                        sub_logs(lz_bound[1], ly_bound[1]))
            actual = find_corners(ly_bound, lz_bound)
            self.assertAlmostEqual(actual, expected)

    def test_find_corners_y_encloses_z(self):
        """Check find_corners works when z bin is a subset of the y bin."""
        lz_bound = (1., 3.)
        for ly_bound in ((0., 3.), (0., 4.)):
            expected = (sub_logs(lz_bound[0], ly_bound[0]),
                        None,
                        sub_logs(lz_bound[1], ly_bound[0]),
                        None)
            actual = find_corners(ly_bound, lz_bound)
            self.assertAlmostEqual(actual, expected)

    def test_find_corners_z_encloses_y(self):
        """Check find_corners works when y bin is a subset of the z bin."""
        lz_bound = (0., 3.)
        for ly_bound in ((0., 2.), (1., 2.)):
            expected = (None,
                        None,
                        sub_logs(lz_bound[1], ly_bound[0]),
                        sub_logs(lz_bound[1], ly_bound[1]))
            actual = find_corners(ly_bound, lz_bound)
            self.assertAlmostEqual(actual, expected)

    def test_find_corners_empty_set(self):
        """Check find_corners works when y bin is larger than the z bin."""
        expected = (None, None,
                    None, None)
        lz_bound = (0., 1.)
        for ly_bound in ((1., 3.), (2., 3.)):
            actual = find_corners(ly_bound, lz_bound)
            self.assertAlmostEqual(actual, expected)

    def test_min_max_ly(self):
        """Check that min_max_ly gives the right limits for each btype."""
        lx_bound = (0., 1.)
        y_bound_p = (2., 3.)
        y_bound_p_vary = (sub_logs(y_bound_p[0], lx_bound[1]),
                          sub_logs(y_bound_p[1], lx_bound[0]))
        btypes = [CNST, LOVA, UPVA, BOVA]
        expecteds = [y_bound_p,
                     (y_bound_p_vary[0], y_bound_p[1]),
                     (y_bound_p[0], y_bound_p_vary[1]),
                     y_bound_p_vary]
        for btype, expected in zip(btypes, expecteds):
            actual = min_max_ly(lx_bound, y_bound_p,
                                btype=btype)
            self.assertEqual(actual, expected)

    def test_get_lxs_and_btypes_assertions(self):
        """Check that get_lxs_and_btypes raises ValueError for bad bounds."""
        bad_bound = (0., -1.)
        good_bound = (0., 1.)
        with self.assertRaises(ValueError):
            get_lxs_and_btypes(bad_bound, good_bound, good_bound)
        with self.assertRaises(ValueError):
            get_lxs_and_btypes(good_bound, bad_bound, good_bound)
        with self.assertRaises(ValueError):
            get_lxs_and_btypes(good_bound, good_bound, bad_bound)

    def test_get_lxs_and_btypes_wide_lx(self):
        """Check get_lxs_and_btypes when lx range is too wide to matter."""
        lx_bound = (-20., 20.)
        ly_bound = (0., 1.)
        # Case where lz has a large range well-separated from y bin.
        lz_bound = (2., 3.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               ["tl", "bl", "tr", "br"],
                               [LOVA, CNST, UPVA])
        # Case where lz has a narrow range close to y bin, leading to a narrow
        # region of integration.
        lz_bound = (1.01, 1.02)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               ["tl", "tr", "bl", "br"],
                               [LOVA, BOVA, UPVA])

    def test_get_lxs_and_btypes_no_overlap(self):
        """Check get_lxs_and_btypes when x+y puts no particles in the z bin."""
        ly_bound = (0., 1.)
        lz_bound = (2., 3.)
        for lx_bound in ((19., 20.), (-20., -19.)):
            self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                                   [],
                                   [])

    def test_get_lxs_and_btypes_lower_lx(self):
        """Check get_lxs_and_btypes affected by lower lx bound.

        In these cases, the lower lx bound truncates the region from the left.
        """
        ly_bound = (0., 1.)
        lz_bound = (1.01, 2.)
        # As lx lower bound increases, more and more of the region is cut off.
        for i, lx_bound in enumerate(((0.5, 20.), (1., 20.), (1.7, 20.))):
            self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                                   [lx_bound[0]] + ["bl", "tr", "br"][i:],
                                   [LOVA, CNST, UPVA][i:])

    def test_get_lxs_and_btypes_upper_lx(self):
        """Check get_lxs_and_btypes affected by upper lx bound.

        In these cases, the upper lx bound truncates the region from the right.
        """
        ly_bound = (0., 1.)
        lz_bound = (1.01, 2.)
        # As lx upper bound decreases, more and more of the region is cut off.
        for i, lx_bound in enumerate(((-20., 1.7), (-20., 1.), (-20., 0.5))):
            self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                                   ["tl", "bl", "tr"][:3-i] + [lx_bound[1]],
                                   [LOVA, CNST, UPVA][:3-i])

    def test_get_lxs_and_btypes_missing_corners(self):
        """Check get_lxs_and_btypes with missing integration region corners.

        In these cases, some of the corners go to negative infinity.
        """
        lx_bound = (-20., 20.)
        ly_bound = (0., 1.)
        # No top left corner.
        lz_bound = (1., 2.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               [lx_bound[0], "bl", "tr", "br"],
                               [LOVA, CNST, UPVA])
        # No bottom left corner.
        lz_bound = (0., 2.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               [lx_bound[0], "tr", "br"],
                               [CNST, UPVA])
        # No top right corner.
        lz_bound = (0.5, 1.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               [lx_bound[0], "bl", "br"],
                               [BOVA, UPVA])
        # No bottom left or top right corner.
        lz_bound = (0., 1.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               [lx_bound[0], "br"],
                               [UPVA])
        # No bottom left or top right corner, upper x bound matters.
        lx_bound_lowupper = (-20, 0.)
        lz_bound = (0., 1.)
        self.lxs_btypes_assert(lx_bound_lowupper, ly_bound, lz_bound,
                               list(lx_bound_lowupper),
                               [UPVA])
        # No region at all.
        lz_bound = (-2., -1.)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               [],
                               [])

    def test_get_lxs_and_btypes_infinite_lz_upper(self):
        """Check get_lxs_and_btypes when upper lz bound is infinity."""
        # Test specifying an infinite upper bound for lz.
        lx_bound = (-20., 20.)
        ly_bound = (0., 1.)
        lz_bound = (2., np.inf)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               ["tl", "bl", lx_bound[1]],
                               [LOVA, CNST])
        # Test specifying an infinite upper bound for lz when the lower bound
        # is the same as ly's.
        lx_bound = (-2., -1.)
        lz_bound = (0., np.inf)
        self.lxs_btypes_assert(lx_bound, ly_bound, lz_bound,
                               list(lx_bound),
                               [CNST])

    def test_get_y_bound_p(self):
        """Check get_y_bound_p for all BoundType values."""
        btypes = [CNST, LOVA, UPVA, BOVA]
        ly_bound = (0., 1.)
        lz_bound = (2., 3.)
        y_bound_p = get_y_bound_p(ly_bound, lz_bound, btypes)
        self.assertListEqual(list(y_bound_p[:,0]), [0., 2., 0., 2.])
        self.assertListEqual(list(y_bound_p[:,1]), [1., 1., 3., 3.])


def reference_long_cloud(kc, lx_bound, y_bound_p):
    r"""Reference implementation of double integral used for Long's kernel.

    The definite integral being computed is:

    \int_{lxm}^{lxp} \int_{a}^{log(e^b-e^x)} (e^{2x - y} + e^{y}) dy dx
    """
    lxm, lxp = lx_bound
    a, b = y_bound_p
    return kc * (np.exp(b) * np.log((np.exp(b)-np.exp(lxp))
                                    / (np.exp(b)-np.exp(lxm)))
                 + np.exp(-a) / 2. * (np.exp(2.*lxp)-np.exp(2.*lxm))
                 + (np.exp(b)-np.exp(a))*(lxp-lxm))

def reference_long_rain(kr, lx_bound, y_bound_p):
    r"""Reference implementation of double integral used for Long's kernel.

    The definite integral being computed is:

    \int_{lxm}^{lxp} \int_{a}^{log(e^b-e^x)} (e^{x - y} + 1) dy dx
    """
    lxm, lxp = lx_bound
    a, b = y_bound_p
    return kr * (np.log((np.exp(b)-np.exp(lxp))/(np.exp(b)-np.exp(lxm)))
                 + np.exp(-a) * (np.exp(lxp)-np.exp(lxm))
                 - dilogarithm(np.exp(lxp-b)) + dilogarithm(np.exp(lxm-b))
                 + (b-a)*(lxp-lxm))

class TestReferenceLong(unittest.TestCase):
    """
    Test reference functions that will be used to test LongKernel.
    """

    def test_reference_long_cloud(self):
        """Check reference_long_cloud against numerical integration."""
        kc = 2.
        def integrand(y, x):
            return kc * (np.exp(2.*x - y) + np.exp(y))
        lx_bound = (-1., 0.)
        y_bound_p = (0., 1.)
        def hfun(x):
            return np.log(np.exp(y_bound_p[1])-np.exp(x))
        expected, _ = dblquad(integrand, lx_bound[0], lx_bound[1],
                              y_bound_p[0], hfun, epsabs=1.e-13)
        actual = reference_long_cloud(kc, lx_bound, y_bound_p)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_reference_long_rain(self):
        """Check reference_long_rain against numerical integration."""
        kr = 2.
        def integrand(y, x):
            return kr * (np.exp(x - y) + 1.)
        lx_bound = (-1., 0.)
        y_bound_p = (0., 1.)
        def hfun(x):
            return np.log(np.exp(y_bound_p[1])-np.exp(x))
        expected, _ = dblquad(integrand, lx_bound[0], lx_bound[1],
                              y_bound_p[0], hfun, epsabs=1.e-13)
        actual = reference_long_rain(kr, lx_bound, y_bound_p)
        self.assertAlmostEqual(actual, expected, places=12)


class TestCoalescenceKernel(ArrayTestCase):
    """
    Tests of CoalescenceKernel.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        # Mass-doubling grid.
        self.md_grid = GeometricMassGrid(self.constants,
                                         d_min=1.e-6,
                                         d_max=2.e-6,
                                         num_bins=3)
        self.basis = make_piecewise_polynomial_basis(self.md_grid, 1)

    def test_construct_sparsity_pattern(self):
        """Check construct_sparsity_pattern for a simple mass-doubling grid."""
        grid = self.md_grid
        idxs, nums, max_num = \
            CoalescenceKernel.construct_sparsity_structure(self.basis,
                                                           self.md_grid)
        expected_idxs = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 3],
        ])
        expected_nums = np.array([
            [1, 2, 2],
            [2, 1, 2],
            [2, 2, 1],
        ])
        expected_max_num = 2
        self.assertArrayEqual(idxs, np.tile(expected_idxs, (2, 2)))
        self.assertArrayEqual(nums, np.tile(expected_nums, (2, 2)))
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_closed_boundary(self):
        """Check construct_sparsity_pattern for a "closed" upper boundary."""
        grid = self.md_grid
        idxs, nums, max_num = \
            CoalescenceKernel.construct_sparsity_structure(self.basis,
                                                           self.md_grid,
                                                           boundary='closed')
        expected_idxs = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ])
        expected_nums = np.array([
            [1, 2, 1],
            [2, 1, 1],
            [1, 1, 1],
        ])
        expected_max_num = 2
        self.assertArrayEqual(idxs, np.tile(expected_idxs, (2, 2)))
        self.assertArrayEqual(nums, np.tile(expected_nums, (2, 2)))
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_invalid_boundary_raises(self):
        """Check construct_sparsity_pattern error for invalid boundary."""
        with self.assertRaises(ValueError):
            CoalescenceKernel.construct_sparsity_structure(self.basis,
                                                           self.md_grid,
                                                           boundary='nonsense')


class TestLongKernelInit(unittest.TestCase):
    """
    Tests of LongKernel constructor.
    """

    def setUp(self):
        """Create a ModelConstants for testing LongKernel."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)

    def test_fail_if_two_kcs(self):
        """Check that LongKernel.__init__ can only accept one kc setting."""
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc_cgs=1., kc_si=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc_cgs=1., kc=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kc=1., kc_si=1.)

    def test_fail_if_two_krs(self):
        """Check that LongKernel.__init__ can only accept one kr setting."""
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr_cgs=1., kr_si=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr_cgs=1., kr=1.)
        with self.assertRaises(RuntimeError):
            LongKernel(self.constants, kr=1., kr_si=1.)

    def test_kc(self):
        """Check setting kc directly."""
        ckern = LongKernel(self.constants, kc=3.)
        self.assertEqual(ckern.kc, 3.)

    def test_kc_cgs(self):
        """Check setting kc using cgs units."""
        ckern = LongKernel(self.constants, kc_cgs=3.)
        expected = 3. * self.constants.std_mass**2
        self.assertAlmostEqual(ckern.kc, expected, places=25)

    def test_kc_si(self):
        """Check setting kc using SI units."""
        ckern1 = LongKernel(self.constants, kc_cgs=3.)
        ckern2 = LongKernel(self.constants, kc_si=3.)
        self.assertEqual(ckern1.kc, ckern2.kc)

    def test_kc_default(self):
        """Check default value of kc."""
        ckern1 = LongKernel(self.constants)
        ckern2 = LongKernel(self.constants, kc_cgs=9.44e9)
        self.assertAlmostEqual(ckern1.kc, ckern2.kc, places=15)

    def test_kr(self):
        """Check setting kr directly."""
        ckern = LongKernel(self.constants, kr=3.)
        self.assertEqual(ckern.kr, 3.)

    def test_kr_cgs(self):
        """Check setting kr using cgs units."""
        ckern1 = LongKernel(self.constants, kr_cgs=3.)
        ckern2 = LongKernel(self.constants, kr_si=3.e-3)
        self.assertAlmostEqual(ckern1.kr, ckern2.kr, places=15)

    def test_kr_si(self):
        """Check setting kc using SI units."""
        ckern = LongKernel(self.constants, kr_si=3.)
        expected = 3. * self.constants.std_mass
        self.assertAlmostEqual(ckern.kr, expected, places=15)

    def test_kr_default(self):
        """Check default value of kr."""
        ckern1 = LongKernel(self.constants)
        ckern2 = LongKernel(self.constants, kr_cgs=5.78e3)
        self.assertAlmostEqual(ckern1.kr, ckern2.kr, places=15)

    def test_log_rain_m(self):
        """Check setting cloud-rain threshold mass directly."""
        ckern = LongKernel(self.constants, rain_m=0.8)
        self.assertAlmostEqual(ckern.log_rain_m, np.log(0.8))

    def test_log_rain_m_default(self):
        """Check default value of cloud-rain threshold mass."""
        # Set this constants object to make sure LongKernel is not just copying
        # rain_m from ModelConstants (which was the original behavior).
        constants = ModelConstants(rho_water=1000.,
                                   rho_air=1.2,
                                   diameter_scale=1.e-4,
                                   rain_d=8.e-5)
        ckern = LongKernel(constants)
        # Original Long kernel value.
        self.assertAlmostEqual(ckern.log_rain_m, 0.)



class TestLongKernel(unittest.TestCase):
    """
    Tests of LongKernel methods.
    """

    def setUp(self):
        """Create a LongKernel for testing purposes."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.ckern = LongKernel(self.constants)

    def test_integral_cloud_constant(self):
        """Check _integral_cloud for the CONSTANT BoundType."""
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (-1., 0.)
        expected = reference_long_cloud(ckern.kc, lx_bound,
                                        (y_bound_p[0], 1.)) \
                    - reference_long_cloud(ckern.kc, lx_bound,
                                        (y_bound_p[1], 1.))
        actual = ckern._integral_cloud(lx_bound, y_bound_p, btype=CNST)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_lower_varies(self):
        """Check _integral_cloud for the LOWER_VARIES BoundType."""
        ckern = self.ckern
        lx_bound = (-3., -2.)
        y_bound_p = (-1., 0.)
        expected = -reference_long_cloud(ckern.kc, lx_bound,
                                         (y_bound_p[1], y_bound_p[0]))
        actual = ckern._integral_cloud(lx_bound, y_bound_p, btype=LOVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_upper_varies(self):
        """Check _integral_cloud for the UPPER_VARIES BoundType."""
        ckern = self.ckern
        lx_bound = (-2., -1.)
        y_bound_p = (-1., 0.)
        expected = reference_long_cloud(ckern.kc, lx_bound, y_bound_p)
        actual = ckern._integral_cloud(lx_bound, y_bound_p, btype=UPVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_both_vary(self):
        """Check _integral_cloud for the BOTH_VARY BoundType."""
        ckern = self.ckern
        lx_bound = (-2., -1.)
        y_bound_p = (-0.9, 0.)
        expected = reference_long_cloud(ckern.kc, lx_bound,
                                        (1., y_bound_p[1])) \
                    - reference_long_cloud(ckern.kc, lx_bound,
                                           (1., y_bound_p[0]))
        actual = ckern._integral_cloud(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_constant(self):
        """Check _integral_rain for the CONSTANT BoundType."""
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (-1., 0.)
        expected = reference_long_rain(ckern.kr, lx_bound,
                                       (y_bound_p[0], 1.)) \
                    - reference_long_rain(ckern.kr, lx_bound,
                                       (y_bound_p[1], 1.))
        actual = ckern._integral_rain(lx_bound, y_bound_p, btype=CNST)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_lower_varies(self):
        """Check _integral_rain for the LOWER_VARIES BoundType."""
        ckern = self.ckern
        lx_bound = (-3., -2.)
        y_bound_p = (-1., 0.)
        expected = -reference_long_rain(ckern.kr, lx_bound,
                                        (y_bound_p[1], y_bound_p[0]))
        actual = ckern._integral_rain(lx_bound, y_bound_p, btype=LOVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_upper_varies(self):
        """Check _integral_rain for the UPPER_VARIES BoundType."""
        ckern = self.ckern
        lx_bound = (-2., -1.)
        y_bound_p = (-1., 0.)
        expected = reference_long_rain(ckern.kr, lx_bound, y_bound_p)
        actual = ckern._integral_rain(lx_bound, y_bound_p, btype=UPVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_both_vary(self):
        """Check _integral_rain for the BOTH_VARY BoundType."""
        ckern = self.ckern
        lx_bound = (-3., -2.)
        y_bound_p = (-1., 0.)
        expected = reference_long_rain(ckern.kr, lx_bound,
                                        (1., y_bound_p[1])) \
                    - reference_long_rain(ckern.kr, lx_bound,
                                           (1., y_bound_p[0]))
        actual = ckern._integral_rain(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_kernel_integral_cloud(self):
        """Check kernel_integral for two cloud bins."""
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (-1., 0.)
        expected = ckern._integral_cloud(lx_bound, y_bound_p, btype=CNST)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=CNST)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_cloud_varying_btype(self):
        """Check kernel_integral using cloud formula with BoundType = BOTH_VARY.

        Specifically, this covers the case where the y_bound_p values are in the
        rain-sized range, but the relevant portions of the x and y bins are both
        pure cloud.
        """
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (add_logs(0., -2.), add_logs(0., -1.))
        expected = ckern._integral_cloud(lx_bound, y_bound_p, btype=BOVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_rain(self):
        """Check kernel_integral when at least one bin is purely rain-sized."""
        ckern = self.ckern
        # Cases where x bin is rain and y bin is cloud, x is cloud and y is
        # rain, and both bins are rain.
        lx_bounds = [(0., 1.), (-1., 0.), (0., 1.)]
        y_bound_ps = [(-1., 0.), (0., 1.), (0., 1.)]
        for lx_bound, y_bound_p in zip(lx_bounds, y_bound_ps):
            expected = ckern._integral_rain(lx_bound, y_bound_p, btype=CNST)
            actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=CNST)
            self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_lx_spans_rain_m(self):
        """Check kernel_integral, CONSTANT btype, x bin is part rain."""
        ckern = self.ckern
        lx_bound = (-1., 1.)
        y_bound_p = (-1., 0.)
        expected = \
            ckern._integral_cloud((lx_bound[0], ckern.log_rain_m),
                                   y_bound_p, btype=CNST)
        expected += \
            ckern._integral_rain((ckern.log_rain_m, lx_bound[1]),
                                  y_bound_p, btype=CNST)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=CNST)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_constant(self):
        """Check kernel_integral, CONSTANT btype, y bin is part rain."""
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (-1., 1.)
        expected = \
            ckern._integral_cloud(lx_bound, (y_bound_p[0], ckern.log_rain_m),
                                   btype=CNST)
        expected += \
            ckern._integral_rain(lx_bound, (ckern.log_rain_m, y_bound_p[1]),
                                  btype=CNST)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=CNST)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_lower_varies(self):
        """Check kernel_integral, LOWER_VARIES btype, y bin is part rain.

        This is the case where the y bounds do not cross the cloud-rain
        threshold mass.
        """
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (add_logs(-1., 0.), 1.)
        expected = \
            ckern._integral_cloud(lx_bound, (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_rain(lx_bound, (ckern.log_rain_m, y_bound_p[1]),
                                  btype=CNST)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=LOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_lower_crossing(self):
        """Check kernel_integral, region lower boundary crosses y=rain_m."""
        ckern = self.ckern
        lx_bound = (-1., 0.)
        y_bound_p = (add_logs(-1., 0.1), 1.)
        cross_x = sub_logs(y_bound_p[0], ckern.log_rain_m)
        expected = \
            ckern._integral_cloud((cross_x, lx_bound[1]),
                                   (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_rain((cross_x, lx_bound[1]),
                                  (ckern.log_rain_m, y_bound_p[1]),
                                  btype=CNST)
        expected += \
            ckern._integral_rain((lx_bound[0], cross_x), y_bound_p, btype=LOVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=LOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_upper_varies(self):
        """Check kernel_integral, UPPER_VARIES btype, y bin is part rain.

        This is the case where the y bounds do not cross the cloud-rain
        threshold mass.
        """
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (-1., add_logs(-0.5, 0.))
        expected = \
            ckern._integral_cloud(lx_bound, (y_bound_p[0], ckern.log_rain_m),
                                   btype=CNST)
        expected += \
            ckern._integral_rain(lx_bound, (ckern.log_rain_m, y_bound_p[1]),
                                  btype=UPVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=UPVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_upper_crossing(self):
        """Check kernel_integral, region upper boundary crosses y=rain_m."""
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (-1., add_logs(-0.5, -0.1))
        cross_x = sub_logs(y_bound_p[1], ckern.log_rain_m)
        expected = \
            ckern._integral_cloud((lx_bound[0], cross_x),
                                   (y_bound_p[0], ckern.log_rain_m),
                                   btype=CNST)
        expected += \
            ckern._integral_cloud((cross_x, lx_bound[1]),
                                   y_bound_p, btype=UPVA)
        expected += \
            ckern._integral_rain((lx_bound[0], cross_x),
                                  (ckern.log_rain_m, y_bound_p[1]),
                                  btype=UPVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=UPVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_both_vary(self):
        """Check kernel_integral, BOTH_VARY btype, y bin is part rain.

        This is the case where the y bounds do not cross the cloud-rain
        threshold mass.
        """
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (add_logs(-1.5, 0.), add_logs(-0.5, 0.))
        expected = \
            ckern._integral_cloud(lx_bound, (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_rain(lx_bound, (ckern.log_rain_m, y_bound_p[1]),
                                  btype=UPVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_lower_crossing_both_vary(self):
        """Check kernel_integral, lower crosses y=rain_m, upper varies."""
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (add_logs(-1.5, 0.05), add_logs(-0.5, 0.))
        cross_x = sub_logs(y_bound_p[0], ckern.log_rain_m)
        expected = \
            ckern._integral_cloud((cross_x, lx_bound[1]),
                                   (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_rain((cross_x, lx_bound[1]),
                                  (ckern.log_rain_m, y_bound_p[1]),
                                  btype=UPVA)
        expected += \
            ckern._integral_rain((lx_bound[0], cross_x),
                                  y_bound_p, btype=BOVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_upper_crossing_both_vary(self):
        """Check kernel_integral, upper crosses y=rain_m, lower varies."""
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (add_logs(-1.5, 0.), add_logs(-0.5, -0.1))
        cross_x = sub_logs(y_bound_p[1], ckern.log_rain_m)
        expected = \
            ckern._integral_cloud((lx_bound[0], cross_x),
                                   (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_cloud((cross_x, lx_bound[1]),
                                   y_bound_p, btype=BOVA)
        expected += \
            ckern._integral_rain((lx_bound[0], cross_x),
                                  (ckern.log_rain_m, y_bound_p[1]), btype=UPVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_both_crossing(self):
        """Check kernel_integral, upper and lower cross y=rain_m."""
        ckern = self.ckern
        lx_bound = (-1.5, -0.5)
        y_bound_p = (add_logs(-1.5, 0.05), add_logs(-0.5, -0.1))
        x_low = sub_logs(y_bound_p[0], ckern.log_rain_m)
        x_high = sub_logs(y_bound_p[1], ckern.log_rain_m)
        expected = \
            ckern._integral_cloud((x_low, x_high),
                                   (y_bound_p[0], ckern.log_rain_m),
                                   btype=LOVA)
        expected += \
            ckern._integral_cloud((x_high, lx_bound[1]), y_bound_p, btype=BOVA)
        expected += \
            ckern._integral_rain((x_low, x_high),
                                  (ckern.log_rain_m, y_bound_p[1]), btype=UPVA)
        expected += \
            ckern._integral_rain((lx_bound[0], x_low), y_bound_p, btype=BOVA)
        actual = ckern.kernel_integral_deg0(lx_bound, y_bound_p, btype=BOVA)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_integrate_over_bins(self):
        """Check integrate_over_bins output.

        This test checks against the answer derived from an earlier version of
        this code; it is therefore only useful as an integration test to check
        for regressions.
        """
        ckern = self.ckern
        basis_x = PolynomialOnInterval(-1., 0., 0)
        basis_y = PolynomialOnInterval(-1., 0., 0)
        lz_bound = (0., 1.)
        actual = ckern.integrate_over_bins(basis_x, basis_y, lz_bound)
        expected = 3.1444320613930285e-09
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_degree_0(self):
        """Check kernel_integral vs. kernel_integral_deg0."""
        lx_bound = (-2.5, -2.)
        y_bound_p = (-1., 0.)
        basis_x = PolynomialOnInterval(lx_bound[0], lx_bound[1], 0)
        basis_y = PolynomialOnInterval(y_bound_p[0], y_bound_p[1], 0)
        actual = self.ckern.kernel_integral(basis_x, basis_y, lx_bound, y_bound_p, CNST)
        expected = self.ckern.kernel_integral_deg0(lx_bound, y_bound_p, CNST)
        self.assertEqual(actual, expected)

    def test_kernel_integral_higher_degree(self):
        """Check kernel_integral vs. numerical integration for higher degree."""
        lx_bound = (-2.5, -2.)
        y_bound_p = (-1., 0.)
        deg_lx = 1
        deg_ly = 2
        lx0 = -2.5
        ly0 = -1.
        scale_x = 1.8
        scale_y = 0.7
        basis_x = PolynomialOnInterval(lx_bound[0], lx_bound[1], deg_lx,
                                       root=lx0, scale=scale_x)
        basis_y = PolynomialOnInterval(y_bound_p[0], y_bound_p[1], deg_ly,
                                       root=ly0, scale=scale_y)
        def f(ly, lx):
            return (lx-lx0)**deg_lx * (ly-ly0)**deg_ly * self.ckern.kernel_x(np.exp(lx), np.exp(ly)) * np.exp(-ly)
        actual = self.ckern.kernel_integral(basis_x, basis_y, lx_bound, y_bound_p, CNST)
        expected, _ = dblquad(f, lx_bound[0], lx_bound[1], y_bound_p[0], y_bound_p[1])
        expected *= scale_x * scale_y
        self.assertAlmostEqual(actual / expected, 1.)


class TestMakeGolovinKernel(unittest.TestCase):
    """
    Test output of make_golovin_kernel.
    """

    def setUp(self):
        """Create a LongKernel for testing purposes."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)

    def test_golovin_vs_long(self):
        """Test Golovin kernel gives the same result as equivalent Long kernel."""
        ckern = make_golovin_kernel(self.constants)
        long_ckern = LongKernel(self.constants, rain_m=1.e-30, kr_si=6.e3)
        basis_x = PolynomialOnInterval(-1., 0., 0)
        basis_y = PolynomialOnInterval(-1., 0., 0)
        lz_bound = (0., 1.)
        actual = ckern.integrate_over_bins(basis_x, basis_y, lz_bound)
        expected = long_ckern.integrate_over_bins(basis_x, basis_y, lz_bound)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_golovin_vs_long_altered_b(self):
        """Test make_golovin_kernel handles the b argument correctly."""
        altered_b = 5.e3
        ckern = make_golovin_kernel(self.constants, b=altered_b)
        long_ckern = LongKernel(self.constants, rain_m=1.e-30, kr_si=altered_b)
        basis_x = PolynomialOnInterval(-1., 0., 0)
        basis_y = PolynomialOnInterval(-1., 0., 0)
        lz_bound = (0., 1.)
        actual = ckern.integrate_over_bins(basis_x, basis_y, lz_bound)
        expected = long_ckern.integrate_over_bins(basis_x, basis_y, lz_bound)
        self.assertAlmostEqual(actual, expected, places=20)


class TestHallKernel(unittest.TestCase):
    """
    Test HallKernel methods.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        self.ckern = HallKernel(self.constants)

    def test_bad_efficiency_string_raises_error(self):
        """Check error raised if initialized with a bad efficiency_string."""
        with self.assertRaises(ValueError):
            HallKernel(self.constants, 'nonsense')

    def test_efficiencies(self):
        """Check that the expected efficiency is being used by the kernel."""
        d1 = 10.e-6
        d2 = 100.e-6
        self.assertAlmostEqual(self.ckern.efficiency(d1, d2),
                               hall_efficiency(d1, d2))
        sc_ckern = HallKernel(self.constants, 'ScottChen')
        self.assertAlmostEqual(sc_ckern.efficiency(d1, d2),
                               sc_efficiency(d1, d2))

    def test_kernel_d(self):
        """Check kernel_d against the Hall kernel formula."""
        const = self.constants
        d1 = 10.e-6
        d2 = 100.e-6
        actual = self.ckern.kernel_d(d1, d2)
        expected = np.abs(beard_v(const, d1) - beard_v(const, d2))
        expected *= hall_efficiency(d1, d2)
        expected *= 0.25 * np.pi * (d1 + d2)**2
        self.assertAlmostEqual(actual / expected, 1.)
        # pylint: disable-next=arguments-out-of-order
        self.assertEqual(actual, self.ckern.kernel_d(d2, d1))

    def test_kernel_x(self):
        """Check that kernel_x correctly converts kernel_d output."""
        const = self.constants
        d1 = 10.e-6
        d2 = 100.e-6
        x1 = const.diameter_to_scaled_mass(d1)
        x2 = const.diameter_to_scaled_mass(d2)
        actual = self.ckern.kernel_x(x1, x2)
        expected = self.ckern.kernel_d(d1, d2)
        self.assertAlmostEqual(actual / expected, 1.)
        # pylint: disable-next=arguments-out-of-order
        self.assertEqual(actual, self.ckern.kernel_x(x2, x1))

    def test_kernel_integral(self):
        """Check kernel_integral vs. numerical integration, for all btypes."""
        lx_bound = (-2.5, -2.)
        y_bound_p = (-1., 0.)
        basis_x = PolynomialOnInterval(lx_bound[0], lx_bound[1], 0)
        basis_y = PolynomialOnInterval(-10., 10., 0)
        def f(ly, lx):
            return self.ckern.kernel_x(np.exp(lx), np.exp(ly)) * np.exp(-ly)
        def g(lx):
            return sub_logs(y_bound_p[0], lx)
        def h(lx):
            return sub_logs(y_bound_p[1], lx)
        btypes = [CNST, LOVA, UPVA, BOVA]
        gs = [y_bound_p[0], g, y_bound_p[0], g]
        hs = [y_bound_p[1], y_bound_p[1], h, h]
        for btype, g, h in zip(btypes, gs, hs):
            actual = self.ckern.kernel_integral(basis_x, basis_y, lx_bound, y_bound_p, btype)
            expected, _ = dblquad(f, lx_bound[0], lx_bound[1], g, h)
            self.assertAlmostEqual(actual / expected, 1.)

    def test_kernel_integral_higher_degree(self):
        """Check kernel_integral vs. numerical integration for higher degree."""
        lx_bound = (-2.5, -2.)
        y_bound_p = (-1., 0.)
        deg_lx = 1
        deg_ly = 2
        lx0 = -2.5
        ly0 = -1.
        scale_x = 1.8
        scale_y = 0.7
        basis_x = PolynomialOnInterval(lx_bound[0], lx_bound[1], deg_lx,
                                       root=lx0, scale=scale_x)
        basis_y = PolynomialOnInterval(y_bound_p[0], y_bound_p[1], deg_ly,
                                       root=ly0, scale=scale_y)
        def f(ly, lx):
            return (lx-lx0)**deg_lx * (ly-ly0)**deg_ly * self.ckern.kernel_x(np.exp(lx), np.exp(ly)) * np.exp(-ly)
        actual = self.ckern.kernel_integral(basis_x, basis_y, lx_bound, y_bound_p, CNST)
        expected, _ = dblquad(f, lx_bound[0], lx_bound[1], y_bound_p[0], y_bound_p[1])
        expected *= scale_x * scale_y
        self.assertAlmostEqual(actual / expected, 1.)

    def test_kernel_integral_skips_close_x_bounds(self):
        """Check kernel_integral returns 0 for extremely narrow x ranges."""
        lx1 = -3.
        lx2 = lx1 + 1.e-14
        basis_x = PolynomialOnInterval(-20., 20., 0)
        basis_y = PolynomialOnInterval(-20., 20., 0)
        actual = self.ckern.kernel_integral(basis_x, basis_y, (lx1, lx2), (-1., 0.), btype=CNST)
        self.assertEqual(actual, 0.)
