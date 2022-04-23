import unittest

from scipy.integrate import dblquad

from bin_model import *

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


class TestModelConstants(unittest.TestCase):
    """
    Test conversion methods on ModelConstants objects.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)

    def test_diameter_to_scaled_mass(self):
        self.assertAlmostEqual(self.constants.diameter_to_scaled_mass(1.e-5),
                               1.e-3)

    def test_scaled_mass_to_diameter(self):
        self.assertAlmostEqual(self.constants.scaled_mass_to_diameter(1.e-3),
                               1.e-5)

    def test_std_mass(self):
        self.assertAlmostEqual(self.constants.std_mass,
                               self.constants.rho_water * np.pi/6. * 1.e-12)

    def test_rain_m(self):
        self.assertEqual(self.constants.rain_m, 1.)


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
        with self.assertRaises(AssertionError):
            actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        ly1 = 0.
        ly2 = 1.
        lz1 = 3.
        lz2 = 2.
        with self.assertRaises(AssertionError):
            actual = kernel.find_corners(ly1, ly2, lz1, lz2)

    def test_find_corners_bins_separated(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), sub_logs(lz1, ly2),
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_bins_overlap(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 0.
        ly2 = 2.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_large_y2(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 3.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), None)
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 0.
        ly2 = 4.
        lz1 = 1.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), None,
                    sub_logs(lz2, ly1), None)
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_large_y1(self):
        kernel = self.kernel
        ly1 = 0.
        ly2 = 2.
        lz1 = 0.
        lz2 = 3.
        expected = (None, None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 1.
        ly2 = 2.
        lz1 = 0.
        lz2 = 3.
        expected = (None, None,
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)

    def test_find_corners_empty_set(self):
        kernel = self.kernel
        ly1 = 1.
        ly2 = 3.
        lz1 = 0.
        lz2 = 1.
        expected = (None, None,
                    None, None)
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)
        kernel = self.kernel
        ly1 = 2.
        ly2 = 3.
        lz1 = 0.
        lz2 = 1.
        expected = (None, None,
                    None, None)
        actual = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)

    def test_min_max_ly_asserts_valid_btype(self):
        kernel = self.kernel
        a = 2.
        b = 3.
        lxm = 0.
        lxp = 1.
        with self.assertRaises(AssertionError):
            kernel.min_max_ly(a, b, lxm, lxp, btype=-1)
        with self.assertRaises(AssertionError):
            kernel.min_max_ly(a, b, lxm, lxp, btype=4)

    def test_min_max_ly(self):
        kernel = self.kernel
        a = 2.
        b = 3.
        lxm = 0.
        lxp = 1.
        expected = (a, b)
        actual = kernel.min_max_ly(a, b, lxm, lxp, btype=0)
        self.assertEqual(actual, expected)
        expected = (sub_logs(a, lxp), b)
        actual = kernel.min_max_ly(a, b, lxm, lxp, btype=1)
        self.assertEqual(actual, expected)
        expected = (a, sub_logs(b, lxm))
        actual = kernel.min_max_ly(a, b, lxm, lxp, btype=2)
        self.assertEqual(actual, expected)
        expected = (sub_logs(a, lxp), sub_logs(b, lxm))
        actual = kernel.min_max_ly(a, b, lxm, lxp, btype=3)
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
        with self.assertRaises(AssertionError):
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        lx1 = 0.
        lx2 = 1.
        ly1 = 0.
        ly2 = -1.
        with self.assertRaises(AssertionError):
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        ly1 = 0.
        ly2 = 1.
        lz1 = 3.
        lz2 = 2.
        with self.assertRaises(AssertionError):
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)

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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [tl, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.01
        lz2 = 1.02
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [])
        self.assertListEqual(actual_btypes, [])
        lx1 = -20.
        lx2 = -19.
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # x bound large enough to cut off middle region
        lx1 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, tr, br])
        self.assertListEqual(actual_btypes, [0, 2])
        # x bound large enough to cut off right region
        lx1 = 1.7
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [tl, bl, tr, lx2])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # x small enough to cut off middle region
        lx2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, bl, tr, br])
        self.assertListEqual(actual_btypes, [1, 0, 2])
        # No bottom left corner.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, tr, br])
        self.assertListEqual(actual_btypes, [0, 2])
        # No top right corner.
        ly1 = 0.
        ly2 = 2.
        lz1 = 1.
        lz2 = 2.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, bl, br])
        self.assertListEqual(actual_btypes, [3, 2])
        # No bottom left or top right corner.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, br])
        self.assertListEqual(actual_btypes, [2])
        # No bottom left or top right corner, upper x bound matters.
        lx2 = 0.
        ly1 = 0.
        ly2 = 1.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, lx2])
        self.assertListEqual(actual_btypes, [2])
        # No region at all.
        ly1 = 1.
        ly2 = 2.
        lz1 = 0.
        lz2 = 1.
        actual_lxs, actual_btypes = \
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
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
            kernel.get_lxs_and_btypes(lx1, lx2, ly1, ly2, lz1, lz2)
        bl, tl, br, tr = kernel.find_corners(ly1, ly2, lz1, lz2)
        self.assertListEqual(actual_lxs, [lx1, lx2])
        self.assertListEqual(actual_btypes, [0])

    def test_get_ly_bounds(self):
        kernel = self.kernel
        btypes = [0, 1, 2, 3]
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        actual_as, actual_bs = kernel.get_ly_bounds(ly1, ly2, lz1, lz2, btypes)
        self.assertListEqual(actual_as, [0., 2., 0., 2.])
        self.assertListEqual(actual_bs, [1., 1., 3., 3.])

    def test_integrate_over_bins_raises(self):
        kernel = self.kernel
        lx1 = 0.
        lx2 = 1.
        ly1 = 0.
        ly2 = 1.
        lz1 = 1.
        lz2 = 2.
        with self.assertRaises(NotImplementedError):
            kernel.integrate_over_bins(lx1, lx2, ly1, ly2, lz1, lz2)


def reference_long_cloud(a, b, lxm, lxp):
    """Reference implementation of double integral used for Long's kernel.

    The definite integral being computed is:

    \int_{lxm}^{lxp} \int_{a}^{log(e^b-e^x)} (e^{2x - y} + e^{y}) dy dx
    """
    return np.exp(b) * np.log((np.exp(b)-np.exp(lxp))/(np.exp(b)-np.exp(lxm)))\
        + np.exp(-a) / 2. * (np.exp(2.*lxp)-np.exp(2.*lxm))\
        + (np.exp(b)-np.exp(a))*(lxp-lxm)

def reference_long_rain(a, b, lxm, lxp):
    """Reference implementation of double integral used for Long's kernel.

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
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kc_cgs=1., kc_si=1.)

    def test_fail_if_two_krs(self):
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kr_cgs=1., kr_si=1.)

    def test_kc_cgs(self):
        kernel = LongKernel(self.constants, kc_cgs=3.)
        expected = 3. * self.constants.rho_air * self.constants.std_mass**2
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
        expected = 3. * self.constants.rho_air * self.constants.std_mass
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
        with self.assertRaises(AssertionError):
            actual = kernel._integral_cloud(a, b, lxm, lxp, btype=-1)
        with self.assertRaises(AssertionError):
            actual = kernel._integral_cloud(a, b, lxm, lxp, btype=4)

    def test_integral_cloud_type_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = reference_long_cloud(a, 1., lxm, lxp) \
                    - reference_long_cloud(b, 1., lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_cloud(b, a, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_cloud(a, b, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_cloud_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_cloud(1., b, lxm, lxp) \
                    - reference_long_cloud(1., a, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_btype_bounds(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        with self.assertRaises(AssertionError):
            actual = kernel._integral_rain(a, b, lxm, lxp, btype=-1)
        with self.assertRaises(AssertionError):
            actual = kernel._integral_rain(a, b, lxm, lxp, btype=4)

    def test_integral_rain_type_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = reference_long_rain(a, 1., lxm, lxp) \
                    - reference_long_rain(b, 1., lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_rain(b, a, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_rain(a, b, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_integral_rain_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_rain(1., b, lxm, lxp) \
                    - reference_long_rain(1., a, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_kernel_integral_cloud(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 0.
        expected = kernel.kc * kernel._integral_cloud(a, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_cloud_btype_3(self):
        kernel = LongKernel(self.constants)
        a = add_logs(0., -2.)
        b = add_logs(0., -1.)
        lxm = -1.
        lxp = 0.
        expected = kernel.kc * kernel._integral_cloud(a, b, lxm, lxp, btype=3)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_rain(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = 0.
        lxp = 1.
        expected = kernel.kr * kernel._integral_rain(a, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)
        kernel = LongKernel(self.constants)
        a = 0.
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = kernel.kr * kernel._integral_rain(a, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)
        kernel = LongKernel(self.constants)
        a = 0.
        b = 1.
        lxm = 0.
        lxp = 1.
        expected = kernel.kr * kernel._integral_rain(a, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_lx_spans_rain_m(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -1.
        lxp = 1.
        expected = kernel.kc * \
            kernel._integral_cloud(a, b, lxm, kernel.log_rain_m, btype=0)
        expected += kernel.kr * \
            kernel._integral_rain(a, b, kernel.log_rain_m, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_0(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, lxm, lxp, btype=0)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=0)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_1_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1., 0.)
        b = 1.
        lxm = -1.
        lxp = 0.
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, lxm, lxp, btype=1)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, lxm, lxp, btype=0)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_1_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1., 0.1)
        b = 1.
        lxm = -1.
        lxp = 0.
        cross_x = sub_logs(a, kernel.log_rain_m)
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, cross_x, lxp, btype=1)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, cross_x, lxp, btype=0)
        expected += kernel.kr * \
            kernel._integral_rain(a, b, lxm, cross_x, btype=1)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_2_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, lxm, lxp, btype=0)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, lxm, lxp, btype=2)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_2_crossing(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(b, kernel.log_rain_m)
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, lxm, cross_x, btype=0)
        expected += kernel.kc * \
            kernel._integral_cloud(a, b, cross_x, lxp, btype=2)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, lxm, cross_x, btype=2)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_no_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.)
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m, lxm, lxp, btype=1)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b, lxm, lxp, btype=2)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_bot_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.05)
        b = add_logs(-0.5, 0.)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(a, kernel.log_rain_m)
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m,
                                   cross_x, lxp, btype=1)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b,
                                  cross_x, lxp, btype=2)
        expected += kernel.kr * \
            kernel._integral_rain(a, b,
                                  lxm, cross_x, btype=3)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_top_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.)
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        cross_x = sub_logs(b, kernel.log_rain_m)
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m,
                                   lxm, cross_x, btype=1)
        expected += kernel.kc * \
            kernel._integral_cloud(a, b,
                                   cross_x, lxp, btype=3)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b,
                                  lxm, cross_x, btype=2)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=20)

    def test_kernel_integral_ly_spans_rain_m_btype_3_both_crossing(self):
        kernel = LongKernel(self.constants)
        a = add_logs(-1.5, 0.05)
        b = add_logs(-0.5, -0.1)
        lxm = -1.5
        lxp = -0.5
        x_low = sub_logs(a, kernel.log_rain_m)
        x_high = sub_logs(b, kernel.log_rain_m)
        expected = kernel.kc * \
            kernel._integral_cloud(a, kernel.log_rain_m,
                                   x_low, x_high, btype=1)
        expected += kernel.kc * \
            kernel._integral_cloud(a, b,
                                   x_high, lxp, btype=3)
        expected += kernel.kr * \
            kernel._integral_rain(kernel.log_rain_m, b,
                                  x_low, x_high, btype=2)
        expected += kernel.kr * \
            kernel._integral_rain(a, b,
                                  lxm, x_low, btype=3)
        actual = kernel.kernel_integral(a, b, lxm, lxp, btype=3)
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
        actual = kernel.integrate_over_bins(lx1, lx2, ly1, ly2, lz1, lz2)
        expected = 3.773318473671634e-09
        self.assertAlmostEqual(actual, expected, places=20)


class TestMassGrid(unittest.TestCase):
    """
    Tests of MassGrid methods and attributes.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)
        # This will put lx bin boundaries at 10^-6, 10^-5, ..., 10^3.
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=9)
        # Mass-doubling grid.
        self.md_grid = GeometricMassGrid(self.constants,
                                         d_min=1.e-6,
                                         d_max=2.e-6,
                                         num_bins=3)
        # Fine grid.
        self.fine_grid = GeometricMassGrid(self.constants,
                                           d_min=1.e-6,
                                           d_max=2.e-6,
                                           num_bins=6)

    def test_find_bin(self):
        for i in range(9):
            lx = np.log(10.**(-5.5+i))
            self.assertEqual(self.grid.find_bin(lx), i)

    def test_find_bin_lower_edge(self):
        lx = np.log(10.**-6.5)
        self.assertEqual(self.grid.find_bin(lx), -1)
        lx = np.log(10.**-10)
        self.assertEqual(self.grid.find_bin(lx), -1)

    def test_find_bin_upper_edge(self):
        lx = np.log(10.**3.5)
        self.assertEqual(self.grid.find_bin(lx), 9)
        lx = np.log(10.**10)
        self.assertEqual(self.grid.find_bin(lx), 9)

    def test_find_sum_bins(self):
        # Note that this may not be sufficient to capture non-geometric cases,
        # since geometrically-spaced grids should always have num_sum_bins
        # equal to 2 (or maybe 1).
        grid = self.grid
        lx1 = grid.bin_bounds[0]
        lx2 = grid.bin_bounds[1]
        ly1 = grid.bin_bounds[0]
        ly2 = grid.bin_bounds[1]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 0)
        self.assertEqual(num, 2)
        ly1 = grid.bin_bounds[1]
        ly2 = grid.bin_bounds[2]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 1)
        self.assertEqual(num, 2)
        # Different results for grid with spacing dividing log(2) evenly.
        grid = self.md_grid
        lx1 = grid.bin_bounds[0]
        lx2 = grid.bin_bounds[1]
        ly1 = grid.bin_bounds[0]
        ly2 = grid.bin_bounds[1]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 1)
        self.assertEqual(num, 1)
        ly1 = grid.bin_bounds[1]
        ly2 = grid.bin_bounds[2]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 1)
        self.assertEqual(num, 2)

    def test_find_sum_bins_upper_range(self):
        grid = self.md_grid
        lx1 = grid.bin_bounds[2]
        lx2 = grid.bin_bounds[3]
        ly1 = grid.bin_bounds[2]
        ly2 = grid.bin_bounds[3]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 3)
        self.assertEqual(num, 1)
        lx1 = grid.bin_bounds[1]
        lx2 = grid.bin_bounds[2]
        ly1 = grid.bin_bounds[2]
        ly2 = grid.bin_bounds[3]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 2)
        self.assertEqual(num, 2)

    def test_find_sum_bins_fine(self):
        grid = self.fine_grid
        lx1 = grid.bin_bounds[5]
        lx2 = grid.bin_bounds[6]
        ly1 = grid.bin_bounds[5]
        ly2 = grid.bin_bounds[6]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 6)
        self.assertEqual(num, 1)

    def test_construct_sparsity_pattern(self):
        grid = self.md_grid
        idxs, nums, max_num = grid.construct_sparsity_structure()
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
        self.assertEqual(idxs.shape, expected_idxs.shape)
        for i in range(len(idxs.flat)):
            self.assertEqual(idxs.flat[i], expected_idxs.flat[i])
        self.assertEqual(nums.shape, expected_nums.shape)
        for i in range(len(nums.flat)):
            self.assertEqual(nums.flat[i], expected_nums.flat[i])
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_closed_boundary(self):
        grid = self.md_grid
        idxs, nums, max_num = \
            grid.construct_sparsity_structure(boundary='closed')
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
        self.assertEqual(idxs.shape, expected_idxs.shape)
        for i in range(len(idxs.flat)):
            self.assertEqual(idxs.flat[i], expected_idxs.flat[i])
        self.assertEqual(nums.shape, expected_nums.shape)
        for i in range(len(nums.flat)):
            self.assertEqual(nums.flat[i], expected_nums.flat[i])
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_invalid_boundary_raises(self):
        with self.assertRaises(AssertionError):
            self.grid.construct_sparsity_structure(boundary='nonsense')


class TestGeometricMassGrid(unittest.TestCase):
    """
    Tests of GeometricMassGrid methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)
        self.d_min = 1.e-6
        self.d_max = 1.e-3
        self.num_bins = 90
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=self.d_min,
                                      d_max=self.d_max,
                                      num_bins=self.num_bins)

    def test_geometric_grid_init_scalars(self):
        const = self.constants
        grid = self.grid
        self.assertEqual(grid.d_min, self.d_min)
        self.assertEqual(grid.d_max, self.d_max)
        self.assertEqual(grid.num_bins, self.num_bins)
        x_min = const.diameter_to_scaled_mass(self.d_min)
        self.assertAlmostEqual(grid.x_min, x_min)
        x_max = const.diameter_to_scaled_mass(self.d_max)
        self.assertAlmostEqual(grid.x_max, x_max)
        lx_min = np.log(x_min)
        self.assertAlmostEqual(grid.lx_min, lx_min)
        lx_max = np.log(x_max)
        self.assertAlmostEqual(grid.lx_max, lx_max)
        dlx = (lx_max - lx_min) / self.num_bins
        self.assertAlmostEqual(grid.dlx, dlx)

    def test_geometric_grid_init_arrays(self):
        const = self.constants
        grid = self.grid
        bin_bounds = np.linspace(grid.lx_min, grid.lx_max, self.num_bins+1)
        self.assertEqual(len(grid.bin_bounds), self.num_bins+1)
        for i in range(self.num_bins+1):
            self.assertAlmostEqual(grid.bin_bounds[i], bin_bounds[i])
        bin_bounds_d = np.array([const.scaled_mass_to_diameter(np.exp(b))
                                 for b in grid.bin_bounds])
        self.assertEqual(len(grid.bin_bounds_d), self.num_bins+1)
        for i in range(self.num_bins+1):
            self.assertAlmostEqual(grid.bin_bounds_d[i], bin_bounds_d[i])
        bin_widths = bin_bounds[1:] - bin_bounds[:-1]
        self.assertEqual(len(grid.bin_widths), self.num_bins)
        for i in range(self.num_bins):
            self.assertAlmostEqual(grid.bin_widths[i], bin_widths[i])


class TestKernelTensor(unittest.TestCase):
    """
    Tests of KernelTensor methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)
        self.kernel = LongKernel(self.constants)
        self.num_bins = 6
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_kernel_init(self):
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ktens = KernelTensor(self.kernel, self.grid)
        self.assertEqual(ktens.scaling, 1.)
        self.assertEqual(ktens.boundary, 'open')
        idxs, nums, max_num = self.grid.construct_sparsity_structure()
        self.assertEqual(ktens.idxs.shape, idxs.shape)
        for i in range(len(idxs.flat)):
            self.assertEqual(ktens.idxs.flat[i], idxs.flat[i])
        self.assertEqual(ktens.nums.shape, nums.shape)
        for i in range(len(nums.flat)):
            self.assertEqual(ktens.nums.flat[i], nums.flat[i])
        self.assertEqual(ktens.max_num, max_num)
        self.assertEqual(ktens.data.shape, (nb, nb, max_num))
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[0]
        ly2 = bb[1]
        lz1 = bb[2]
        lz2 = bb[3]
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[0,0,0], expected)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[1]
        ly2 = bb[2]
        lz1 = bb[3]
        lz2 = bb[4]
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[0,1,1], expected)
        expected = self.kernel.integrate_over_bins(ly1, ly2, lx1, lx2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[1,0,1], expected)
        lx1 = bb[5]
        lx2 = bb[6]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[5,5,0], expected)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = bb[6]
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[0,5,0], expected)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[0,5,1], expected)

    def test_kernel_init_scaling(self):
        ktens = KernelTensor(self.kernel, self.grid, scaling=2.)
        self.assertEqual(ktens.scaling, 2.)
        ktens_noscale = KernelTensor(self.kernel, self.grid)
        self.assertEqual(ktens.data.shape, ktens_noscale.data.shape)
        for i in range(len(ktens.data.flat)):
            self.assertAlmostEqual(ktens.data.flat[i],
                                   ktens_noscale.data.flat[i]/2.,
                                   places=25)

    def test_kernel_init_invalid_boundary_raises(self):
        with self.assertRaises(AssertionError):
            ktens = KernelTensor(self.kernel, self.grid, boundary='nonsense')

    def test_kernel_init_boundary(self):
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ktens = KernelTensor(self.kernel, self.grid, boundary='closed')
        self.assertEqual(ktens.boundary, 'closed')
        idxs, nums, max_num = \
            self.grid.construct_sparsity_structure(boundary='closed')
        self.assertEqual(ktens.idxs.shape, idxs.shape)
        for i in range(len(idxs.flat)):
            self.assertEqual(ktens.idxs.flat[i], idxs.flat[i])
        self.assertEqual(ktens.nums.shape, nums.shape)
        for i in range(len(nums.flat)):
            self.assertEqual(ktens.nums.flat[i], nums.flat[i])
        self.assertEqual(ktens.max_num, max_num)
        self.assertEqual(ktens.data.shape, (nb, nb, max_num))
        lx1 = bb[5]
        lx2 = bb[6]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[5,5,0], expected)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertEqual(ktens.data[0,5,0], expected)
