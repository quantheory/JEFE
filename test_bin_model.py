import unittest

from scipy.integrate import quad, dblquad

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

    def test_mass_conc_scale(self):
        self.assertEqual(self.constants.mass_conc_scale, 1.)

    def test_time_scale(self):
        self.assertEqual(self.constants.time_scale, 1.)


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
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kc_cgs=1., kc=1.)
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kc=1., kc_si=1.)

    def test_fail_if_two_krs(self):
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kr_cgs=1., kr_si=1.)
        with self.assertRaises(AssertionError):
            LongKernel(self.constants, kr_cgs=1., kr=1.)
        with self.assertRaises(AssertionError):
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
        expected = 3.1444320613930285e-09
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
        self.assertAlmostEqual(ktens.data[0,0,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[1]
        ly2 = bb[2]
        lz1 = bb[3]
        lz2 = bb[4]
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[0,1,1], expected / ktens.scaling)
        expected = self.kernel.integrate_over_bins(ly1, ly2, lx1, lx2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[1,0,1], expected / ktens.scaling)
        lx1 = bb[5]
        lx2 = bb[6]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[5,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = bb[6]
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[0,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[0,5,1], expected / ktens.scaling)

    def test_kernel_init_scaling(self):
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale = 2.,
                               time_scale=3.)
        kernel = LongKernel(const)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=2.e-6,
                                 num_bins=self.num_bins)
        ktens = KernelTensor(kernel, grid)
        self.assertEqual(ktens.scaling, const.std_mass
                             / (const.mass_conc_scale * const.time_scale))
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale=1.,
                               time_scale=1.)
        kernel = LongKernel(const)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=2.e-6,
                                 num_bins=self.num_bins)
        ktens_noscale = KernelTensor(kernel, grid)
        self.assertEqual(ktens.data.shape, ktens_noscale.data.shape)
        for i in range(len(ktens.data.flat)):
            self.assertAlmostEqual(ktens.data.flat[i],
                                   ktens_noscale.data.flat[i]
                                       * const.std_mass
                                       / ktens.scaling)

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
        self.assertAlmostEqual(ktens.data[5,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins(lx1, lx2, ly1, ly2,
                                                   lz1, lz2)
        self.assertAlmostEqual(ktens.data[0,5,0], expected / ktens.scaling)

    def test_calc_rate(self):
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        # Expected value in first bin.
        idx = 0
        expected_bot = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_bot -= ktens.data[idx,i,j] * f[idx] * f[i]
        expected_bot /= bw[idx]
        # Expected value in fourth bin.
        idx = 3
        expected_middle = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_middle -= ktens.data[idx,i,j] * f[idx] * f[i]
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                num = ktens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_middle += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_middle /= bw[idx]
        # Expected value in top bin.
        idx = 5
        expected_top = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_top -= ktens.data[idx,i,j] * f[idx] * f[i]
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                num = ktens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_top += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_top /= bw[idx]
        # Expected value in extra bin.
        idx = 6
        expected_extra = 0
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                num = ktens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_extra += ktens.data[i,j,k_idx] * f[i] * f[j]
        self.assertAlmostEqual(dfdt[0], expected_bot, places=15)
        self.assertAlmostEqual(dfdt[3], expected_middle, places=15)
        self.assertAlmostEqual(dfdt[5], expected_top, places=15)
        self.assertAlmostEqual(dfdt[6], expected_extra, places=15)
        mass_change = np.zeros((nb+1,))
        mass_change[:nb] = dfdt[:nb] * bw
        mass_change[-1] = dfdt[-1]
        self.assertAlmostEqual(np.sum(mass_change/mass_change.max()), 0.)

    def test_calc_rate_closed(self):
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.kernel, self.grid, boundary='closed')
        f = np.linspace(1., nb, nb)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        # Expected value in first bin.
        idx = 0
        expected_bot = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_bot -= ktens.data[idx,i,j] * f[idx] * f[i]
        expected_bot /= bw[idx]
        # Expected value in fourth bin.
        idx = 3
        expected_middle = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_middle -= ktens.data[idx,i,j] * f[idx] * f[i]
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                num = ktens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_middle += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_middle /= bw[idx]
        # Expected value in top bin.
        idx = 5
        expected_top = 0
        for i in range(nb-1):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                num = ktens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_top += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_top /= bw[idx]
        self.assertAlmostEqual(dfdt[0], expected_bot, places=25)
        self.assertAlmostEqual(dfdt[3], expected_middle, places=25)
        self.assertAlmostEqual(dfdt[5], expected_top, places=25)
        mass_change = dfdt * bw
        self.assertAlmostEqual(np.sum(mass_change/mass_change.max()), 0.)

    def test_calc_rate_valid_sizes(self):
        nb = self.num_bins
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(1., nb+2, nb+2)
        with self.assertRaises(AssertionError):
            ktens.calc_rate(f)
        with self.assertRaises(AssertionError):
            ktens.calc_rate(f[:nb-1])
        dfdt_1 = ktens.calc_rate(f[:nb+1])
        dfdt_2 = ktens.calc_rate(f[:nb])
        self.assertEqual(len(dfdt_1), nb+1)
        self.assertEqual(len(dfdt_2), nb)
        for i in range(nb):
            self.assertEqual(dfdt_1[i], dfdt_2[i])

    def test_calc_rate_no_closed_boundary_flux(self):
        nb = self.num_bins
        ktens = KernelTensor(self.kernel, self.grid, boundary='closed')
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(dfdt[-1], 0.)

    def test_calc_rate_valid_shapes(self):
        nb = self.num_bins
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        f_row = np.reshape(f, (1, nb+1))
        dfdt_row = ktens.calc_rate(f_row)
        self.assertEqual(dfdt_row.shape, f_row.shape)
        for i in range(nb+1):
            self.assertAlmostEqual(dfdt_row[0,i], dfdt[i], places=25)
        f_col = np.reshape(f, (nb+1, 1))
        dfdt_col = ktens.calc_rate(f_col)
        self.assertEqual(dfdt_col.shape, f_col.shape)
        for i in range(nb+1):
            self.assertAlmostEqual(dfdt_col[i,0], dfdt[i], places=25)
        f = np.linspace(1., nb, nb)
        dfdt = ktens.calc_rate(f)
        f_row = np.reshape(f, (1, nb))
        dfdt_row = ktens.calc_rate(f_row)
        self.assertEqual(dfdt_row.shape, f_row.shape)
        for i in range(nb):
            self.assertAlmostEqual(dfdt_row[0,i], dfdt[i], places=25)
        f_col = np.reshape(f, (nb, 1))
        dfdt_col = ktens.calc_rate(f_col)
        self.assertEqual(dfdt_col.shape, f_col.shape)
        for i in range(nb):
            self.assertAlmostEqual(dfdt_col[i,0], dfdt[i], places=25)

    def test_calc_rate_force_out_flux(self):
        nb = self.num_bins
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f[:nb], out_flux=True)
        expected = ktens.calc_rate(f[:nb+1])
        self.assertEqual(len(dfdt), nb+1)
        for i in range(nb+1):
            self.assertEqual(dfdt[i], expected[i])

    def test_calc_rate_force_no_out_flux(self):
        nb = self.num_bins
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f, out_flux=False)
        expected = ktens.calc_rate(f[:nb])
        self.assertEqual(len(dfdt), nb)
        for i in range(nb):
            self.assertEqual(dfdt[i], expected[i])

    def test_calc_rate_derivative(self):
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(2., nb+1, nb+1)
        dfdt, rate_deriv = ktens.calc_rate(f, derivative=True)
        self.assertEqual(rate_deriv.shape, (nb+1, nb+1))
        # First column, perturbation in lowest bin.
        idx = 0
        expected = np.zeros((nb+1,))
        # Effect of increased fluxes out of this bin.
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                this_rate = ktens.data[idx,i,j] * f[i]
                expected[idx] -= this_rate
                expected[ktens.idxs[idx,i] + j] += this_rate
        # Effect of increased transfer of mass out of other bins.
        # (Including this one; the double counting is correct.)
        for i in range(nb):
            for j in range(ktens.nums[i,idx]):
                this_rate = ktens.data[i,idx,j] * f[i]
                expected[i] -= this_rate
                expected[ktens.idxs[i,idx] + j] += this_rate
        expected[:nb] /= bw
        for i in range(nb+1):
            self.assertAlmostEqual(rate_deriv[i,idx], expected[i], places=25)
        # Last column, perturbation in highest bin.
        idx = 5
        expected = np.zeros((nb+1,))
        # Effect of increased fluxes out of this bin.
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                this_rate = ktens.data[idx,i,j] * f[i]
                expected[idx] -= this_rate
                expected[ktens.idxs[idx,i] + j] += this_rate
        # Effect of increased transfer of mass out of other bins.
        # (Including this one; the double counting is correct.)
        for i in range(nb):
            for j in range(ktens.nums[i,idx]):
                this_rate = ktens.data[i,idx,j] * f[i]
                expected[i] -= this_rate
                expected[ktens.idxs[i,idx] + j] += this_rate
        expected[:nb] /= bw
        for i in range(nb+1):
            self.assertAlmostEqual(rate_deriv[i,idx], expected[i], places=25)
        # Effect of perturbations to bottom bin should be 0.
        for i in range(nb+1):
            self.assertEqual(rate_deriv[i,6], 0.)

    def test_calc_rate_derivative_no_out_flux(self):
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.kernel, self.grid)
        f = np.linspace(2., nb+1, nb+1)
        _, rate_deriv = ktens.calc_rate(f, derivative=True, out_flux=False)
        self.assertEqual(rate_deriv.shape, (nb, nb))
        _, outflux_rate_deriv = ktens.calc_rate(f, derivative=True)
        for i in range(nb):
            for j in range(nb):
                self.assertAlmostEqual(rate_deriv[i,j],
                                       outflux_rate_deriv[i,j],
                                       places=25)


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


class TestModelStateDescriptor(unittest.TestCase):
    """
    Test ModelStateDescriptor methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3)
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=90)

    def test_init(self):
        const = self.constants
        grid = self.grid
        desc = ModelStateDescriptor(const, grid)

    def test_state_len_dsd_only(self):
        const = self.constants
        grid = self.grid
        desc = ModelStateDescriptor(const, grid)
        self.assertEqual(desc.state_len(), grid.num_bins+1)

    def test_dsd_loc(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        self.assertEqual(desc.dsd_loc(), (0, nb))

    def test_dsd_loc_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        self.assertEqual(desc.dsd_loc(with_fallout=True), (0, nb+1))

    def test_fallout_loc(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        self.assertEqual(desc.fallout_loc(), nb)

    def test_dsd_raw(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        raw = np.linspace(0., nb+1, nb+1)
        actual = desc.dsd_raw(raw)
        self.assertEqual(len(actual), nb)
        for i in range(nb):
            self.assertEqual(actual[i], raw[i])

    def test_dsd_raw_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        raw = np.linspace(0., nb+1, nb+1)
        actual = desc.dsd_raw(raw, with_fallout=True)
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertEqual(actual[i], raw[i])

    def test_fallout_raw(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        raw = np.linspace(0., nb+1, nb+1)
        self.assertEqual(desc.fallout_raw(raw), raw[nb])

    def test_construct_raw(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        dsd_scale = const.mass_conc_scale
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        raw = desc.construct_raw(dsd, fallout=fallout)
        self.assertEqual(len(raw), desc.state_len())
        actual_dsd = desc.dsd_raw(raw)
        self.assertEqual(len(actual_dsd), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd[i], dsd[i] / dsd_scale)
        self.assertAlmostEqual(desc.fallout_raw(raw), fallout / dsd_scale)

    def test_construct_raw_wrong_dsd_size_raises(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb+1, nb+1)
        with self.assertRaises(AssertionError):
            desc.construct_raw(dsd)
        dsd = np.linspace(0, nb-1, nb-1)
        with self.assertRaises(AssertionError):
            desc.construct_raw(dsd)

    def test_construct_raw_fallout_default_zero(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        raw = desc.construct_raw(dsd)
        self.assertEqual(desc.fallout_raw(raw), 0.)

    def test_no_derivatives(self):
        const = self.constants
        grid = self.grid
        desc = ModelStateDescriptor(const, grid)
        self.assertEqual(desc.dsd_deriv_num, 0)
        self.assertEqual(desc.dsd_deriv_names, [])
        self.assertEqual(len(desc.dsd_deriv_scales), 0)

    def test_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        self.assertEqual(desc.dsd_deriv_num, len(dsd_deriv_names))
        self.assertEqual(desc.dsd_deriv_names, dsd_deriv_names)
        for i in range(2):
            self.assertEqual(desc.dsd_deriv_scales[i], dsd_deriv_scales[i])
        self.assertEqual(desc.state_len(), 3*nb+3)

    def test_empty_derivatives(self):
        const = self.constants
        grid = self.grid
        desc = ModelStateDescriptor(const, grid)
        dsd_deriv_names = []
        dsd_deriv_scales = np.zeros((0,))
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        self.assertEqual(desc.dsd_deriv_num, 0)
        self.assertEqual(desc.dsd_deriv_names, [])
        self.assertEqual(len(desc.dsd_deriv_scales), 0)

    def test_derivatives_raises_on_duplicate(self):
        const = self.constants
        grid = self.grid
        dsd_deriv_names = ['lambda', 'lambda']
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid,
                                        dsd_deriv_names=dsd_deriv_names)

    def test_derivatives_default_scales(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        self.assertEqual(len(desc.dsd_deriv_scales), 2)
        self.assertEqual(desc.dsd_deriv_scales[0], 1.)
        self.assertEqual(desc.dsd_deriv_scales[1], 1.)

    def test_derivatives_raises_for_extra_scales(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_scales = np.array([3., 4.])
        with self.assertRaises(AssertionError):
            ModelStateDescriptor(const, grid,
                                 dsd_deriv_scales=dsd_deriv_scales)

    def test_derivatives_raises_for_mismatched_scale_num(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3., 4.])
        with self.assertRaises(AssertionError):
            ModelStateDescriptor(const, grid,
                                 dsd_deriv_names=dsd_deriv_names,
                                 dsd_deriv_scales=dsd_deriv_scales)

    def test_dsd_deriv_loc_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idxs, num = desc.dsd_deriv_loc()
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_all_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idxs, num = desc.dsd_deriv_loc(with_fallout=True)
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb+1)

    def test_dsd_deriv_loc_all_without_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idxs, num = desc.dsd_deriv_loc(with_fallout=False)
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idx, num = desc.dsd_deriv_loc('lambda')
        self.assertEqual(idx, nb+1)
        self.assertEqual(num, nb)
        idx, num = desc.dsd_deriv_loc('nu')
        self.assertEqual(idx, 2*nb+2)
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_individual_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idx, num = desc.dsd_deriv_loc('lambda', with_fallout=True)
        self.assertEqual(idx, nb+1)
        self.assertEqual(num, nb+1)
        idx, num = desc.dsd_deriv_loc('nu', with_fallout=True)
        self.assertEqual(idx, 2*nb+2)
        self.assertEqual(num, nb+1)

    def test_dsd_deriv_loc_individual_without_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idx, num = desc.dsd_deriv_loc('lambda', with_fallout=False)
        self.assertEqual(idx, nb+1)
        self.assertEqual(num, nb)
        idx, num = desc.dsd_deriv_loc('nu', with_fallout=False)
        self.assertEqual(idx, 2*nb+2)
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_raises_for_bad_string(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        with self.assertRaises(ValueError):
            desc.dsd_deriv_loc('nonsense')

    def test_fallout_deriv_loc_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idxs = desc.fallout_deriv_loc()
        self.assertEqual(idxs, [2*nb+1, 3*nb+2])

    def test_fallout_deriv_loc_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        idx = desc.fallout_deriv_loc('lambda')
        self.assertEqual(idx, 2*nb+1)
        idx = desc.fallout_deriv_loc('nu')
        self.assertEqual(idx, 3*nb+2)

    def test_fallout_deriv_loc_raises_for_bad_string(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        with self.assertRaises(ValueError):
            idxs = desc.fallout_deriv_loc('nonsense')

    def test_dsd_deriv_raw_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        derivs = desc.dsd_deriv_raw(raw)
        self.assertEqual(derivs.shape, (2, nb))
        for i in range(nb):
            self.assertEqual(derivs[0,i], raw[nb+1+i])
            self.assertEqual(derivs[1,i], raw[2*nb+2+i])

    def test_dsd_deriv_raw_all_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        derivs = desc.dsd_deriv_raw(raw, with_fallout=True)
        self.assertEqual(derivs.shape, (2, nb+1))
        for i in range(nb+1):
            self.assertEqual(derivs[0,i], raw[nb+1+i])
            self.assertEqual(derivs[1,i], raw[2*nb+2+i])

    def test_dsd_deriv_raw_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        deriv = desc.dsd_deriv_raw(raw, 'lambda')
        self.assertEqual(len(deriv), nb)
        for i in range(nb):
            self.assertEqual(deriv[i], raw[nb+1+i])
        deriv = desc.dsd_deriv_raw(raw, 'nu')
        self.assertEqual(len(deriv), nb)
        for i in range(nb):
            self.assertEqual(deriv[i], raw[2*nb+2+i])

    def test_dsd_deriv_raw_individual_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        deriv = desc.dsd_deriv_raw(raw, 'lambda', with_fallout=True)
        self.assertEqual(len(deriv), nb+1)
        for i in range(nb+1):
            self.assertEqual(deriv[i], raw[nb+1+i])
        deriv = desc.dsd_deriv_raw(raw, 'nu', with_fallout=True)
        self.assertEqual(len(deriv), nb+1)
        for i in range(nb+1):
            self.assertEqual(deriv[i], raw[2*nb+2+i])

    def test_dsd_deriv_raw_raises_for_bad_string(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        with self.assertRaises(ValueError):
            desc.dsd_deriv_raw(raw, 'nonsense')

    def test_fallout_deriv_raw_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        fallout_derivs = desc.fallout_deriv_raw(raw)
        self.assertEqual(len(fallout_derivs), 2)
        for i in range(2):
            self.assertEqual(fallout_derivs[i], raw[nb+(i+1)*(nb+1)])

    def test_fallout_deriv_raw_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        fallout_deriv = desc.fallout_deriv_raw(raw, 'lambda')
        self.assertEqual(fallout_deriv, raw[2*nb+1])
        fallout_deriv = desc.fallout_deriv_raw(raw, 'nu')
        self.assertEqual(fallout_deriv, raw[3*nb+2])

    def test_construct_raw_with_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd_scale = const.mass_conc_scale
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_dsd_deriv = desc.dsd_deriv_raw(raw)
        self.assertEqual(actual_dsd_deriv.shape, dsd_deriv.shape)
        for i in range(len(actual_dsd_deriv.flat)):
            self.assertAlmostEqual(actual_dsd_deriv.flat[i],
                                   dsd_deriv.flat[i] / dsd_scale)

    def test_construct_raw_raises_for_missing_derivative(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, fallout=fallout)

    def test_construct_raw_raises_for_extra_derivative(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)

    def test_construct_raw_raises_for_wrong_derivative_shape(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb+1))
        with self.assertRaises(AssertionError):
            desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)
        dsd_deriv = np.zeros((3, nb))
        with self.assertRaises(AssertionError):
            desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)

    def test_construct_raw_allows_empty_derivative(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((0, nb))
        # Just checking that this doesn't throw an exception.
        desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)

    def test_construct_raw_with_derivatives_scaling(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd_scale = const.mass_conc_scale
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_dsd_deriv = desc.dsd_deriv_raw(raw)
        self.assertEqual(actual_dsd_deriv.shape, dsd_deriv.shape)
        for i in range(2):
            for j in range(nb):
                self.assertAlmostEqual(actual_dsd_deriv[i,j],
                                       dsd_deriv[i,j] / dsd_deriv_scales[i]
                                          / dsd_scale)

    def test_construct_raw_with_fallout_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd_scale = const.mass_conc_scale
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        self.assertEqual(len(actual_fallout_deriv), 2)
        for i in range(2):
            self.assertAlmostEqual(actual_fallout_deriv[i],
                                   fallout_deriv[i] / dsd_scale)

    def test_construct_raw_missing_fallout_derivative_is_zero(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        self.assertEqual(len(actual_fallout_deriv), 2)
        for i in range(2):
            self.assertAlmostEqual(actual_fallout_deriv[i], 0.)

    def test_construct_raw_with_fallout_deriv_raises_for_wrong_length(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800., 900.])
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv,
                                     fallout_deriv=fallout_deriv)

    def test_construct_raw_with_extra_fallout_deriv_raises(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        fallout_deriv = np.array([700., 800., 900.])
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, fallout=fallout,
                                     fallout_deriv=fallout_deriv)

    def test_construct_raw_allows_empty_fallout(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        fallout_deriv = np.zeros((0,))
        raw = desc.construct_raw(dsd, fallout=fallout,
                                 fallout_deriv=fallout_deriv)

    def test_construct_raw_with_derivatives_scaling(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd_scale = const.mass_conc_scale
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        self.assertEqual(actual_fallout_deriv.shape, fallout_deriv.shape)
        for i in range(2):
            self.assertAlmostEqual(actual_fallout_deriv[i],
                                   fallout_deriv[i] / dsd_deriv_scales[i]
                                      / dsd_scale)

    def test_perturbation_covariance(self):
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale=1.e-3,
                               time_scale=400.)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=1.e-3,
                                 num_bins=90)
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        self.assertEqual(desc.perturb_num, nvar)
        self.assertIsNone(desc.correction_time)
        self.assertEqual(desc.state_len(), 3*nb + 3 + nvar*nvar)
        self.assertEqual(desc.perturb_wvs.shape, (nvar, nb))
        for i in range(nvar):
            for j in range(nb):
                self.assertEqual(desc.perturb_wvs[i,j],
                                 perturbed_variables[i][0][j])
        self.assertEqual(desc.perturb_scales.shape, (nvar,))
        for i in range(nvar):
            self.assertEqual(desc.perturb_scales[i], perturbed_variables[i][2])
        self.assertEqual(desc.perturbation_rate.shape, (nvar, nvar))
        for i in range(nvar):
            for j in range(nvar):
                self.assertAlmostEqual(desc.perturbation_rate[i,j],
                                         perturbation_rate[i,j] \
                                             / perturbed_variables[i][2] \
                                             / perturbed_variables[j][2] \
                                             * const.time_scale)

    def test_perturbation_covariance_raises_for_mismatched_sizes(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
        ]
        perturbation_rate = np.eye(3)
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_raises_for_rate_without_variable(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        perturbation_rate = np.eye(3)
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_allows_variables_without_rate(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
        ]
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables)
        self.assertEqual(desc.perturbation_rate.shape, (2, 2))
        for i in range(2):
            for j in range(2):
                self.assertEqual(desc.perturbation_rate[i,j], 0.)

    def test_perturbation_covariance_correction_time(self):
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale=1.e-3,
                               time_scale=400.)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=1.e-3,
                                 num_bins=90)
        nb = grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        correction_time = 5.
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time)
        self.assertAlmostEqual(desc.correction_time,
                               correction_time / const.time_scale)

    def test_perturb_cov_requires_correction_time_when_dims_mismatch(self):
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale=1.e-3,
                               time_scale=400.)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=1.e-3,
                                 num_bins=90)
        nb = grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid,
                                        dsd_deriv_names=dsd_deriv_names,
                                        dsd_deriv_scales=dsd_deriv_scales,
                                        perturbed_variables=perturbed_variables,
                                        perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_correction_time_without_pv_raises(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        correction_time = 5.
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(self.constants, self.grid,
                                        dsd_deriv_names=dsd_deriv_names,
                                        dsd_deriv_scales=dsd_deriv_scales,
                                        correction_time=correction_time)

    def test_perturb_cov_loc(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        self.assertEqual(desc.perturb_cov_loc(), (3*nb+3, nvar**2))

    def test_perturb_cov_raw(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        sl = desc.state_len()
        raw = np.zeros((sl,))
        raw[-nvar*nvar:] = np.linspace(1, nvar*nvar, nvar*nvar)
        actual = desc.perturb_cov_raw(raw)
        expected = np.reshape(np.linspace(1, nvar*nvar, nvar*nvar),
                              (nvar, nvar))
        self.assertEqual(actual.shape, expected.shape)
        for i in range(nvar):
            for j in range(nvar):
                self.assertEqual(actual[i,j],
                                 expected[i,j])

    def test_perturb_cov_construct_raw(self):
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               std_diameter=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale=1.e-3,
                               time_scale=400.)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=1.e-3,
                                 num_bins=90)
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        dsd = np.linspace(0, nb, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        orig = np.reshape(np.linspace(1, nvar*nvar, nvar*nvar),
                          (nvar, nvar))
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 perturb_cov=orig)
        self.assertEqual(len(raw), desc.state_len())
        actual = desc.perturb_cov_raw(raw)
        self.assertEqual(actual.shape, orig.shape)
        for i in range(nvar):
            for j in range(nvar):
                expected = orig[i,j] \
                    / perturbed_variables[i][2] \
                    / perturbed_variables[j][2]
                self.assertAlmostEqual(actual[i,j],
                                       expected)

    def test_perturb_cov_construct_raw_raises_if_perturb_unexpected(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        nvar = 3
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        orig = np.reshape(np.linspace(1, nvar*nvar, nvar*nvar),
                          (nvar, nvar))
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, perturb_cov=orig)

    def test_perturb_cov_construct_raw_raises_if_perturb_is_wrong_size(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        dsd = np.linspace(0, nb, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        orig = np.zeros((2, 2))
        with self.assertRaises(AssertionError):
            raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                     perturb_cov=orig)

    def test_perturb_cov_construct_raw_defaults_to_zero_cov(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        dsd = np.linspace(0, nb, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual = desc.perturb_cov_raw(raw)
        self.assertEqual(actual.shape, (nvar, nvar))
        for i in range(nvar):
            for j in range(nvar):
                self.assertEqual(actual[i,j], 0.)


class TestModelState(unittest.TestCase):
    """
    Test ModelState methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        nb = 90
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=nb)
        self.desc = ModelStateDescriptor(self.constants,
                                         self.grid)
        self.dsd = np.linspace(0, nb, nb)
        self.fallout = 200.
        self.raw = self.desc.construct_raw(self.dsd, self.fallout)

    def test_init(self):
        desc = self.desc
        state = ModelState(desc, self.raw)
        self.assertEqual(len(state.raw), desc.state_len())

    def test_dsd(self):
        desc = self.desc
        nb = self.grid.num_bins
        state = ModelState(desc, self.raw)
        actual = state.dsd()
        self.assertEqual(len(actual), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual[i], self.dsd[i])

    def test_fallout(self):
        desc = self.desc
        state = ModelState(desc, self.raw)
        self.assertEqual(state.fallout(), self.fallout)

    def test_dsd_moment(self):
        grid = self.grid
        desc = self.desc
        nu = 5.
        lam = nu / 1.e-5
        dsd = (np.pi/6. * self.constants.rho_water) \
            * gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3),
                               1., places=6)
        # Note the fairly low accuracy in the moment calculations at modest
        # grid resolutions.
        self.assertAlmostEqual(state.dsd_moment(6)
                               / (lam**-3 * (nu + 3.) * (nu + 4.) * (nu + 5.)),
                               1.,
                               places=2)
        self.assertAlmostEqual(state.dsd_moment(0)
                               / (lam**3 / (nu * (nu + 1.) * (nu + 2.))
                               * (1. - gammainc(nu, lam*grid.bin_bounds_d[0]))),
                               1.,
                               places=2)

    def test_dsd_moment_cloud_only_and_rain_only_raises(self):
        desc = self.desc
        state = ModelState(desc, self.raw)
        state.dsd_moment(3, cloud_only=True, rain_only=False)
        state.dsd_moment(3, cloud_only=False, rain_only=True)
        state.dsd_moment(3, cloud_only=False, rain_only=False)
        with self.assertRaises(AssertionError):
            state.dsd_moment(3, cloud_only=True, rain_only=True)

    def test_dsd_cloud_moment(self):
        grid = self.grid
        desc = self.desc
        nb = grid.num_bins
        bw = grid.bin_widths
        dsd = np.zeros((nb,))
        dsd[0] = (np.pi/6. * self.constants.rho_water) / bw[0]
        dsd[-1] = (np.pi/6. * self.constants.rho_water) / bw[-1]
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        # Make sure that only half the mass is counted.
        self.assertAlmostEqual(state.dsd_moment(3, cloud_only=True),
                               1.)
        # Since almost all the number will be counted, these should be
        # approximately equal.
        self.assertAlmostEqual(state.dsd_moment(0, cloud_only=True)
                               / state.dsd_moment(0),
                               1.)

    def test_dsd_cloud_moment_all_rain(self):
        nb = 10
        grid = GeometricMassGrid(self.constants,
                                 d_min=2.e-4,
                                 d_max=1.e-3,
                                 num_bins=nb)
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = np.ones((nb,))
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, cloud_only=True), 0.)

    def test_dsd_cloud_moment_all_cloud(self):
        nb = 10
        grid = GeometricMassGrid(self.constants,
                                 d_min=1.e-6,
                                 d_max=1.e-5,
                                 num_bins=nb)
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = np.ones((nb,))
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, cloud_only=True),
                               state.dsd_moment(3))

    def test_dsd_cloud_moment_bin_spanning_threshold(self):
        const = self.constants
        nb = 1
        grid = GeometricMassGrid(self.constants,
                                 d_min=5.e-5,
                                 d_max=2.e-4,
                                 num_bins=nb)
        bb = grid.bin_bounds
        bw = grid.bin_widths
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = (np.pi/6. * self.constants.rho_water) / bw
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, cloud_only=True)
                                   / state.dsd_moment(3),
                               0.5)
        self.assertAlmostEqual(state.dsd_moment(0, cloud_only=True)
                                   / ((np.exp(-bb[0]) - (1./const.rain_m))
                                      * dsd[0] / self.constants.std_mass),
                               1.)

    def test_dsd_rain_moment(self):
        grid = self.grid
        desc = self.desc
        nb = grid.num_bins
        bw = grid.bin_widths
        dsd = np.zeros((nb,))
        dsd[0] = (np.pi/6. * self.constants.rho_water) / bw[0]
        dsd[-1] = (np.pi/6. * self.constants.rho_water) / bw[-1]
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        # Make sure that only half the mass is counted.
        self.assertAlmostEqual(state.dsd_moment(3, rain_only=True),
                               1.)
        # Since almost all the M6 will be counted, these should be
        # approximately equal.
        self.assertAlmostEqual(state.dsd_moment(6, rain_only=True)
                               / state.dsd_moment(6),
                               1.)

    def test_dsd_rain_moment_all_rain(self):
        nb = 10
        grid = GeometricMassGrid(self.constants,
                                 d_min=2.e-4,
                                 d_max=1.e-3,
                                 num_bins=nb)
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = np.ones((nb,))
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, rain_only=True),
                               state.dsd_moment(3))

    def test_dsd_rain_moment_all_cloud(self):
        nb = 10
        grid = GeometricMassGrid(self.constants,
                                 d_min=1.e-6,
                                 d_max=1.e-5,
                                 num_bins=nb)
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = np.ones((nb,))
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, rain_only=True), 0.)

    def test_dsd_rain_moment_bin_spanning_threshold(self):
        const = self.constants
        nb = 1
        grid = GeometricMassGrid(self.constants,
                                 d_min=5.e-5,
                                 d_max=2.e-4,
                                 num_bins=nb)
        bb = grid.bin_bounds
        bw = grid.bin_widths
        desc = ModelStateDescriptor(self.constants,
                                    grid)
        dsd = (np.pi/6. * self.constants.rho_water) / bw
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        self.assertAlmostEqual(state.dsd_moment(3, rain_only=True)
                                   / state.dsd_moment(3),
                               0.5)
        self.assertAlmostEqual(state.dsd_moment(0, rain_only=True)
                                   / (((1./const.rain_m) - np.exp(-bb[1]))
                                      * dsd[0] / self.constants.std_mass),
                               1.)

    def test_dsd_deriv_all(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        actual_dsd_deriv = state.dsd_deriv()
        self.assertEqual(actual_dsd_deriv.shape, dsd_deriv.shape)
        for i in range(2*nb):
            self.assertAlmostEqual(actual_dsd_deriv.flat[i], dsd_deriv.flat[i])

    def test_dsd_deriv_all_scaling(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        actual_dsd_deriv = state.dsd_deriv()
        self.assertEqual(actual_dsd_deriv.shape, dsd_deriv.shape)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd_deriv[0,i], dsd_deriv[0,i])
            self.assertAlmostEqual(actual_dsd_deriv[1,i], dsd_deriv[1,i])

    def test_dsd_deriv_individual(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        actual_dsd_deriv = state.dsd_deriv('lambda')
        self.assertEqual(len(actual_dsd_deriv), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd_deriv[i], dsd_deriv[0,i])
        actual_dsd_deriv = state.dsd_deriv('nu')
        self.assertEqual(len(actual_dsd_deriv), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd_deriv[i], dsd_deriv[1,i])

    def test_dsd_deriv_individual_raises_if_not_found(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        with self.assertRaises(ValueError):
            state.dsd_deriv('nonsense')

    def test_dsd_deriv_scaling(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        actual_dsd_deriv = state.dsd_deriv('lambda')
        self.assertEqual(len(actual_dsd_deriv), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd_deriv[i], dsd_deriv[0,i])
        actual_dsd_deriv = state.dsd_deriv('nu')
        self.assertEqual(len(actual_dsd_deriv), nb)
        for i in range(nb):
            self.assertAlmostEqual(actual_dsd_deriv[i], dsd_deriv[1,i])

    def test_fallout_deriv_all(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv()
        self.assertEqual(len(actual_fallout_deriv), 2)
        for i in range(2):
            self.assertAlmostEqual(actual_fallout_deriv[i],
                                   fallout_deriv[i])

    def test_fallout_deriv_all_scaling(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv()
        self.assertEqual(len(actual_fallout_deriv), 2)
        for i in range(2):
            self.assertAlmostEqual(actual_fallout_deriv[i],
                                   fallout_deriv[i])

    def test_fallout_deriv_individual(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv('lambda')
        self.assertAlmostEqual(actual_fallout_deriv[0], fallout_deriv[0])
        actual_fallout_deriv = state.fallout_deriv('nu')
        self.assertAlmostEqual(actual_fallout_deriv[1], fallout_deriv[1])

    def test_fallout_deriv_individual_scaling(self):
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = np.array([3., 4.])
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv('lambda')
        self.assertAlmostEqual(actual_fallout_deriv[0], fallout_deriv[0])
        actual_fallout_deriv = state.fallout_deriv('nu')
        self.assertAlmostEqual(actual_fallout_deriv[1], fallout_deriv[1])

    def test_dsd_time_deriv_raw(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.dsd_time_deriv_raw([ktens])
        expected = ktens.calc_rate(dsd_raw, out_flux=True)
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_dsd_time_deriv_raw_two_kernels(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.dsd_time_deriv_raw([ktens, ktens])
        expected = 2.*ktens.calc_rate(dsd_raw, out_flux=True)
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_dsd_time_deriv_raw_no_kernels(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.dsd_time_deriv_raw([])
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertAlmostEqual(actual[i], 0., places=10)

    def test_time_derivative_raw(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([ktens])
        expected = state.dsd_time_deriv_raw([ktens])
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_time_derivative_raw_two_kernels(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([ktens, ktens])
        expected = 2. * ktens.calc_rate(dsd_raw, out_flux=True)
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_time_derivative_raw_no_kernels(self):
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([])
        self.assertEqual(len(actual), nb+1)
        for i in range(nb+1):
            self.assertEqual(actual[i], 0.)

    def test_time_derivative_raw_with_derivs(self):
        grid = self.grid
        nb = grid.num_bins
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean(),
                                  dsd_deriv[1,-4:].mean()])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([ktens])
        expected = np.zeros((3*nb+3,))
        expected[:nb+1], derivative = ktens.calc_rate(dsd_raw, derivative=True,
                                                      out_flux=True)
        dsd_scale = self.constants.mass_conc_scale
        deriv_plus_fallout = np.zeros((nb+1,))
        for i in range(2):
            deriv_plus_fallout[:nb] = dsd_deriv[i,:] / dsd_deriv_scales[i] \
                / dsd_scale
            deriv_plus_fallout[nb] = fallout_deriv[i] / dsd_deriv_scales[i] \
                / dsd_scale
            expected[(i+1)*(nb+1):(i+2)*(nb+1)] = \
                derivative @ deriv_plus_fallout
        self.assertEqual(len(actual), 3*nb+3)
        for i in range(3*nb+3):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_linear_func_raw(self):
        const = self.constants
        weight_vector = self.grid.moment_weight_vector(3, cloud_only=True)
        state = ModelState(self.desc, self.raw)
        actual = state.linear_func_raw(weight_vector)
        expected = state.dsd_moment(3, cloud_only=True)
        expected *= const.std_mass \
            / (const.std_diameter**3 * const.mass_conc_scale)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_linear_func_raw_with_derivative(self):
        const = self.constants
        weight_vector = self.grid.moment_weight_vector(3, cloud_only=True)
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        actual, actual_deriv = state.linear_func_raw(weight_vector,
                                                     derivative=True)
        expected = state.dsd_moment(3, cloud_only=True)
        expected *= const.std_mass \
            / (const.std_diameter**3 * const.mass_conc_scale)
        self.assertAlmostEqual(actual / expected, 1.)
        self.assertEqual(actual_deriv.shape, (2,))
        dsd_deriv_raw = desc.dsd_deriv_raw(state.raw)
        for i in range(2):
            expected = np.dot(dsd_deriv_raw[i], weight_vector)
            self.assertAlmostEqual(actual_deriv[i] / expected, 1.)

    def test_linear_func_raw_with_time_derivative(self):
        const = self.constants
        weight_vector = self.grid.moment_weight_vector(3, cloud_only=True)
        nb = self.grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        state = ModelState(desc, raw)
        dfdt = dsd * 0.1
        actual, actual_deriv = state.linear_func_raw(weight_vector,
                                                     derivative=True,
                                                     dfdt=dfdt)
        expected = state.dsd_moment(3, cloud_only=True)
        expected *= const.std_mass \
            / (const.std_diameter**3 * const.mass_conc_scale)
        self.assertAlmostEqual(actual / expected, 1.)
        self.assertEqual(actual_deriv.shape, (3,))
        dsd_deriv_raw = np.zeros((3, nb))
        dsd_deriv_raw[0,:] = dfdt
        dsd_deriv_raw[1:,:] = desc.dsd_deriv_raw(state.raw)
        for i in range(3):
            expected = np.dot(dsd_deriv_raw[i], weight_vector)
            self.assertAlmostEqual(actual_deriv[i] / expected, 1.)

    def test_linear_func_rate_raw(self):
        state = ModelState(self.desc, self.raw)
        dsd = self.dsd
        dfdt = dsd * 0.1
        weight_vector = self.grid.moment_weight_vector(3, cloud_only=True)
        actual = state.linear_func_rate_raw(weight_vector, dfdt)
        expected = np.dot(weight_vector, dfdt)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_linear_func_rate_derivative(self):
        nb = self.grid.num_bins
        state = ModelState(self.desc, self.raw)
        dsd = self.dsd
        dfdt_deriv = np.zeros((2, nb))
        dfdt_deriv[0,:] = dsd + 1.
        dfdt_deriv[1,:] = dsd + 2.
        dfdt = dsd * 0.1
        weight_vector = self.grid.moment_weight_vector(3, cloud_only=True)
        actual, actual_deriv = \
            state.linear_func_rate_raw(weight_vector, dfdt,
                                       dfdt_deriv=dfdt_deriv)
        expected = np.dot(weight_vector, dfdt)
        self.assertAlmostEqual(actual / expected, 1.)
        self.assertEqual(actual_deriv.shape, (2,))
        expected_deriv = dfdt_deriv @ weight_vector
        for i in range(2):
            self.assertAlmostEqual(actual_deriv[i] / expected_deriv[i], 1.)

    def test_rain_prod_breakdown(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        raw = desc.construct_raw(dsd)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        cloud_idx = grid.find_bin(np.log(const.rain_m))
        cloud_vector = np.zeros((nb,))
        cloud_vector[:cloud_idx] = 1.
        actual = state.rain_prod_breakdown(ktens, cloud_vector)
        self.assertEqual(len(actual), 2)
        cloud_weight_vector = grid.moment_weight_vector(3, cloud_only=True)
        rain_weight_vector = grid.moment_weight_vector(3, rain_only=True)
        cloud_inter = ktens.calc_rate(dsd_raw * cloud_vector, out_flux=True)
        auto = np.dot(rain_weight_vector, cloud_inter[:nb]) + cloud_inter[nb]
        auto *= const.mass_conc_scale / const.time_scale
        self.assertAlmostEqual(actual[0] / auto, 1.)
        total_inter = ktens.calc_rate(dsd_raw, out_flux=True)
        no_cloud_sc_or_auto = total_inter - cloud_inter
        accr = -np.dot(cloud_weight_vector, no_cloud_sc_or_auto[:nb])
        accr *= const.mass_conc_scale / const.time_scale
        self.assertAlmostEqual(actual[1] / accr, 1.)

    def test_rain_prod_breakdown_with_derivative(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        desc = self.desc
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(grid, lam, nu)
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        dsd_deriv_raw = np.zeros((3, nb+1))
        dsd_deriv_raw[0,:] = state.dsd_time_deriv_raw([ktens])
        dsd_deriv_raw[1:,:] = desc.dsd_deriv_raw(raw, with_fallout=True)
        cloud_idx = grid.find_bin(np.log(const.rain_m))
        cloud_vector = np.zeros((nb,))
        cloud_vector[:cloud_idx] = 1.
        actual, actual_deriv = state.rain_prod_breakdown(ktens, cloud_vector,
                                                         derivative=True)
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual_deriv.shape, (2, 3))
        cloud_weight_vector = grid.moment_weight_vector(3, cloud_only=True)
        rain_weight_vector = grid.moment_weight_vector(3, rain_only=True)
        cloud_inter, cloud_deriv = ktens.calc_rate(dsd_raw * cloud_vector,
                                                   out_flux=True,
                                                   derivative=True)
        auto = np.dot(rain_weight_vector, cloud_inter[:nb]) + cloud_inter[nb]
        auto *= const.mass_conc_scale / const.time_scale
        self.assertAlmostEqual(actual[0] / auto, 1.)
        cloud_dsd_deriv = np.transpose(dsd_deriv_raw).copy()
        for i in range(3):
            cloud_dsd_deriv[:nb,i] *= cloud_vector
            cloud_dsd_deriv[nb,i] = 0.
        cloud_f_deriv = cloud_deriv @ cloud_dsd_deriv
        auto_deriv = rain_weight_vector @ cloud_f_deriv[:nb,:] \
            + cloud_f_deriv[nb,:]
        auto_deriv *= const.mass_conc_scale / const.time_scale
        auto_deriv[0] /= const.time_scale
        auto_deriv[1:] *= desc.dsd_deriv_scales
        for i in range(3):
            self.assertAlmostEqual(actual_deriv[0,i] / auto_deriv[i], 1.)
        total_inter, total_deriv = ktens.calc_rate(dsd_raw, out_flux=True,
                                                   derivative=True)
        no_cloud_sc_or_auto = total_inter - cloud_inter
        accr = -np.dot(cloud_weight_vector, no_cloud_sc_or_auto[:nb])
        accr *= const.mass_conc_scale / const.time_scale
        self.assertAlmostEqual(actual[1] / accr, 1.)
        no_csc_or_auto_deriv = total_deriv @ dsd_deriv_raw.T - cloud_f_deriv
        accr_deriv = -cloud_weight_vector @ no_csc_or_auto_deriv[:nb,:]
        accr_deriv *= const.mass_conc_scale / const.time_scale
        accr_deriv[0] /= const.time_scale
        accr_deriv[1:] *= desc.dsd_deriv_scales
        for i in range(3):
            self.assertAlmostEqual(actual_deriv[1,i] / accr_deriv[i], 1.)

    def test_perturb_cov(self):
        grid = self.grid
        nb = grid.num_bins
        dsd_deriv_names = ['lambda', 'nu']
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
            (wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        expected = np.eye(nvar)
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 perturb_cov=expected)
        state = ModelState(desc, raw)
        actual = state.perturb_cov()
        self.assertEqual(actual.shape, expected.shape)
        for i in range(nvar):
            for j in range(nvar):
                self.assertAlmostEqual(actual[i,j], expected[i,j])
                self.assertEqual(actual[i,j], actual[j,i])

    def test_time_derivative_raw_with_perturb_cov(self):
        grid = self.grid
        nb = grid.num_bins
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean(),
                                  dsd_deriv[1,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((nvar, nvar)) + np.eye(nvar))
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv,
                                 perturb_cov=perturb_cov_init)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([ktens])
        expected = np.zeros((3*nb+3+nvar**2,))
        expected[:nb+1], rate_deriv = ktens.calc_rate(dsd_raw, derivative=True,
                                                      out_flux=True)
        dsd_scale = self.constants.mass_conc_scale
        deriv_plus_fallout = np.zeros((nb+1,))
        for i in range(2):
            deriv_plus_fallout[:nb] = dsd_deriv[i,:] / dsd_deriv_scales[i] \
                / dsd_scale
            deriv_plus_fallout[nb] = fallout_deriv[i] / dsd_deriv_scales[i] \
                / dsd_scale
            expected[(i+1)*(nb+1):(i+2)*(nb+1)] = \
                rate_deriv @ deriv_plus_fallout
        dfdt = expected[:nb]
        dsd_deriv_raw = np.zeros((3, nb))
        dsd_deriv_raw[0,:] = dfdt
        dsd_deriv_raw[1:,:] = desc.dsd_deriv_raw(state.raw)
        dfdt_deriv = dsd_deriv_raw @ rate_deriv[:nb,:nb].T
        perturb_cov_raw = desc.perturb_cov_raw(state.raw)
        moms = np.zeros((nvar,))
        mom_jac = np.zeros((nvar, 3))
        mom_rates = np.zeros((nvar,))
        mom_rate_jac = np.zeros((nvar, 3))
        for i in range(nvar):
            wv = perturbed_variables[i][0]
            moms[i], mom_jac[i,:] = state.linear_func_raw(wv, derivative=True,
                                                          dfdt=dfdt)
            mom_rates[i], mom_rate_jac[i,:] = \
                state.linear_func_rate_raw(wv, dfdt,
                                           dfdt_deriv=dfdt_deriv)
        transform = np.diag(LogTransform().derivative(moms))
        jacobian = la.inv(transform @ mom_jac)
        jacobian = transform @ mom_rate_jac @ jacobian
        sof_deriv = LogTransform().second_over_first_derivative(moms)
        jacobian += np.diag(mom_rates * sof_deriv)
        cov_rate = jacobian @ perturb_cov_raw
        cov_rate += cov_rate.T
        cov_rate += desc.perturbation_rate
        expected[-nvar*nvar:] = np.reshape(cov_rate, (nvar*nvar,))
        self.assertEqual(len(actual), 3*nb+3 + nvar**2)
        for i in range(3*nb+3 + nvar**2):
            self.assertAlmostEqual(actual[i], expected[i], places=9)

    def test_time_derivative_raw_with_perturb_cov_and_correction(self):
        grid = self.grid
        nb = grid.num_bins
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = [self.constants.std_diameter]
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        correction_time = 5.
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time)
        self.assertAlmostEqual(desc.correction_time,
                               correction_time / self.constants.time_scale)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((1, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((nvar, nvar)) + np.eye(nvar))
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv,
                                 perturb_cov=perturb_cov_init)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        actual = state.time_derivative_raw([ktens])
        expected = np.zeros((2*nb+2+nvar**2,))
        expected[:nb+1], rate_deriv = ktens.calc_rate(dsd_raw, derivative=True,
                                                      out_flux=True)
        dsd_scale = self.constants.mass_conc_scale
        deriv_plus_fallout = np.zeros((nb+1,))
        for i in range(1):
            deriv_plus_fallout[:nb] = dsd_deriv[i,:] / dsd_deriv_scales[i] \
                / dsd_scale
            deriv_plus_fallout[nb] = fallout_deriv[i] / dsd_deriv_scales[i] \
                / dsd_scale
            expected[(i+1)*(nb+1):(i+2)*(nb+1)] = \
                rate_deriv @ deriv_plus_fallout
        dfdt = expected[:nb]
        dsd_deriv_raw = np.zeros((2, nb))
        dsd_deriv_raw[0,:] = dfdt
        dsd_deriv_raw[1:,:] = desc.dsd_deriv_raw(state.raw)
        dfdt_deriv = dsd_deriv_raw @ rate_deriv[:nb,:nb].T
        perturb_cov_raw = desc.perturb_cov_raw(state.raw)
        moms = np.zeros((nvar,))
        mom_jac = np.zeros((nvar, 2))
        mom_rates = np.zeros((nvar,))
        mom_rate_jac = np.zeros((nvar, 2))
        for i in range(nvar):
            wv = perturbed_variables[i][0]
            moms[i], mom_jac[i,:] = state.linear_func_raw(wv, derivative=True,
                                                          dfdt=dfdt)
            mom_rates[i], mom_rate_jac[i,:] = \
                state.linear_func_rate_raw(wv, dfdt,
                                           dfdt_deriv=dfdt_deriv)
        transform = np.diag(LogTransform().derivative(moms))
        zeta_to_v = transform @ mom_jac
        jacobian = transform @ mom_rate_jac @ la.pinv(zeta_to_v)
        sof_deriv = LogTransform().second_over_first_derivative(moms)
        jacobian += np.diag(mom_rates * sof_deriv)
        sigma = desc.perturbation_rate
        projection = la.inv(zeta_to_v.T @ sigma @ zeta_to_v)
        projection = zeta_to_v @ projection @ zeta_to_v.T @ sigma
        perturb_cov_projected = projection @ perturb_cov_raw @ projection.T
        cov_rate = jacobian @ perturb_cov_projected
        cov_rate += cov_rate.T
        cov_rate += desc.perturbation_rate
        cov_rate += (perturb_cov_projected - perturb_cov_raw) \
            / desc.correction_time
        expected[-nvar*nvar:] = np.reshape(cov_rate, (nvar*nvar,))
        self.assertEqual(len(actual), 2*nb+2 + nvar**2)
        for i in range(2*nb+2 + nvar**2):
            self.assertAlmostEqual(actual[i], expected[i], places=10)

    def test_zeta_cov(self):
        grid = self.grid
        nb = grid.num_bins
        kernel = LongKernel(self.constants)
        ktens = KernelTensor(kernel, self.grid)
        ddn = 2
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        pn = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(pn)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((ddn, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean(),
                                  dsd_deriv[1,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((pn, pn)) + np.eye(pn))
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv,
                                 perturb_cov=perturb_cov_init)
        dsd_raw = desc.dsd_raw(raw)
        state = ModelState(desc, raw)
        ddsddt_raw = state.dsd_time_deriv_raw([ktens])[:nb]
        actual = state.zeta_cov_raw(ddsddt_raw)
        lfs = np.zeros((pn,))
        lf_jac = np.zeros((pn, ddn+1))
        for i in range(pn):
            wv = desc.perturb_wvs[i]
            lfs[i], lf_jac[i,:] = state.linear_func_raw(wv, derivative=True,
                                                        dfdt=ddsddt_raw)
        transform_mat = np.diag([desc.perturb_transforms[i].derivative(lfs[i])
                                 for i in range(pn)])
        v_to_zeta = la.pinv(transform_mat @ lf_jac)
        perturb_cov = desc.perturb_cov_raw(state.raw)
        expected = v_to_zeta @ perturb_cov @ v_to_zeta.T
        self.assertEqual(actual.shape, expected.shape)
        for i in range(len(expected.flat)):
            self.assertAlmostEqual(actual.flat[i] / expected.flat[i], 1.)


class TestIntegrator(unittest.TestCase):
    """
    Test Integrator methods.
    """
    def test_integrate_raw_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Integrator().integrate_raw(1., 2., 3.)

    def test_to_netcdf_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Integrator().to_netcdf(None)


class TestRK45Integrator(unittest.TestCase):
    """
    Test RK45Integrator methods and attributes.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        nb = 30
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=nb)
        self.kernel = LongKernel(self.constants)
        self.ktens = KernelTensor(self.kernel, self.grid)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        self.desc = ModelStateDescriptor(self.constants,
                                         self.grid,
                                         dsd_deriv_names=dsd_deriv_names,
                                         dsd_deriv_scales=dsd_deriv_scales)
        nu = 5.
        lam = nu / 1.e-4
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        self.raw = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        self.state = ModelState(self.desc, self.raw)
        ddn = 1
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = [self.constants.std_diameter]
        pn = 3
        wv0 = self.grid.moment_weight_vector(0)
        wv6 = self.grid.moment_weight_vector(6)
        wv9 = self.grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(pn)
        correction_time = 5.
        self.pc_desc = ModelStateDescriptor(self.constants,
                                     self.grid,
                                     dsd_deriv_names=dsd_deriv_names,
                                     dsd_deriv_scales=dsd_deriv_scales,
                                     perturbed_variables=perturbed_variables,
                                     perturbation_rate=perturbation_rate,
                                     correction_time=correction_time)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((ddn, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((pn, pn)) + np.eye(pn))
        self.pc_raw = self.pc_desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                      fallout_deriv=fallout_deriv,
                                      perturb_cov=perturb_cov_init)
        self.pc_state = ModelState(self.pc_desc, self.pc_raw)

    def test_integrate_raw(self):
        tscale = self.constants.time_scale
        dt = 1.e-5
        num_step = 2
        integrator = RK45Integrator(self.constants, dt)
        times, actual = integrator.integrate_raw(num_step*dt / tscale,
                                                 self.state,
                                                 [self.ktens])
        expected = np.linspace(0., num_step*dt, num_step+1) / tscale
        self.assertEqual(times.shape, (num_step+1,))
        for i in range(num_step):
            self.assertAlmostEqual(times[i], expected[i])
        self.assertEqual(actual.shape, (num_step+1, len(self.raw)))
        expected = np.zeros((num_step+1, len(self.raw)))
        dt_scaled = dt / tscale
        expected[0,:] = self.raw
        for i in range(num_step):
            expect_state = ModelState(self.desc, expected[i,:])
            expected[i+1,:] = expected[i,:] \
                + dt_scaled*expect_state.time_derivative_raw([self.ktens])
        scale = expected.max()
        for i in range(num_step+1):
            for j in range(len(self.raw)):
                self.assertAlmostEqual(actual[i,j]/scale, expected[i,j]/scale)

    def test_integrate(self):
        nb = self.grid.num_bins
        dt = 1.e-5
        num_step = 2
        integrator = RK45Integrator(self.constants, dt)
        exp = integrator.integrate(num_step*dt,
                                   self.state,
                                   [self.ktens])
        self.assertIs(exp.desc, self.state.desc)
        self.assertEqual(len(exp.proc_tens), 1)
        self.assertIs(exp.proc_tens[0], self.ktens)
        self.assertIs(exp.integrator, integrator)
        times = exp.times
        states = exp.states
        expected = np.linspace(0., num_step*dt, num_step+1)
        self.assertEqual(times.shape, (num_step+1,))
        for i in range(num_step):
            self.assertAlmostEqual(times[i], expected[i])
        self.assertEqual(len(states), num_step+1)
        expected = np.zeros((num_step+1, len(self.raw)))
        dt_scaled = dt / self.constants.time_scale
        expected[0,:] = self.raw
        for i in range(num_step):
            expect_state = ModelState(self.desc, expected[i,:])
            expected[i+1,:] = expected[i,:] \
                + dt_scaled*expect_state.time_derivative_raw([self.ktens])
        for i in range(num_step+1):
            actual_dsd = states[i].dsd()
            expected_dsd = expected[i,:nb] * self.constants.mass_conc_scale
            self.assertEqual(actual_dsd.shape, expected_dsd.shape)
            scale = expected_dsd.max()
            for j in range(nb):
                self.assertAlmostEqual(actual_dsd[j]/scale,
                                       expected_dsd[j]/scale)

    def test_integrate_with_perturb_cov(self):
        nb = self.grid.num_bins
        dt = 1.e-5
        num_step = 2
        integrator = RK45Integrator(self.constants, dt)
        exp = integrator.integrate(num_step*dt,
                                   self.pc_state,
                                   [self.ktens])
        for i in range(num_step+1):
            actual = exp.ddsddt[i,:]
            expected = exp.states[i].dsd_time_deriv_raw([self.ktens])[:nb]
            self.assertEqual(actual.shape, expected.shape)
            scale = expected.max()
            for j in range(len(expected)):
                self.assertAlmostEqual(actual[j] / scale,
                                       expected[j] / scale)
            actual = exp.zeta_cov[i,:,:]
            expected = exp.states[i].zeta_cov_raw(expected)
            self.assertEqual(actual.shape, expected.shape)
            scale = expected.max()
            for j in range(len(expected.flat)):
                self.assertAlmostEqual(actual.flat[j] / scale,
                                       expected.flat[j] / scale)


class TestTransform(unittest.TestCase):
    """
    Test Transform methods.
    """
    def test_transform(self):
        with self.assertRaises(NotImplementedError):
            Transform().transform(2.)

    def test_derivative(self):
        with self.assertRaises(NotImplementedError):
            Transform().derivative(2.)

    def test_second_over_first_derivative(self):
        with self.assertRaises(NotImplementedError):
            Transform().second_over_first_derivative(2.)

    def test_type_string(self):
        with self.assertRaises(NotImplementedError):
            Transform().type_string()

    def test_get_parameters(self):
        with self.assertRaises(NotImplementedError):
            Transform().get_parameters()


class TestLogTransform(unittest.TestCase):
    """
    Test LogTransform methods.
    """
    def test_log_transform(self):
        self.assertEqual(LogTransform().transform(2.), np.log(2.))

    def test_log_transform_deriv(self):
        self.assertEqual(LogTransform().derivative(2.), 1./2.)

    def test_log_transform_second_over_first_derivative(self):
        self.assertEqual(LogTransform().second_over_first_derivative(2.),
                         -1./2.)

    def test_type_string(self):
        self.assertEqual(LogTransform().type_string(), 'Log')

    def test_get_parameters(self):
        self.assertEqual(LogTransform().get_parameters(), [])


class TestBeardV(unittest.TestCase):
    """
    Test beard_v function.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4)

    def test_low_beard_v(self):
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 1.e-5),
                               0.0030440)

    def test_medium_beard_v(self):
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 6.e-4),
                               2.4455837)

    def test_high_beard_v(self):
        const = self.constants
        self.assertAlmostEqual(beard_v(const, 2.e-3),
                               6.5141471)

    def test_very_high_beard_v(self):
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
        d1 = 50.e-6
        x = 0.7
        d2 = x * d1
        self.assertAlmostEqual(sc_efficiency(d1, d2),
                               (0.8 / (1. + x))**2,
                               places=2)
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))

    def test_sc_efficiency_low_diameter(self):
        d1 = 5.e-6
        x = 0.7
        d2 = x * d1
        self.assertAlmostEqual(sc_efficiency(d1, d2),
                               sc_efficiency(20.e-6, x * 20.e-6))
        self.assertEqual(sc_efficiency(d1, d2), sc_efficiency(d2, d1))


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
        with self.assertRaises(AssertionError):
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

    def test_kernel_integral(self):
        a = -1.
        afun = lambda lx: sub_logs(a, lx)
        b = 0.
        bfun = lambda lx: sub_logs(b, lx)
        lxm = -2.5
        lxp = -2.
        f = lambda ly, lx: np.exp(lx) * self.kernel.kernel_lx(lx, ly)
        btype = 0
        actual = self.kernel.kernel_integral(-1., 0., lxm, lxp, btype)
        expected, _ = dblquad(f, lxm, lxp, a, b)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 1
        actual = self.kernel.kernel_integral(-1., 0., lxm, lxp, btype)
        expected, _ = dblquad(f, lxm, lxp, afun, b)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 2
        actual = self.kernel.kernel_integral(-1., 0., lxm, lxp, btype)
        expected, _ = dblquad(f, lxm, lxp, a, bfun)
        self.assertAlmostEqual(actual / expected, 1.)
        btype = 3
        actual = self.kernel.kernel_integral(-1., 0., lxm, lxp, btype)
        expected, _ = dblquad(f, lxm, lxp, afun, bfun)
        self.assertAlmostEqual(actual / expected, 1.)

    def test_kernel_integral_skips_close_x_bounds(self):
        lx1 = -3.
        lx2 = lx1 + 1.e-14
        actual = self.kernel.kernel_integral(-1., 0., lx1, lx2, btype=0)
        self.assertEqual(actual, 0.)


class TestExperiment(unittest.TestCase):
    """
    Test Experiment methods.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        nb = 30
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=nb)
        self.kernel = LongKernel(self.constants)
        self.ktens = KernelTensor(self.kernel, self.grid)
        ddn = 2
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        pn = 3
        wv0 = self.grid.moment_weight_vector(0)
        wv6 = self.grid.moment_weight_vector(6)
        wv9 = self.grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(pn)
        correction_time = 5.
        self.desc = ModelStateDescriptor(self.constants,
                                     self.grid,
                                     dsd_deriv_names=dsd_deriv_names,
                                     dsd_deriv_scales=dsd_deriv_scales,
                                     perturbed_variables=perturbed_variables,
                                     perturbation_rate=perturbation_rate,
                                     correction_time=correction_time)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((ddn, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean(),
                                  dsd_deriv[1,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((pn, pn)) + np.eye(pn))
        self.raw = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                      fallout_deriv=fallout_deriv,
                                      perturb_cov=perturb_cov_init)
        self.state = ModelState(self.desc, self.raw)
        nu2 = 0.
        lam = nu / 5.e-5
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((ddn, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        self.raw2 = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                      fallout_deriv=fallout_deriv,
                                      perturb_cov=perturb_cov_init)
        self.state2 = ModelState(self.desc, self.raw2)
        self.times = np.array([0., 1.])
        dt = 15.
        self.integrator = RK45Integrator(self.constants, dt)

    def test_init(self):
        times = self.times
        ntimes = len(times)
        raws = np.zeros((ntimes, len(self.state.raw)))
        raws[0,:] = self.state.raw
        raws[1,:] = self.state2.raw
        states = [self.state, self.state2]
        exp = Experiment(self.desc, [self.ktens], self.integrator, times, raws)
        self.assertEqual(exp.constants.rho_air, self.constants.rho_air)
        self.assertEqual(exp.mass_grid.num_bins, self.grid.num_bins)
        self.assertEqual(exp.proc_tens[0].kernel.kc, self.kernel.kc)
        self.assertEqual(exp.proc_tens[0].data.shape, self.ktens.data.shape)
        self.assertTrue(np.all(exp.proc_tens[0].data == self.ktens.data))
        self.assertEqual(exp.times.shape, (ntimes,))
        self.assertEqual(len(exp.states), ntimes)
        for i in range(ntimes):
            self.assertEqual(exp.states[i].dsd_moment(0),
                             states[i].dsd_moment(0))
        self.assertEqual(exp.desc.dsd_deriv_num, 2)
        self.assertEqual(exp.num_time_steps, ntimes)

    def test_init_with_ddsddt(self):
        nb = self.grid.num_bins
        times = self.times
        ntimes = len(times)
        raws = np.zeros((ntimes, len(self.state.raw)))
        raws[0,:] = self.state.raw
        raws[1,:] = self.state2.raw
        states = [self.state, self.state2]
        ddsddt = np.zeros((ntimes, nb))
        for i in range(ntimes):
            ddsddt = np.linspace(-nb+i, -i, nb)
        exp = Experiment(self.desc, [self.ktens], self.integrator, times, raws,
                         ddsddt=ddsddt)
        self.assertEqual(exp.ddsddt.shape, ddsddt.shape)
        for i in range(len(ddsddt.flat)):
            self.assertEqual(exp.ddsddt.flat[i],
                             ddsddt.flat[i])

    def test_init_with_zeta_cov(self):
        nb = self.grid.num_bins
        times = self.times
        ntimes = len(times)
        raws = np.zeros((ntimes, len(self.state.raw)))
        raws[0,:] = self.state.raw
        raws[1,:] = self.state2.raw
        states = [self.state, self.state2]
        ddn = self.desc.dsd_deriv_num
        zeta_cov = np.zeros((ntimes, ddn, ddn))
        for i in range(ntimes):
            zeta_cov = np.reshape(np.linspace(50. + i, 50. + (i + ddn**2-1),
                                                ddn**2),
                                    (ddn, ddn))
        exp = Experiment(self.desc, [self.ktens], self.integrator, times, raws,
                         zeta_cov=zeta_cov)
        self.assertEqual(exp.zeta_cov.shape, zeta_cov.shape)
        for i in range(len(zeta_cov.flat)):
            self.assertEqual(exp.zeta_cov.flat[i],
                             zeta_cov.flat[i])


class TestNetcdfFile(unittest.TestCase):
    """
    Test NetcdfFile methods.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        nb = 30
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=nb)
        self.kernel = LongKernel(self.constants)
        self.ktens = KernelTensor(self.kernel, self.grid)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        nvar = 3
        wv0 = self.grid.moment_weight_vector(0)
        wv6 = self.grid.moment_weight_vector(6)
        wv9 = self.grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        correction_time = 5.
        self.desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time)
        nu = 5.
        lam = nu / 1.e-4
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        self.raw = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        self.state = ModelState(self.desc, self.raw)
        nu2 = 0.
        lam = nu / 5.e-5
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        raw2 = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        raws = np.zeros((2, len(self.raw)))
        raws[0,:] = self.raw
        raws[1,:] = raw2
        self.times = np.array([0., 1.])
        dt = 15.
        self.integrator = RK45Integrator(self.constants, dt)
        self.exp = Experiment(self.desc, self.ktens, self.integrator,
                              self.times, raws)
        self.dataset = nc4.Dataset('test.nc', 'w', diskless=True)
        self.NetcdfFile = NetcdfFile(self.dataset)

    def tearDown(self):
        self.dataset.close()

    def test_variable_is_present(self):
        x = 25.
        self.NetcdfFile.write_scalar('x', x, 'f8', '1',
                                     'A description')
        self.assertTrue(self.NetcdfFile.variable_is_present('x'))
        self.assertFalse(self.NetcdfFile.variable_is_present('y'))

    def test_read_write_scalar(self):
        x = 25.
        self.NetcdfFile.write_scalar('x', x, 'f8', '1',
                                     'A description')
        actual = self.NetcdfFile.read_scalar('x')
        self.assertEqual(actual, x)

    def test_read_write_dimension(self):
        dim = 20
        self.NetcdfFile.write_dimension('dim', dim)
        actual = self.NetcdfFile.read_dimension('dim')
        self.assertEqual(actual, dim)

    def test_read_write_characters(self):
        self.NetcdfFile.write_dimension('string_len', 16)
        string = "Hi there!"
        self.NetcdfFile.write_characters('string', string, 'string_len',
                                         'A description')
        actual = self.NetcdfFile.read_characters('string')
        self.assertEqual(actual, string)

    def test_read_write_too_many_characters_raises(self):
        self.NetcdfFile.write_dimension('string_len', 2)
        string = "Hi there!"
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_characters('string', string, 'string_len',
                                             'A description')

    def test_read_write_array(self):
        dim = 5
        self.NetcdfFile.write_dimension('dim', dim)
        array = np.linspace(0., dim-1., dim)
        self.NetcdfFile.write_array('array', array,
            'f8', ['dim'], '1', 'A description')
        array2 = self.NetcdfFile.read_array('array')
        self.assertEqual(array2.shape, array.shape)
        for i in range(dim):
            self.assertEqual(array2[i], array[i])

    def test_write_array_raises_for_wrong_dimension(self):
        dim = 5
        self.NetcdfFile.write_dimension('dim', dim)
        array = np.linspace(0., dim-1., dim+1)
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_array('array', array,
                'f8', ['dim'], '1', 'A description')

    def test_read_write_characters_array(self):
        self.NetcdfFile.write_dimension('string_len', 16)
        self.NetcdfFile.write_dimension('string_num', 2)
        string = "Hi there!"
        string2 = "Bye there!"
        self.NetcdfFile.write_characters(
            'strings', [string, string2], ['string_num', 'string_len'],
            'A description')
        actual = self.NetcdfFile.read_characters('strings')
        self.assertEqual(actual, [string, string2])

    def test_read_write_too_many_characters_in_array_raises(self):
        self.NetcdfFile.write_dimension('string_len', 2)
        self.NetcdfFile.write_dimension('string_num', 2)
        string = "Hi"
        string2 = "Bye"
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_characters(
                'strings', [string, string2], ['string_len', 'string_num'],
                'A description')

    def test_read_write_characters_array_raises_for_wrong_dimension(self):
        self.NetcdfFile.write_dimension('string_len', 16)
        self.NetcdfFile.write_dimension('string_num', 3)
        string = "Hi"
        string2 = "Bye"
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_characters(
                'strings', [string, string2], ['string_len', 'string_num'],
                'A description')

    def test_read_write_characters_array_raises_for_too_many_dimensions(self):
        self.NetcdfFile.write_dimension('string_len', 2)
        self.NetcdfFile.write_dimension('string_num', 2)
        self.NetcdfFile.write_dimension('lang_num', 2)
        string = "Hi"
        string2 = "By"
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_characters(
                'strings', [string, string2],
                ['string_len', 'lang_num', 'string_num'],
                'A description')

    def test_read_write_characters_multidimensional_array(self):
        self.NetcdfFile.write_dimension('string_len', 16)
        self.NetcdfFile.write_dimension('string_num', 2)
        self.NetcdfFile.write_dimension('lang_num', 2)
        string = "Hi there!"
        string2 = "Bye there!"
        string3 = "Hola!"
        string4 = "Adios!"
        strings = [[string, string2], [string3, string4]]
        self.NetcdfFile.write_characters(
            'strings', strings, ['string_num', 'lang_num', 'string_len'],
            'A description')
        actual = self.NetcdfFile.read_characters('strings')
        self.assertEqual(actual, strings)

    def test_read_write_characters_raises_for_too_few_dimensions(self):
        self.NetcdfFile.write_dimension('string_len', 2)
        self.NetcdfFile.write_dimension('string_num', 2)
        string = "Hi"
        string2 = "By"
        string3 = "Ho"
        string4 = "Ad"
        strings = [[string, string2], [string3, string4]]
        with self.assertRaises(AssertionError):
            self.NetcdfFile.write_characters(
                'strings', strings, ['string_num', 'string_len'],
                'A description')

    def test_constants_io(self):
        const = self.constants
        self.NetcdfFile.write_constants(const)
        const2 = self.NetcdfFile.read_constants()
        self.assertEqual(const.rho_water, const2.rho_water)
        self.assertEqual(const.rho_air, const2.rho_air)
        self.assertEqual(const.std_diameter, const2.std_diameter)
        self.assertEqual(const.rain_d, const2.rain_d)
        self.assertEqual(const.mass_conc_scale, const2.mass_conc_scale)
        self.assertEqual(const.time_scale, const2.time_scale)

    def test_long_kernel_io(self):
        kernel = self.kernel
        self.NetcdfFile.write_kernel(kernel)
        kernel2 = self.NetcdfFile.read_kernel(self.constants)
        self.assertEqual(kernel.kc, kernel2.kc)
        self.assertEqual(kernel.kr, kernel2.kr)
        self.assertEqual(kernel.log_rain_m, kernel2.log_rain_m)

    def test_hall_kernel_io(self):
        kernel = HallKernel(self.constants, 'ScottChen')
        self.NetcdfFile.write_kernel(kernel)
        kernel2 = self.NetcdfFile.read_kernel(self.constants)
        self.assertEqual(kernel.efficiency_name, kernel2.efficiency_name)

    def test_kernel_to_netcdf_not_implemented(self):
        kernel = Kernel()
        with self.assertRaises(NotImplementedError):
            self.NetcdfFile.write_kernel(kernel)

    def test_bad_kernel_type_raises(self):
        self.NetcdfFile.write_dimension('kernel_type_str_len',
                                        Kernel.kernel_type_str_len)
        self.NetcdfFile.write_characters('kernel_type',
                                         'nonsense',
                                         'kernel_type_str_len',
                                         'Type of kernel')
        with self.assertRaises(AssertionError):
            self.NetcdfFile.read_kernel(self.constants)

    def test_mass_grid_io(self):
        grid = self.grid
        self.NetcdfFile.write_mass_grid(grid)
        grid2 = self.NetcdfFile.read_mass_grid(self.constants)
        self.assertEqual(grid.d_min, grid2.d_min)
        self.assertEqual(grid.d_max, grid2.d_max)
        self.assertEqual(grid.num_bins, grid2.num_bins)

    def test_grid_to_netcdf_not_implemented(self):
        grid = MassGrid()
        with self.assertRaises(NotImplementedError):
            self.NetcdfFile.write_mass_grid(grid)

    def test_bad_grid_type_raises(self):
        self.NetcdfFile.write_dimension('mass_grid_type_str_len',
                                        MassGrid.mass_grid_type_str_len)
        self.NetcdfFile.write_characters('mass_grid_type',
                                         'nonsense',
                                         'mass_grid_type_str_len',
                                         'Type of mass grid')
        with self.assertRaises(AssertionError):
            self.NetcdfFile.read_mass_grid(self.constants)

    def test_ktens_io(self):
        ktens = self.ktens
        self.NetcdfFile.write_mass_grid(self.grid)
        self.NetcdfFile.write_kernel_tensor(ktens)
        ktens2 = self.NetcdfFile.read_kernel_tensor(self.kernel, self.grid)
        self.assertEqual(ktens2.boundary, ktens.boundary)
        self.assertEqual(ktens2.data.shape, ktens.data.shape)
        scale = ktens.data.max()
        for i in range(len(ktens.data.flat)):
            self.assertAlmostEqual(ktens2.data.flat[i] / scale,
                                   ktens.data.flat[i] / scale)

    def test_cgk_io(self):
        const = self.constants
        kernel = self.kernel
        grid = self.grid
        ktens = self.ktens
        self.NetcdfFile.write_cgk(ktens)
        const2, kernel2, grid2, ktens2 = self.NetcdfFile.read_cgk()
        self.assertEqual(const2.rho_water, const.rho_water)
        self.assertEqual(kernel2.kc, kernel.kc)
        self.assertEqual(grid2.d_min, grid.d_min)
        self.assertEqual(ktens2.data.shape, ktens.data.shape)
        scale = ktens.data.max()
        for i in range(len(ktens.data.flat)):
            self.assertAlmostEqual(ktens2.data.flat[i] / scale,
                                   ktens.data.flat[i] / scale)

    def test_desc_io(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        desc = self.desc
        self.NetcdfFile.write_cgk(self.ktens)
        self.NetcdfFile.write_descriptor(desc)
        desc2 = self.NetcdfFile.read_descriptor(const, grid)
        self.assertEqual(desc2.dsd_deriv_num, desc.dsd_deriv_num)
        for i in range(desc.dsd_deriv_num):
            self.assertEqual(desc2.dsd_deriv_names[i],
                             desc.dsd_deriv_names[i])
            self.assertEqual(desc2.dsd_deriv_scales[i],
                             desc.dsd_deriv_scales[i])
        self.assertEqual(desc2.perturb_num, desc.perturb_num)
        for i in range(desc.perturb_num):
            for j in range(nb):
                self.assertEqual(desc2.perturb_wvs[i,j],
                                 desc.perturb_wvs[i,j])
            self.assertEqual(desc2.perturb_transforms[i].transform(2.),
                             desc.perturb_transforms[i].transform(2.))
            self.assertEqual(desc2.perturb_scales[i],
                             desc.perturb_scales[i])
            for j in range(desc.perturb_num):
                self.assertEqual(desc2.perturbation_rate[i,j],
                                 desc.perturbation_rate[i,j])
        self.assertEqual(desc2.correction_time, desc.correction_time)

    def test_desc_io_bad_transform_type(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        desc = self.desc
        self.NetcdfFile.write_cgk(self.ktens)
        self.NetcdfFile.write_descriptor(desc)
        ttsl = self.NetcdfFile.read_dimension("transform_type_str_len")
        self.NetcdfFile.nc['perturb_transform_types'][0,:] = \
            nc4.stringtochar(np.array(['nonsense'], 'S{}'.format(ttsl)))
        with self.assertRaises(AssertionError):
            self.NetcdfFile.read_descriptor(const, grid)

    def test_simple_desc_io(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        self.NetcdfFile.write_cgk(self.ktens)
        desc = ModelStateDescriptor(const, grid)
        self.NetcdfFile.write_descriptor(desc)
        desc2 = self.NetcdfFile.read_descriptor(const, grid)
        self.assertEqual(desc2.dsd_deriv_num, desc.dsd_deriv_num)
        self.assertEqual(desc2.perturb_num, desc.perturb_num)

    def test_desc_io_no_correction_time(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        self.NetcdfFile.write_cgk(self.ktens)
        dsd_deriv_names = ['lambda', 'nu']
        dsd_deriv_scales = [self.constants.std_diameter, 1.]
        nvar = 3
        wv0 = self.grid.moment_weight_vector(0)
        wv6 = self.grid.moment_weight_vector(6)
        wv9 = self.grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), scale),
            (wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(const, grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        self.NetcdfFile.write_descriptor(desc)
        desc2 = self.NetcdfFile.read_descriptor(const, grid)
        self.assertEqual(desc2.dsd_deriv_num, desc.dsd_deriv_num)
        for i in range(desc.dsd_deriv_num):
            self.assertEqual(desc2.dsd_deriv_names[i],
                             desc.dsd_deriv_names[i])
            self.assertEqual(desc2.dsd_deriv_scales[i],
                             desc.dsd_deriv_scales[i])
        self.assertEqual(desc2.perturb_num, desc.perturb_num)
        for i in range(desc.perturb_num):
            for j in range(nb):
                self.assertEqual(desc2.perturb_wvs[i,j],
                                 desc.perturb_wvs[i,j])
            self.assertEqual(desc2.perturb_transforms[i].transform(2.),
                             desc.perturb_transforms[i].transform(2.))
            self.assertEqual(desc2.perturb_scales[i],
                             desc.perturb_scales[i])
            for j in range(desc.perturb_num):
                self.assertEqual(desc2.perturbation_rate[i,j],
                                 desc.perturbation_rate[i,j])
        self.assertEqual(desc2.correction_time, desc.correction_time)

    def test_integrator_io(self):
        const = self.constants
        integrator = self.integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, RK45Integrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_bad_integrator_type_raises(self):
        const = self.constants
        integrator = self.integrator
        self.NetcdfFile.write_integrator(integrator)
        itsl = self.NetcdfFile.read_dimension("integrator_type_str_len")
        self.NetcdfFile.nc['integrator_type'][:] = \
            nc4.stringtochar(np.array(['nonsense'], 'S{}'.format(itsl)))
        with self.assertRaises(AssertionError):
            integrator2 = self.NetcdfFile.read_integrator(const)

    def test_simple_experiment_io(self):
        desc = self.desc
        ktens = self.ktens
        integrator = self.integrator
        exp = self.exp
        self.NetcdfFile.write_experiment(exp)
        exp2 = self.NetcdfFile.read_experiment(desc, [ktens], integrator)
        num_step = len(exp.times) - 1
        self.assertEqual(len(exp2.times), num_step+1)
        for i in range(num_step+1):
            self.assertEqual(exp2.times[i], exp.times[i])
        self.assertEqual(exp2.raws.shape, exp.raws.shape)
        for i in range(len(exp.raws.flat)):
            self.assertEqual(exp2.raws.flat[i], exp.raws.flat[i])
        self.assertIsNone(exp2.ddsddt)
        self.assertIsNone(exp2.zeta_cov)

    def test_complex_experiment_io(self):
        const = self.constants
        grid = self.grid
        desc = self.desc
        ktens = self.ktens
        integrator = self.integrator
        self.NetcdfFile.write_constants(const)
        self.NetcdfFile.write_mass_grid(grid)
        exp = integrator.integrate(integrator.dt*2., self.state, [ktens])
        self.NetcdfFile.write_experiment(exp)
        exp2 = self.NetcdfFile.read_experiment(desc, [ktens], integrator)
        num_step = len(exp.times) - 1
        self.assertEqual(len(exp2.times), num_step+1)
        for i in range(num_step+1):
            self.assertEqual(exp2.times[i], exp.times[i])
        self.assertEqual(exp2.raws.shape, exp.raws.shape)
        for i in range(len(exp.raws.flat)):
            self.assertEqual(exp2.raws.flat[i], exp.raws.flat[i])
        self.assertEqual(exp2.ddsddt.shape, exp.ddsddt.shape)
        scale = exp.ddsddt.max()
        for i in range(len(exp.ddsddt.flat)):
            self.assertAlmostEqual(exp2.ddsddt.flat[i] / scale,
                                   exp.ddsddt.flat[i] / scale)
        self.assertEqual(exp2.zeta_cov.shape, exp.zeta_cov.shape)
        scale = exp.zeta_cov.max()
        for i in range(len(exp.zeta_cov.flat)):
            self.assertAlmostEqual(exp2.zeta_cov.flat[i] / scale,
                                   exp.zeta_cov.flat[i] / scale)

    def test_full_experiment_io(self):
        const = self.constants
        grid = self.grid
        desc = self.desc
        ktens = self.ktens
        integrator = self.integrator
        exp = integrator.integrate(integrator.dt*2., self.state, [ktens])
        self.NetcdfFile.write_full_experiment(exp, ["k1.nc", "k2.nc"])
        exp2 = self.NetcdfFile.read_full_experiment([ktens])
        files = self.NetcdfFile.read_characters('proc_tens_files')
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], "k1.nc")
        self.assertEqual(files[1], "k2.nc")
        self.assertEqual(exp2.desc.perturb_num, exp.desc.perturb_num)
        self.assertIs(exp2.proc_tens[0], ktens)
        self.assertEqual(exp2.integrator.dt, exp.integrator.dt)
        self.assertEqual(exp2.num_time_steps, exp.num_time_steps)
