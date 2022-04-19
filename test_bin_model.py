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
                                        std_diameter=1.e-4)

    def test_diameter_to_scaled_mass(self):
        self.assertAlmostEqual(self.constants.diameter_to_scaled_mass(1.e-5),
                               1.e-3)

    def test_scaled_mass_to_diameter(self):
        self.assertAlmostEqual(self.constants.scaled_mass_to_diameter(1.e-3),
                               1.e-5)

    def test_std_mass(self):
        self.assertAlmostEqual(self.constants.std_mass,
                               self.constants.rho_water * np.pi/6. * 1.e-12)

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
                                        std_diameter=1.e-4)

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
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_cloud_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_cloud(b, a, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_cloud_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_cloud(a, b, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_cloud_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_cloud(1., b, lxm, lxp) \
                    - reference_long_cloud(1., a, lxm, lxp)
        actual = kernel._integral_cloud(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=12)

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
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_rain_type_1(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = -reference_long_rain(b, a, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=1)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_rain_type_2(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -2.
        lxp = -1.
        expected = reference_long_rain(a, b, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=2)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_integral_rain_type_3(self):
        kernel = LongKernel(self.constants)
        a = -1.
        b = 0.
        lxm = -3.
        lxp = -2.
        expected = reference_long_rain(1., b, lxm, lxp) \
                    - reference_long_rain(1., a, lxm, lxp)
        actual = kernel._integral_rain(a, b, lxm, lxp, btype=3)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_calculate_edges_invalid(self):
        kernel = LongKernel(self.constants)
        ly1 = 1.
        ly2 = 0.
        lz1 = 2.
        lz2 = 3.
        with self.assertRaises(AssertionError):
            actual = kernel._calculate_edges(ly1, ly2, lz1, lz2)
        ly1 = 0.
        ly2 = 1.
        lz1 = 3.
        lz2 = 2.
        with self.assertRaises(AssertionError):
            actual = kernel._calculate_edges(ly1, ly2, lz1, lz2)

    def test_calculate_edges_bins_separated(self):
        kernel = LongKernel(self.constants)
        ly1 = 0.
        ly2 = 1.
        lz1 = 2.
        lz2 = 3.
        expected = (sub_logs(lz1, ly1), sub_logs(lz1, ly2),
                    sub_logs(lz2, ly1), sub_logs(lz2, ly2))
        actual = kernel._calculate_edges(ly1, ly2, lz1, lz2)
        self.assertAlmostEqual(actual, expected)
