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

import unittest

from bin_model import ModelConstants, GeometricMassGrid
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.basis import *
from .array_assert import ArrayTestCase


class TestBasisFunction(ArrayTestCase):
    """
    Test BasisFunction methods.
    """
    def test_equality_polynomial(self):
        """Test comparing two PolynomialOnInterval objects for equality."""
        lower_bound = -1.
        upper_bound = 0.
        degree = 3
        root = 0.5
        scale = 2.
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                   scale)
        poi2 = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                   scale)
        self.assertEqual(poi, poi2)
        coi = PolynomialOnInterval(lower_bound, upper_bound, 0, scale=scale)
        coi2 = PolynomialOnInterval(lower_bound, upper_bound, 0, scale=scale)
        self.assertEqual(coi, coi2)
        self.assertNotEqual(poi, coi)
        poi_diff = PolynomialOnInterval(99., upper_bound, degree, root,
                                        scale)
        self.assertNotEqual(poi, poi_diff)
        poi_diff = PolynomialOnInterval(lower_bound, 99., degree, root,
                                        scale)
        self.assertNotEqual(poi, poi_diff)
        poi_diff = PolynomialOnInterval(lower_bound, upper_bound, degree, 99.,
                                        scale)
        self.assertNotEqual(poi, poi_diff)
        poi_diff = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                        99.)
        self.assertNotEqual(poi, poi_diff)

    def test_equality_delta(self):
        """Test comparing two DiracDelta objects for equality."""
        location = 1.
        scale = 2.
        delta = DiracDelta(location, scale=scale)
        delta2 = DiracDelta(location, scale=scale)
        self.assertEqual(delta, delta2)
        delta_diff = DiracDelta(99., scale=scale)
        self.assertNotEqual(delta, delta_diff)
        delta_diff = DiracDelta(location, scale=99.)
        self.assertNotEqual(delta, delta_diff)

    def test_non_equal_types(self):
        """Test comparing two different types of basis function for equality."""
        lower_bound = -1.
        upper_bound = 0.
        degree = 3
        root = 0.5
        scale = 2.
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                   scale)
        location = 1.
        scale = 2.
        delta = DiracDelta(location, scale=scale)
        self.assertNotEqual(poi, delta)

    def test_from_parameters_raises_on_bad_type_string(self):
        """Test that from_parameters raises an error given a bad type string."""
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("nonsense", [], [])

    def test_polynomial_on_interval_from_parameters(self):
        """Test getting PolynomialOnInterval from from_parameters."""
        type_string = "PolynomialOnInterval"
        lower_bound = -1.
        upper_bound = 0.
        degree = 3
        root = 0.5
        scale = 2.
        actual = BasisFunction.from_parameters(type_string, [degree],
                                               [lower_bound, upper_bound,
                                                root, scale])
        expected = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                        scale=scale)
        self.assertEqual(actual, expected)

    def test_constant_on_interval_from_parameters(self):
        """Test getting constant PolynomialOnInterval from from_parameters."""
        type_string = "PolynomialOnInterval"
        lower_bound = -1.
        upper_bound = 0.
        degree = 0
        scale = 2.
        actual = BasisFunction.from_parameters(type_string, [degree],
                                               [lower_bound, upper_bound, scale])
        expected = PolynomialOnInterval(lower_bound, upper_bound, degree,
                                        scale=scale)
        self.assertEqual(actual, expected)

    def test_poly_from_parameters_raises_on_wrong_parameter_number(self):
        """Test from_parameters errors raised for PolynomialOnInterval."""
        # Too few integer params.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [],
                                          [0., 1., 0.5, 2.])
        # Too many integer params.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [2, 3],
                                          [0., 1., 0.5, 2.])
        # Too few real params for constant.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [0],
                                          [0., 1.])
        # Too many real params for constant.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [0],
                                          [0., 1., 0.5, 2.])
        # Too few real params for polynomial.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [3],
                                          [0., 1., 2.])
        # Too many real params for polynomial.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("PolynomialOnInterval", [3],
                                          [0., 1., 0.5, 2., 3.])

    def test_dirac_delta_from_parameters(self):
        """Test getting DiracDelta from from_parameters."""
        type_string = "DiracDelta"
        location = 1.
        scale = 2.
        actual = BasisFunction.from_parameters(type_string, [],
                                               [location, scale])
        expected = DiracDelta(location, scale=scale)
        self.assertEqual(actual, expected)

    def test_delta_from_parameters_raises_on_wrong_parameter_number(self):
        """Test from_parameters errors raised for DiracDelta."""
        # Too many integer params.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("DiracDelta", [0], [1., 2.])
        # Too few real params.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("DiracDelta", [], [1.])
        # Too many real params.
        with self.assertRaises(ValueError):
            BasisFunction.from_parameters("DiracDelta", [], [1., 2., 3.])


class TestPolynomialOnInterval(ArrayTestCase):
    """
    Test PolynomialOnInterval objects.
    """
    def test_root_of_constant_raises(self):
        """Test defining root of a constant basis function raises an error."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 0
        root = 3.
        with self.assertRaises(RuntimeError):
            poi = PolynomialOnInterval(lower_bound, upper_bound, degree, root)

    def test_constant_on_interval(self):
        """Test constant basis function over an interval."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 0
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree)
        self.assertEqual(poi.lower_bound, lower_bound)
        self.assertEqual(poi.upper_bound, upper_bound)
        self.assertEqual(poi.degree, degree)
        self.assertIsNone(poi.root)
        self.assertEqual(poi.scale, 1.)
        inputs = np.array((1., 2., 3., 4., 5., 6.))
        expected = np.array((0., 1., 1., 1., 1., 0.))
        actual = np.array([poi(x) for x in inputs])
        self.assertArrayEqual(actual, expected)

    def test_constant_on_interval_with_scale(self):
        """Test constant basis function over an interval."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 0
        scale = 2.
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree,
                                   scale=scale)
        self.assertEqual(poi.lower_bound, lower_bound)
        self.assertEqual(poi.upper_bound, upper_bound)
        self.assertEqual(poi.degree, degree)
        self.assertIsNone(poi.root)
        self.assertEqual(poi.scale, scale)
        inputs = np.array((1., 2., 3., 4., 5., 6.))
        expected = np.array((0., 1., 1., 1., 1., 0.)) * scale
        actual = np.array([poi(x) for x in inputs])
        self.assertArrayEqual(actual, expected)

    def test_no_root_of_poly_raises(self):
        """Test that a non-constant polynomial requires supplying a root."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 3
        with self.assertRaises(RuntimeError):
            poi = PolynomialOnInterval(lower_bound, upper_bound, degree)

    def test_poly_on_interval(self):
        """Test non-constant polynomial basis function over an interval."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 3
        root = 3.
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree, root)
        self.assertEqual(poi.lower_bound, lower_bound)
        self.assertEqual(poi.upper_bound, upper_bound)
        self.assertEqual(poi.degree, degree)
        self.assertEqual(poi.root, root)
        self.assertEqual(poi.scale, 1.)
        inputs = np.array((1., 2., 3., 4., 5., 6.))
        expected = np.array((0., -1., 0., 1., 8., 0.))
        actual = np.array([poi(x) for x in inputs])
        self.assertArrayEqual(actual, expected)

    def test_poly_on_interval_with_scale(self):
        """Test non-constant polynomial basis function with scaling amount."""
        lower_bound = 2.
        upper_bound = 5.
        degree = 3
        root = 3.
        scale = 2.
        poi = PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                   scale=scale)
        self.assertEqual(poi.lower_bound, lower_bound)
        self.assertEqual(poi.upper_bound, upper_bound)
        self.assertEqual(poi.degree, degree)
        self.assertEqual(poi.root, root)
        self.assertEqual(poi.scale, scale)
        inputs = np.array((1., 2., 3., 4., 5., 6.))
        expected = np.array((0., -1., 0., 1., 8., 0.)) * scale
        actual = np.array([poi(x) for x in inputs])
        self.assertArrayEqual(actual, expected)

    def test_integer_parameters(self):
        """Test output of the integer_parameters function."""
        poi = PolynomialOnInterval(0., 1., 3, 0.5)
        self.assertEqual(poi.integer_parameters(), [3]) # returns degree only

    def test_real_parameters_constant(self):
        """Test output of the real_parameters function for a constant."""
        poi = PolynomialOnInterval(0., 1., 0)
        self.assertEqual(poi.real_parameters(), [0., 1., 1.])
        poi = PolynomialOnInterval(0., 1., 0, scale=2.)
        self.assertEqual(poi.real_parameters(), [0., 1., 2.])

    def test_real_parameters_higher_order(self):
        """Test output of the real_parameters function for a higher-order polynomial."""
        poi = PolynomialOnInterval(0., 1., 3, 0.5)
        self.assertEqual(poi.real_parameters(), [0., 1., 0.5, 1.])
        poi = PolynomialOnInterval(0., 1., 3, 0.5, scale=2.)
        self.assertEqual(poi.real_parameters(), [0., 1., 0.5, 2.])

class TestDiracDelta(ArrayTestCase):
    """
    Test DiracDelta objects.
    """
    def test_dirac_delta(self):
        """Check that expected attributes are set."""
        location = 1.
        delta = DiracDelta(location)
        self.assertEqual(delta.lower_bound, location)
        self.assertEqual(delta.upper_bound, location)
        self.assertEqual(delta.location, location)
        self.assertEqual(delta.scale, 1.)
        scale = 2.
        delta = DiracDelta(location, scale=scale)
        self.assertEqual(delta.lower_bound, location)
        self.assertEqual(delta.upper_bound, location)
        self.assertEqual(delta.location, location)
        self.assertEqual(delta.scale, scale)

    def test_dirac_delta_call_raises(self):
        """Check that calling a DiracDelta raises TypeError."""
        location = 1.
        delta = DiracDelta(location)
        with self.assertRaises(TypeError):
            delta(1.)

    def test_integer_parameters(self):
        """Test output of the integer_parameters function."""
        delta = DiracDelta(1., 2.)
        self.assertEqual(delta.integer_parameters(), [])

    def test_real_parameters(self):
        """Test output of the real_parameters function."""
        delta = DiracDelta(1.)
        self.assertEqual(delta.real_parameters(), [1., 1.])
        delta = DiracDelta(1., 2.)
        self.assertEqual(delta.real_parameters(), [1., 2.])


class TestPiecewisePolynomialBasis(ArrayTestCase):
    """
    Test methods for a basis created by make_piecewise_polynomial_basis.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.num_bins = 2
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def _check_basis_for_degree(self, basis, degree):
        bb = self.grid.bin_bounds
        nb = self.num_bins
        for i in range(nb):
            # Check that inputs below/above the bin bounds are zero.
            self.assertEqual(basis[nb*degree+i](bb[0] - 1.), 0.)
            self.assertEqual(basis[nb*degree+i](bb[-1] + 1.), 0.)
            # Values in the middle of the relevant bin are 0.5 to the power of
            # the degree, and zero elsewhere.
            for k in range(self.num_bins):
                if k != i:
                    self.assertEqual(basis[nb*degree+i](0.5*(bb[k]+bb[k+1])), 0.)
                else:
                    self.assertAlmostEqual(basis[nb*degree+i](0.5*(bb[k]+bb[k+1])),
                                           0.5**degree)

    def test_piecewise_constant(self):
        """Test basis functions that are piecewise constant in log mass."""
        basis = make_piecewise_polynomial_basis(self.grid, 0)
        self.assertEqual(basis.size, self.num_bins)
        self._check_basis_for_degree(basis, 0)

    def test_piecewise_linear(self):
        """Test basis functions that are piecewise linear in log mass."""
        basis = make_piecewise_polynomial_basis(self.grid, 1)
        self.assertEqual(basis.size, 2*self.num_bins)
        self._check_basis_for_degree(basis, 0)
        self._check_basis_for_degree(basis, 1)

    def test_piecewise_quadratic(self):
        """Test basis functions that are piecewise quadratic in log mass."""
        basis = make_piecewise_polynomial_basis(self.grid, 2)
        self.assertEqual(basis.size, 3*self.num_bins)
        self._check_basis_for_degree(basis, 0)
        self._check_basis_for_degree(basis, 1)
        self._check_basis_for_degree(basis, 2)


class TestMakeDeltaOnBoundsBasis(ArrayTestCase):
    """
    Test make_delta_on_bounds_basis.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.num_bins = 2
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_make_delta_on_bounds_basis(self):
        basis = make_delta_on_bounds_basis(self.grid)
        self.assertEqual(basis.size, self.num_bins+1)
        for i in range(basis.size):
            bf = basis[i]
            self.assertIsInstance(bf, DiracDelta)
            self.assertEqual(bf.location, self.grid.bin_bounds[i])
            self.assertEqual(bf.scale, 1.)


class TestBasis(ArrayTestCase):
    """
    Test methods of Basis objects.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.num_bins = 2
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_add(self):
        """Test adding two Basis objects to make a combined Basis."""
        basis1 = make_piecewise_polynomial_basis(self.grid, 0)
        basis2 = make_delta_on_bounds_basis(self.grid)
        new_basis = basis1 + basis2
        self.assertEqual(new_basis.size, basis1.size + basis2.size)
        for i in range(basis1.size):
            self.assertEqual(new_basis[i], basis1[i])
        for i in range(basis2.size):
            self.assertEqual(new_basis[basis1.size+i], basis2[i])

    def test_len(self):
        """Test that len(basis) returns basis.size."""
        basis = make_piecewise_polynomial_basis(self.grid, 0)
        self.assertEqual(len(basis), basis.size)

    def test_finding_indices_of_subset(self):
        """Test finding the indices of one subset within another."""
        basis1 = make_piecewise_polynomial_basis(self.grid, 0)
        basis2 = make_delta_on_bounds_basis(self.grid)
        new_basis = basis1 + basis2
        indices = new_basis.indices_of_subset(basis1)
        self.assertEqual(len(indices), len(basis1))
        for i, idx in enumerate(indices):
            self.assertEqual(new_basis[idx], basis1[i])
        indices = new_basis.indices_of_subset(basis2)
        self.assertEqual(len(indices), len(basis2))
        for i, idx in enumerate(indices):
            self.assertEqual(new_basis[idx], basis2[i])

    def test_finding_indices_of_subset_failed(self):
        """Test that finding the indices of a non-subset returns None."""
        basis1 = make_piecewise_polynomial_basis(self.grid, 0)
        basis2 = make_delta_on_bounds_basis(self.grid)
        self.assertIsNone(basis1.indices_of_subset(basis2))
