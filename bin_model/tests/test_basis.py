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


class TestPiecewisePolyBasis(ArrayTestCase):
    """
    Test PiecewisePolyBasis methods.
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
        basis = PiecewisePolyBasis(self.grid, 0)
        self.assertEqual(basis.size, self.num_bins)
        self.assertEqual(basis.degree, 0)
        self._check_basis_for_degree(basis, 0)

    def test_piecewise_linear(self):
        """Test basis functions that are piecewise linear in log mass."""
        basis = PiecewisePolyBasis(self.grid, 1)
        self.assertEqual(basis.size, 2*self.num_bins)
        self.assertEqual(basis.degree, 1)
        self._check_basis_for_degree(basis, 0)
        self._check_basis_for_degree(basis, 1)

    def test_piecewise_quadratic(self):
        """Test basis functions that are piecewise quadratic in log mass."""
        basis = PiecewisePolyBasis(self.grid, 2)
        self.assertEqual(basis.size, 3*self.num_bins)
        self.assertEqual(basis.degree, 2)
        self._check_basis_for_degree(basis, 0)
        self._check_basis_for_degree(basis, 1)
        self._check_basis_for_degree(basis, 2)
