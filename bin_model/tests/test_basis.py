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

from bin_model import ModelConstants, GeometricMassGrid, PiecewisePolyBasis
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.netcdf import *
from .array_assert import ArrayTestCase


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
        for i in range(self.num_bins):
            # Check that inputs below/above the bin bounds are zero.
            self.assertEqual(basis[i][degree](bb[0] - 1.), 0.)
            self.assertEqual(basis[i][degree](bb[-1] + 1.), 0.)
            # Values in the middle of the relevant bin are 0.5 to the power of
            # the degree, and zero elsewhere.
            for k in range(self.num_bins):
                if k != i:
                    self.assertEqual(basis[i][degree](0.5*(bb[k]+bb[k+1])), 0.)
                else:
                    self.assertAlmostEqual(basis[i][degree](0.5*(bb[k]+bb[k+1])),
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
