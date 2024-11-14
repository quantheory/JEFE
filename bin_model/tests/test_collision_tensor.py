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

"""Test collision_tensor module."""

import numpy as np

from bin_model import ModelConstants, LongKernel, GeometricMassGrid
from bin_model.basis import PolynomialOnInterval
from bin_model.collision_tensor import CollisionTensor
from .array_assert import ArrayTestCase


class TestCollisionTensor(ArrayTestCase):
    """
    Tests of CollisionTensor methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.scaling = self.constants.mass_conc_scale \
            * self.constants.time_scale \
            / self.constants.std_mass
        self.ckern = LongKernel(self.constants)
        self.num_bins = 6
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_ctens_init_raises_without_kernel_or_data(self):
        """Check CollisionTensor.__init__ raises without kernel information."""
        with self.assertRaises(RuntimeError):
            CollisionTensor(self.grid)

    def test_ctens_init_raises_with_both_kernel_and_data(self):
        """Check CollisionTensor.__init__ raises with both kernel and data set."""
        with self.assertRaises(RuntimeError):
            CollisionTensor(self.grid, ckern=self.ckern, data=np.zeros((2,2)))

    def test_ctens_init(self):
        """Check data produced by CollisionTensor.__init__ with kernel input."""
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ctens = CollisionTensor(self.grid, ckern=self.ckern)
        self.assertEqual(ctens.data.shape, (2, nb, nb))
        # Check the following six cases:
        # 1. Two equal bins' output to the smallest nonzero output bin.
        # 2. y > x, output to a larger output bin.
        # 3. x > y, output to a larger output bin.
        # 4. x and y are largest bin, output to out-of-range bin.
        # 5. y >> x, output to y bin.
        # 6. y is largest bin, output to out-of-range bin.
        x_idxs = [0, 0, 1, 5, 0, 0]
        y_idxs = [0, 1, 0, 5, 5, 5]
        z_idxs = [2, 3, 3, 6, 5, 6]
        for x_idx, y_idx, z_idx, in zip(x_idxs, y_idxs, z_idxs):
            basis_x = PolynomialOnInterval(bb[x_idx], bb[x_idx+1], 0)
            basis_y = PolynomialOnInterval(bb[y_idx], bb[y_idx+1], 0)
            ly_bound = bb[y_idx:y_idx+2]
            if z_idx < nb:
                lz_bound = bb[z_idx:z_idx+2]
            else:
                lz_bound = (bb[z_idx], np.inf)
            k = z_idx - ctens.idxs[x_idx,y_idx]
            expected = self.ckern.integrate_over_bins(basis_x, basis_y,
                                                      lz_bound)
            self.assertAlmostEqual(ctens.data[k,x_idx,y_idx],
                                   expected * self.scaling)

    def test_ctens_init_scaling(self):
        """Check application of dimension scalings in CollisionTensor.__init__."""
        const = ModelConstants(rho_water=1000.,
                               rho_air=1.2,
                               diameter_scale=1.e-4,
                               rain_d=1.e-4,
                               mass_conc_scale = 2.,
                               time_scale=3.)
        ckern = LongKernel(const)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=2.e-6,
                                 num_bins=self.num_bins)
        ctens = CollisionTensor(grid, ckern=ckern)
        const_noscale = ModelConstants(rho_water=1000.,
                                       rho_air=1.2,
                                       diameter_scale=1.e-4,
                                       rain_d=1.e-4,
                                       mass_conc_scale=1.,
                                       time_scale=1.)
        ckern = LongKernel(const)
        grid = GeometricMassGrid(const,
                                 d_min=1.e-6,
                                 d_max=2.e-6,
                                 num_bins=self.num_bins)
        ctens_noscale = CollisionTensor(grid, ckern=ckern)
        self.assertArrayAlmostEqual(ctens.data,
                                    ctens_noscale.data
                                    * const_noscale.mass_conc_scale
                                    * const_noscale.time_scale)

    def test_ctens_init_invalid_boundary_raises(self):
        """Check CollisionTensor.__init__ raises error for invalid boundary."""
        with self.assertRaises(ValueError):
            CollisionTensor(self.grid, boundary='nonsense', ckern=self.ckern)

    def test_ctens_init_boundary_closed(self):
        """Check CollisionTensor.__init__ with boundary=closed."""
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ctens = CollisionTensor(self.grid, boundary='closed', ckern=self.ckern)
        self.assertEqual(ctens.data.shape, (2, nb, nb))
        # Check the following two cases:
        # 1. x and y are largest bin, output to largest bin.
        # 2. y is largest bin, output to largest bin.
        x_idxs = [5, 0]
        y_idxs = [5, 5]
        z_idxs = [5, 5]
        for x_idx, y_idx, z_idx in zip(x_idxs, y_idxs, z_idxs):
            basis_x = PolynomialOnInterval(bb[x_idx], bb[x_idx+1], 0)
            basis_y = PolynomialOnInterval(bb[y_idx], bb[y_idx+1], 0)
            if z_idx < nb-1:
                lz_bound = bb[z_idx:z_idx+2]
            else:
                lz_bound = (bb[z_idx], np.inf)
            k = z_idx - ctens.idxs[x_idx,y_idx]
            expected = self.ckern.integrate_over_bins(basis_x, basis_y,
                                                      lz_bound)
            self.assertAlmostEqual(ctens.data[k,x_idx,y_idx],
                                   expected * self.scaling)
