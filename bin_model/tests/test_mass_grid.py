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

import unittest

from bin_model.constants import ModelConstants
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.mass_grid import *


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
