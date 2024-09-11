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

"""Test MassGrid classes."""

from bin_model.constants import ModelConstants
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.mass_grid import *

from .array_assert import ArrayTestCase


class TestMassGrid(ArrayTestCase):
    """
    Tests of MassGrid methods and attributes.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        # This will put lx bin boundaries at 10^-6, 10^-5, ..., 10^3.
        self.geo_grid = GeometricMassGrid(self.constants,
                                          d_min=1.e-6,
                                          d_max=1.e-3,
                                          num_bins=9)
        # Mass-doubling grid.
        self.md_grid = GeometricMassGrid(self.constants,
                                         d_min=1.e-6,
                                         d_max=2.e-6,
                                         num_bins=3)

    def test_find_bin(self):
        """Check that find_bin works on masses within the grid."""
        for i in range(9):
            lx = np.log(10.**(-5.5+i))
            self.assertEqual(self.geo_grid.find_bin(lx), i)

    def test_find_bin_lower_edge(self):
        """Check that find_bin works for masses smaller than all bins."""
        lx = np.log(10.**-6.5)
        self.assertEqual(self.geo_grid.find_bin(lx), -1)
        lx = np.log(10.**-10)
        self.assertEqual(self.geo_grid.find_bin(lx), -1)

    def test_find_bin_upper_edge(self):
        """Check that find_bin works for masses larger than all bins."""
        lx = np.log(10.**3.5)
        self.assertEqual(self.geo_grid.find_bin(lx), 9)
        lx = np.log(10.**10)
        self.assertEqual(self.geo_grid.find_bin(lx), 9)

    def test_find_sum_bins_geo(self):
        """Check that find_sum_bins works on a grid with geometric spacing."""
        grid = self.geo_grid
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

    def test_find_sum_bins_mass_doubling(self):
        """Check that find_sum_bins can return num=1 on a mass-doubling grid."""
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

    def test_find_sum_bins_irregular(self):
        """Check that find_sum_bins works on an irregular grid with num > 2."""
        # Irregular grid for which MassGrid can span many boundaries.
        irreg_bounds = [np.log(0.5)] \
            + [np.log(1. + (0.1 * i)) for i in range(12)]
        grid = MassGrid(self.constants, bin_bounds=np.array(irreg_bounds))
        lx1 = grid.bin_bounds[0]
        lx2 = grid.bin_bounds[1]
        ly1 = grid.bin_bounds[0]
        ly2 = grid.bin_bounds[1]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 1)
        self.assertEqual(num, 10)

    def test_find_sum_bins_exceeding_range(self):
        """Check find_sum_bins when output is partly outside grid."""
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

    def test_find_sum_bins_all_exceeding_range(self):
        """Check find_sum_bins is correct when all output is outside grid."""
        # Finer than mass-doubling grid needed to guarantee zero overlap.
        grid = GeometricMassGrid(self.constants,
                                 d_min=1.e-6,
                                 d_max=2.e-6,
                                 num_bins=6)
        lx1 = grid.bin_bounds[5]
        lx2 = grid.bin_bounds[6]
        ly1 = grid.bin_bounds[5]
        ly2 = grid.bin_bounds[6]
        idx, num = grid.find_sum_bins(lx1, lx2, ly1, ly2)
        self.assertEqual(idx, 6)
        self.assertEqual(num, 1)

    def test_construct_sparsity_pattern(self):
        """Check construct_sparsity_pattern for a simple mass-doubling grid."""
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
        self.assertArrayEqual(idxs, expected_idxs)
        self.assertArrayEqual(nums, expected_nums)
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_closed_boundary(self):
        """Check construct_sparsity_pattern for a "closed" upper boundary."""
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
        self.assertArrayEqual(idxs, expected_idxs)
        self.assertArrayEqual(nums, expected_nums)
        self.assertEqual(max_num, expected_max_num)

    def test_construct_sparsity_pattern_invalid_boundary_raises(self):
        """Check construct_sparsity_pattern error for invalid boundary."""
        with self.assertRaises(ValueError):
            self.geo_grid.construct_sparsity_structure(boundary='nonsense')


class TestMomentWeightVector(ArrayTestCase):
    """
    Tests of MassGrid.moment_weight_vector.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        # Irregular grid spanning cloud/rain boundary.
        wv_bounds_d = [2.5e-5, 7.5e-5, 1.25e-4, 1.75e-4]
        wv_bounds = np.log(np.array([
            self.constants.diameter_to_scaled_mass(d) for d in wv_bounds_d
        ]))
        self.wv_grid = MassGrid(self.constants, wv_bounds)

    def test_moment_weight_vector_cloud_only_and_rain_only_raises(self):
        """Check moment_weight_vector disallows cloud_only with rain_only."""
        with self.assertRaises(RuntimeError):
            self.wv_grid.moment_weight_vector(3,
                                              cloud_only=True, rain_only=True)

    def test_moment_weight_vector_moment_3(self):
        """Check moment_weight_vector for 3rd moment."""
        actual = self.wv_grid.moment_weight_vector(3)
        self.assertArrayAlmostEqual(actual, np.ones((self.wv_grid.num_bins,)))

    def test_moment_weight_vector_moment_3_cloud_only(self):
        """Check moment_weight_vector for 3rd cloud moment."""
        actual = self.wv_grid.moment_weight_vector(3, cloud_only=True)
        expected = np.array([
            1.,
            (np.log(self.constants.rain_m) - self.wv_grid.bin_bounds[1])
            / self.wv_grid.bin_widths[1],
            0.,
        ])
        self.assertArrayAlmostEqual(actual, expected)

    def test_moment_weight_vector_moment_3_rain_only(self):
        """Check moment_weight_vector for 3rd rain moment."""
        actual = self.wv_grid.moment_weight_vector(3, rain_only=True)
        expected = np.array([
            0.,
            (self.wv_grid.bin_bounds[2] - np.log(self.constants.rain_m))
            / self.wv_grid.bin_widths[1],
            1.,
        ])
        self.assertArrayAlmostEqual(actual, expected)

    def test_moment_weight_vector_moment_0(self):
        """Check moment_weight_vector for 0th moment."""
        actual = self.wv_grid.moment_weight_vector(0)
        exp_bb = np.exp(-self.wv_grid.bin_bounds)
        expected = (exp_bb[:-1] - exp_bb[1:]) / self.wv_grid.bin_widths
        self.assertArrayAlmostEqual(actual, expected)

    def test_moment_weight_vector_moment_0_cloud_only(self):
        """Check moment_weight_vector for 0th cloud moment."""
        actual = self.wv_grid.moment_weight_vector(0, cloud_only=True)
        expected = np.array([
            (np.exp(-self.wv_grid.bin_bounds[0])
            - np.exp(-self.wv_grid.bin_bounds[1])) / self.wv_grid.bin_widths[0],
            (np.exp(-self.wv_grid.bin_bounds[1])
            - 1./self.constants.rain_m) / self.wv_grid.bin_widths[1],
            0.,
        ])
        self.assertArrayAlmostEqual(actual, expected)

    def test_moment_weight_vector_moment_0_rain_only(self):
        """Check moment_weight_vector for 0th rain moment."""
        actual = self.wv_grid.moment_weight_vector(0, rain_only=True)
        expected = np.array([
            0.,
            (1./self.constants.rain_m
            - np.exp(-self.wv_grid.bin_bounds[2])) / self.wv_grid.bin_widths[1],
            (np.exp(-self.wv_grid.bin_bounds[2])
            - np.exp(-self.wv_grid.bin_bounds[3])) / self.wv_grid.bin_widths[2],
        ])
        self.assertArrayAlmostEqual(actual, expected)


class TestGeometricMassGrid(ArrayTestCase):
    """
    Tests of GeometricMassGrid methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.d_min = 1.e-6
        self.d_max = 1.e-3
        self.num_bins = 90
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=self.d_min,
                                      d_max=self.d_max,
                                      num_bins=self.num_bins)

    def test_geometric_grid_init_scalars(self):
        """Check scalars constructed by GeometricMassGrid constructor."""
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
        """Check arrays constructed by GeometricMassGrid constructor."""
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
