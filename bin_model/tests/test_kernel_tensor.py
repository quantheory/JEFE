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

import numpy as np

from bin_model import ModelConstants, LongKernel, GeometricMassGrid
from bin_model.kernel_tensor import KernelTensor
from .array_assert import ArrayTestCase


class TestKernelTensor(ArrayTestCase):
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

    def test_ktens_init(self):
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
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[0,0,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[1]
        ly2 = bb[2]
        lz1 = bb[3]
        lz2 = bb[4]
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[0,1,1], expected / ktens.scaling)
        expected = self.kernel.integrate_over_bins((ly1, ly2), (lx1, lx2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[1,0,1], expected / ktens.scaling)
        lx1 = bb[5]
        lx2 = bb[6]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[5,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = bb[6]
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[0,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[6]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[0,5,1], expected / ktens.scaling)

    def test_ktens_init_scaling(self):
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

    def test_ktens_init_invalid_boundary_raises(self):
        with self.assertRaises(ValueError):
            ktens = KernelTensor(self.kernel, self.grid, boundary='nonsense')

    def test_ktens_init_boundary(self):
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
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
        self.assertAlmostEqual(ktens.data[5,5,0], expected / ktens.scaling)
        lx1 = bb[0]
        lx2 = bb[1]
        ly1 = bb[5]
        ly2 = bb[6]
        lz1 = bb[5]
        lz2 = np.inf
        expected = self.kernel.integrate_over_bins((lx1, lx2), (ly1, ly2),
                                                   (lz1, lz2))
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
        with self.assertRaises(ValueError):
            ktens.calc_rate(f)
        with self.assertRaises(ValueError):
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
