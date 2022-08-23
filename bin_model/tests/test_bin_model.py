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

from scipy.special import gammainc

from bin_model import *


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
        nchol = (nvar * (nvar + 1)) // 2
        self.assertEqual(desc.state_len(), 3*nb + 3 + nchol)
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

    def test_perturb_chol_loc(self):
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
        self.assertEqual(desc.perturb_chol_loc(),
                         (3*nb+3, (nvar*(nvar+1)) // 2))

    def test_perturb_chol_raw(self):
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
        num_chol = (nvar * (nvar + 1)) // 2
        raw[-num_chol:] = np.linspace(1, num_chol, num_chol)
        actual = desc.perturb_chol_raw(raw)
        expected = np.zeros((nvar, nvar))
        ic = 0
        for i in range(nvar):
            for j in range(i+1):
                expected[i,j] = raw[-num_chol+ic]
                ic += 1
        self.assertEqual(actual.shape, expected.shape)
        for i in range(nvar):
            for j in range(nvar):
                self.assertAlmostEqual(actual[i,j],
                                       expected[i,j])

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
        num_chol = (nvar * (nvar + 1)) // 2
        raw[-num_chol:] = np.linspace(1, num_chol, num_chol)
        actual = desc.perturb_cov_raw(raw)
        expected = np.zeros((nvar, nvar))
        ic = 0
        for i in range(nvar):
            for j in range(i+1):
                expected[i,j] = raw[-num_chol+ic]
                ic += 1
        expected = expected @ expected.T
        self.assertEqual(actual.shape, expected.shape)
        for i in range(nvar):
            for j in range(nvar):
                self.assertAlmostEqual(actual[i,j],
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
        orig = orig + orig.T + 20. * np.eye(nvar)
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

    def test_perturb_cov_construct_raw_defaults_to_near_zero_cov(self):
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
        expected = 1.e-50 * np.eye(nvar)
        for i in range(nvar):
            for j in range(nvar):
                self.assertAlmostEqual(actual[i,j],
                                       expected[i,j],
                                       places=60)


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
        with self.assertRaises(RuntimeError):
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
        nchol = (nvar * (nvar + 1)) // 2
        expected = np.zeros((3*nb+3+nchol,))
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
        perturb_chol = desc.perturb_chol_raw(state.raw)
        cov_rate = la.solve(perturb_chol, cov_rate)
        cov_rate = np.transpose(la.solve(perturb_chol, cov_rate.T))
        for i in range(nvar):
            cov_rate[i,i] *= 0.5
            for j in range(i+1, nvar):
                cov_rate[i,j] = 0.
        cov_rate = perturb_chol @ cov_rate
        ic = 0
        for i in range(nvar):
            for j in range(i+1):
                expected[-nchol+ic] = cov_rate[i,j]
                ic += 1
        self.assertEqual(len(actual), 3*nb+3 + nchol)
        for i in range(3*nb+3 + nchol):
            self.assertAlmostEqual(actual[i], expected[i], places=8)

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
        nchol = (nvar * (nvar + 1)) // 2
        expected = np.zeros((2*nb+2+nchol,))
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
        perturb_chol = desc.perturb_chol_raw(state.raw)
        cov_rate = la.solve(perturb_chol, cov_rate)
        cov_rate = np.transpose(la.solve(perturb_chol, cov_rate.T))
        for i in range(nvar):
            cov_rate[i,i] *= 0.5
            for j in range(i+1, nvar):
                cov_rate[i,j] = 0.
        cov_rate = perturb_chol @ cov_rate
        nchol = (nvar * (nvar + 1)) // 2
        ic = 0
        for i in range(nvar):
            for j in range(i+1):
                expected[-nchol+ic] = cov_rate[i,j]
                ic += 1
        self.assertEqual(len(actual), 2*nb+2 + nchol)
        for i in range(2*nb+2 + nchol):
            self.assertAlmostEqual(actual[i], expected[i], places=9)

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


class TestIdentityTransform(unittest.TestCase):
    """
    Test IdentityTransform methods.
    """
    def test_identity_transform(self):
        self.assertEqual(IdentityTransform().transform(2.), 2.)

    def test_identity_transform_deriv(self):
        self.assertEqual(IdentityTransform().derivative(2.), 1.)

    def test_identity_transform_second_over_first_derivative(self):
        self.assertEqual(IdentityTransform().second_over_first_derivative(2.),
                         0.)

    def test_type_string(self):
        self.assertEqual(IdentityTransform().type_string(), 'Identity')

    def test_get_parameters(self):
        self.assertEqual(IdentityTransform().get_parameters(), [])


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


class TestQuadToLogTransform(unittest.TestCase):
    """
    Test QuadToLogTransform methods.
    """
    def setUp(self):
        self.trans = QuadToLogTransform(0.3)

    def test_quad_to_log_transform(self):
        self.assertEqual(self.trans.transform(0.15), 0.875)
        self.assertEqual(self.trans.transform(2.),
                         np.log(2. / 0.3) + 1.5)

    def test_identity_transform_deriv(self):
        self.assertEqual(self.trans.derivative(0.15), 5.)
        self.assertEqual(self.trans.derivative(2.), 1./2.)

    def test_identity_transform_second_over_first_derivative(self):
        self.assertEqual(self.trans.second_over_first_derivative(0.15),
                         -20./9.)
        self.assertEqual(self.trans.second_over_first_derivative(2.),
                         -1./2.)

    def test_type_string(self):
        self.assertEqual(self.trans.type_string(), 'QuadToLog')

    def test_get_parameters(self):
        self.assertEqual(self.trans.get_parameters(), [0.3])


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

    def test_get_moments_and_covariances(self):
        grid = self.grid
        nb = grid.num_bins
        end_time = 15.
        exp = self.integrator.integrate(end_time, self.state, [self.ktens])
        wvs = np.zeros((2, nb))
        wvs[0,:] = grid.moment_weight_vector(6)
        wvs[1,:] = grid.moment_weight_vector(3, cloud_only=True)
        mom, cov = exp.get_moments_and_covariances(wvs)
        expected_mom = np.zeros((2,2))
        expected_cov = np.zeros((2,2,2))
        for i in range(2):
            deriv = np.zeros((2, self.desc.dsd_deriv_num+1))
            for j in range(2):
                expected_mom[i,j], deriv[j,:] = \
                    exp.states[i].linear_func_raw(wvs[j], derivative=True,
                                                  dfdt=exp.ddsddt[i,:])
            expected_cov[i,:,:] = deriv @ exp.zeta_cov[i,:,:] @ deriv.T
        self.assertEqual(mom.shape, expected_mom.shape)
        for i in range(len(mom.flat)):
            self.assertEqual(mom.flat[i], expected_mom.flat[i])
        self.assertEqual(cov.shape, expected_cov.shape)
        for i in range(len(cov.flat)):
            self.assertEqual(cov.flat[i], expected_cov.flat[i])

    def test_get_moments_and_covariances_single_moment(self):
        grid = self.grid
        nb = grid.num_bins
        end_time = 15.
        exp = self.integrator.integrate(end_time, self.state, [self.ktens])
        wvs = grid.moment_weight_vector(6)
        mom, cov = exp.get_moments_and_covariances(wvs)
        expected_mom = np.zeros((2,))
        expected_cov = np.zeros((2,))
        for i in range(2):
            expected_mom[i], deriv = \
                exp.states[i].linear_func_raw(wvs, derivative=True,
                                              dfdt=exp.ddsddt[i,:])
            expected_cov[i] = deriv @ exp.zeta_cov[i,:,:] @ deriv.T
        self.assertEqual(mom.shape, expected_mom.shape)
        for i in range(len(mom.flat)):
            self.assertEqual(mom.flat[i], expected_mom.flat[i])
        self.assertEqual(cov.shape, expected_cov.shape)
        for i in range(len(cov.flat)):
            self.assertEqual(cov.flat[i], expected_cov.flat[i])

    def test_get_moments_and_covariances_single_time(self):
        grid = self.grid
        nb = grid.num_bins
        end_time = 15.
        exp = self.integrator.integrate(end_time, self.state, [self.ktens])
        wvs = np.zeros((2, nb))
        wvs[0,:] = grid.moment_weight_vector(6)
        wvs[1,:] = grid.moment_weight_vector(3, cloud_only=True)
        mom, cov = exp.get_moments_and_covariances(wvs, times=[1])
        expected_mom = np.zeros((1,2))
        expected_cov = np.zeros((1,2,2))
        deriv = np.zeros((2, self.desc.dsd_deriv_num+1))
        for j in range(2):
            expected_mom[0,j], deriv[j,:] = \
                exp.states[1].linear_func_raw(wvs[j], derivative=True,
                                              dfdt=exp.ddsddt[1,:])
        expected_cov[0,:,:] = deriv @ exp.zeta_cov[1,:,:] @ deriv.T
        self.assertEqual(mom.shape, expected_mom.shape)
        for i in range(len(mom.flat)):
            self.assertEqual(mom.flat[i], expected_mom.flat[i])
        self.assertEqual(cov.shape, expected_cov.shape)
        for i in range(len(cov.flat)):
            self.assertEqual(cov.flat[i], expected_cov.flat[i])

    def test_get_moments_and_covariances_raises_without_data(self):
        nb = self.grid.num_bins
        ddn = self.desc.dsd_deriv_num
        times = self.times
        ntimes = len(times)
        raws = np.zeros((ntimes, len(self.state.raw)))
        raws[0,:] = self.state.raw
        raws[1,:] = self.state2.raw
        states = [self.state, self.state2]
        ddsddt = np.zeros((ntimes, nb))
        for i in range(ntimes):
            ddsddt = np.linspace(-nb+i, -i, nb)
        zeta_cov = np.zeros((ntimes, ddn, ddn))
        for i in range(ntimes):
            zeta_cov = np.reshape(np.linspace(50. + i, 50. + (i + ddn**2-1),
                                                ddn**2),
                                    (ddn, ddn))
        exp = Experiment(self.desc, [self.ktens], self.integrator, times, raws,
                         ddsddt=ddsddt)
        wvs = self.grid.moment_weight_vector(6)
        with self.assertRaises(AssertionError):
            exp.get_moments_and_covariances(wvs)
        exp = Experiment(self.desc, [self.ktens], self.integrator, times, raws,
                         zeta_cov=zeta_cov)
        with self.assertRaises(AssertionError):
            exp.get_moments_and_covariances(wvs)


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

    def test_bad_kernel_type_raises(self):
        self.NetcdfFile.write_dimension('kernel_type_str_len',
                                        Kernel.kernel_type_str_len)
        self.NetcdfFile.write_characters('kernel_type',
                                         'nonsense',
                                         'kernel_type_str_len',
                                         'Type of kernel')
        with self.assertRaises(RuntimeError):
            self.NetcdfFile.read_kernel(self.constants)

    def test_geometric_mass_grid_io(self):
        grid = self.grid
        self.NetcdfFile.write_mass_grid(grid)
        grid2 = self.NetcdfFile.read_mass_grid(self.constants)
        self.assertEqual(grid.d_min, grid2.d_min)
        self.assertEqual(grid.d_max, grid2.d_max)
        self.assertEqual(grid.num_bins, grid2.num_bins)

    def test_mass_grid_io(self):
        num_bins = 2
        bin_bounds = np.linspace(0., num_bins, num_bins+1)
        grid = MassGrid(self.constants, bin_bounds)
        self.NetcdfFile.write_mass_grid(grid)
        grid2 = self.NetcdfFile.read_mass_grid(self.constants)
        self.assertEqual(grid2.num_bins, grid.num_bins)
        for i in range(num_bins+1):
            self.assertEqual(grid2.bin_bounds[i], grid.bin_bounds[i])

    def test_bad_grid_type_raises(self):
        self.NetcdfFile.write_dimension('mass_grid_type_str_len',
                                        MassGrid.mass_grid_type_str_len)
        self.NetcdfFile.write_characters('mass_grid_type',
                                         'nonsense',
                                         'mass_grid_type_str_len',
                                         'Type of mass grid')
        with self.assertRaises(RuntimeError):
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
            (wv6, IdentityTransform(), scale),
            (wv9, QuadToLogTransform(0.3), scale),
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
