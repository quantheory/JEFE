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

"""Test kernel_tensor module."""

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
        self.scaling = self.constants.mass_conc_scale \
            * self.constants.time_scale \
            / self.constants.std_mass
        self.kernel = LongKernel(self.constants)
        self.num_bins = 6
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_ktens_init_raises_without_kernel_or_data(self):
        """Check KernelTensor.__init__ raises without kernel information."""
        with self.assertRaises(RuntimeError):
            KernelTensor(self.grid)

    def test_ktens_init_raises_with_both_kernel_and_data(self):
        """Check KernelTensor.__init__ raises with both kernel and data set."""
        with self.assertRaises(RuntimeError):
            KernelTensor(self.grid, kernel=self.kernel, data=np.zeros((2,2)))

    def test_ktens_init(self):
        """Check data produced by KernelTensor.__init__ with kernel input."""
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        self.assertEqual(ktens.data.shape, (nb, nb, 2))
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
            lx_bound = bb[x_idx:x_idx+2]
            ly_bound = bb[y_idx:y_idx+2]
            if z_idx < nb:
                lz_bound = bb[z_idx:z_idx+2]
            else:
                lz_bound = (bb[z_idx], np.inf)
            k = z_idx - ktens.idxs[x_idx,y_idx]
            expected = self.kernel.integrate_over_bins(lx_bound, ly_bound,
                                                       lz_bound)
            self.assertAlmostEqual(ktens.data[x_idx,y_idx,k],
                                   expected * self.scaling)

    def test_ktens_init_scaling(self):
        """Check application of dimension scalings in KernelTensor.__init__."""
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
        ktens = KernelTensor(grid, kernel=kernel)
        const_noscale = ModelConstants(rho_water=1000.,
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
        ktens_noscale = KernelTensor(grid, kernel=kernel)
        self.assertArrayAlmostEqual(ktens.data,
                                    ktens_noscale.data
                                    * const_noscale.mass_conc_scale
                                    * const_noscale.time_scale)

    def test_ktens_init_invalid_boundary_raises(self):
        """Check KernelTensor.__init__ raises error for invalid boundary."""
        with self.assertRaises(ValueError):
            KernelTensor(self.grid, boundary='nonsense', kernel=self.kernel)

    def test_ktens_init_boundary_closed(self):
        """Check KernelTensor.__init__ with boundary=closed."""
        nb = self.num_bins
        bb = self.grid.bin_bounds
        ktens = KernelTensor(self.grid, boundary='closed', kernel=self.kernel)
        self.assertEqual(ktens.data.shape, (nb, nb, 2))
        # Check the following two cases:
        # 1. x and y are largest bin, output to largest bin.
        # 2. y is largest bin, output to largest bin.
        x_idxs = [5, 0]
        y_idxs = [5, 5]
        z_idxs = [5, 5]
        for x_idx, y_idx, z_idx in zip(x_idxs, y_idxs, z_idxs):
            lx_bound = bb[x_idx:x_idx+2]
            ly_bound = bb[y_idx:y_idx+2]
            if z_idx < nb-1:
                lz_bound = bb[z_idx:z_idx+2]
            else:
                lz_bound = (bb[z_idx], np.inf)
            k = z_idx - ktens.idxs[x_idx,y_idx]
            expected = self.kernel.integrate_over_bins(lx_bound, ly_bound,
                                                       lz_bound)
            self.assertAlmostEqual(ktens.data[x_idx,y_idx,k],
                                   expected * self.scaling)

    def test_calc_rate_first_bin(self):
        """Check KernelTensor.calc_rate for lowest bin."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, kernel=self.kernel)
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
        self.assertAlmostEqual(dfdt[0], expected_bot, places=15)

    def test_calc_rate_middle_bin(self):
        """Check KernelTensor.calc_rate for one of the middle bins."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        # Expected value in fourth bin.
        idx = 3
        expected_middle = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_middle -= ktens.data[idx,i,j] * f[idx] * f[i]
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                if 0 <= k_idx < ktens.nums[i,j]:
                    expected_middle += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_middle /= bw[idx]
        self.assertAlmostEqual(dfdt[3], expected_middle, places=15)

    def test_calc_rate_top_bins(self):
        """Check KernelTensor.calc_rate for top and out-of-range bin."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        # Expected value in top bin.
        idx = 5
        expected_top = 0
        for i in range(nb):
            for j in range(ktens.nums[idx,i]):
                expected_top -= ktens.data[idx,i,j] * f[idx] * f[i]
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                if 0 <= k_idx < ktens.nums[i,j]:
                    expected_top += ktens.data[i,j,k_idx] * f[i] * f[j]
        expected_top /= bw[idx]
        # Expected value in out-of-range bin.
        idx = 6
        expected_extra = 0
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ktens.idxs[i,j]
                if 0 <= k_idx < ktens.nums[i,j]:
                    expected_extra += ktens.data[i,j,k_idx] * f[i] * f[j]
        self.assertAlmostEqual(dfdt[5], expected_top, places=15)
        self.assertAlmostEqual(dfdt[6], expected_extra, places=15)

    def test_calc_rate_conservation(self):
        """Check KernelTensor.calc_rate conserves mass."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        mass_change = np.zeros((nb+1,))
        mass_change[:nb] = dfdt[:nb] * bw
        mass_change[-1] = dfdt[-1]
        self.assertAlmostEqual(np.sum(mass_change/mass_change.max()), 0.)

    def test_calc_rate_closed(self):
        """Check KernelTensor.calc_rate with closed boundary."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, boundary='closed', kernel=self.kernel)
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

    def test_calc_rate_correct_sizes(self):
        """Check that KernelTensor.calc_rate uses input vector size."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, kernel=self.kernel)
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
        """Check calc_rate produces no out-of-range mass for closed boundary."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, boundary='closed', kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        self.assertEqual(dfdt[-1], 0.)

    def test_calc_rate_correct_shapes(self):
        """Check that KernelTensor.calc_rate produces correct output shapes."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = ktens.calc_rate(f)
        # Row vector.
        f_row = np.reshape(f, (1, nb+1))
        dfdt_row = ktens.calc_rate(f_row)
        self.assertArrayAlmostEqual(dfdt_row,
                                    np.reshape(dfdt, f_row.shape),
                                    places=25)
        # Column vector
        f_col = np.reshape(f, (nb+1, 1))
        dfdt_col = ktens.calc_rate(f_col)
        self.assertArrayAlmostEqual(dfdt_col,
                                    np.reshape(dfdt, f_col.shape),
                                    places=25)
        f = np.linspace(1., nb, nb)
        dfdt = ktens.calc_rate(f)
        # Row vector, no out-of-range bin.
        f_row = np.reshape(f, (1, nb))
        dfdt_row = ktens.calc_rate(f_row)
        self.assertArrayAlmostEqual(dfdt_row,
                                    np.reshape(dfdt, f_row.shape),
                                    places=25)
        # Column vector, no out-of-range bin.
        f_col = np.reshape(f, (nb, 1))
        dfdt_col = ktens.calc_rate(f_col)
        self.assertArrayAlmostEqual(dfdt_col,
                                    np.reshape(dfdt, f_col.shape),
                                    places=25)

    def test_calc_rate_force_out_flux(self):
        """Check KernelTensor.calc_rate affected by out_flux=True."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        actual = ktens.calc_rate(f[:nb], out_flux=True)
        expected = ktens.calc_rate(f[:nb+1])
        self.assertArrayEqual(actual, expected)

    def test_calc_rate_force_no_out_flux(self):
        """Check KernelTensor.calc_rate is affected by out_flux=False."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(1., nb+1, nb+1)
        actual = ktens.calc_rate(f, out_flux=False)
        expected = ktens.calc_rate(f[:nb])
        self.assertArrayEqual(actual, expected)

    def test_calc_rate_derivative(self):
        """Check derivative output of calc_rate."""
        nb = self.num_bins
        bw = self.grid.bin_widths
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(2., nb+1, nb+1)
        _, rate_deriv = ktens.calc_rate(f, derivative=True)
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
        self.assertArrayAlmostEqual(rate_deriv[:,idx], expected, places=15)
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
        self.assertArrayAlmostEqual(rate_deriv[:,idx], expected, places=15)
        # Effect of perturbations to bottom bin should be 0.
        self.assertArrayEqual(rate_deriv[:,6], np.zeros((nb+1,)))

    def test_calc_rate_derivative_no_out_flux(self):
        """Check derivative output of calc_rate with out_flux=False."""
        nb = self.num_bins
        ktens = KernelTensor(self.grid, kernel=self.kernel)
        f = np.linspace(2., nb+1, nb+1)
        _, rate_deriv = ktens.calc_rate(f, derivative=True, out_flux=False)
        self.assertEqual(rate_deriv.shape, (nb, nb))
        _, outflux_rate_deriv = ktens.calc_rate(f, derivative=True)
        self.assertArrayAlmostEqual(rate_deriv,
                                    outflux_rate_deriv[:nb, :nb],
                                    places=25)
