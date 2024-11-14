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

"""Test Process classes."""

from bin_model import ModelConstants, LongKernel, GeometricMassGrid, \
    ConstantReconstruction, CollisionTensor
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.process import *

from .array_assert import ArrayTestCase


class TestCollisionCoalescence(ArrayTestCase):
    """
    Tests of CollisionCoalescence.
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
        self.recon = ConstantReconstruction(self.grid)
        self.basis = self.recon.output_basis()
        self.ctens = CollisionTensor(self.grid, basis=self.basis,
                                     ckern=self.ckern)

    def test_calc_rate_first_bin(self):
        """Check CollisionTensor.calc_rate for lowest bin."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        f[:nb] /= self.grid.bin_widths
        # Expected value in first bin.
        idx = 0
        expected_bot = 0
        for i in range(nb):
            for j in range(self.ctens.nums[idx,i]):
                expected_bot -= self.ctens.data[j,idx,i] * f[idx] * f[i]
        self.assertAlmostEqual(dfdt[0], expected_bot, places=15)

    def test_calc_rate_middle_bin(self):
        """Check CollisionTensor.calc_rate for one of the middle bins."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        f[:nb] /= self.grid.bin_widths
        # Expected value in fourth bin.
        idx = 3
        expected_middle = 0
        for i in range(nb):
            for j in range(self.ctens.nums[idx,i]):
                expected_middle -= self.ctens.data[j,idx,i] * f[idx] * f[i]
            for j in range(nb):
                k_idx = idx - self.ctens.idxs[i,j]
                if 0 <= k_idx < self.ctens.nums[i,j]:
                    expected_middle += self.ctens.data[k_idx,i,j] * f[i] * f[j]
        self.assertAlmostEqual(dfdt[3], expected_middle, places=15)

    def test_calc_rate_top_bins(self):
        """Check CollisionTensor.calc_rate for top and out-of-range bin."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        f[:nb] /= self.grid.bin_widths
        # Expected value in top bin.
        idx = 5
        expected_top = 0
        for i in range(nb):
            for j in range(self.ctens.nums[idx,i]):
                expected_top -= self.ctens.data[j,idx,i] * f[idx] * f[i]
            for j in range(nb):
                k_idx = idx - self.ctens.idxs[i,j]
                if 0 <= k_idx < self.ctens.nums[i,j]:
                    expected_top += self.ctens.data[k_idx,i,j] * f[i] * f[j]
        # Expected value in out-of-range bin.
        idx = 6
        expected_extra = 0
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - self.ctens.idxs[i,j]
                if 0 <= k_idx < self.ctens.nums[i,j]:
                    expected_extra += self.ctens.data[k_idx,i,j] * f[i] * f[j]
        self.assertAlmostEqual(dfdt[5], expected_top, places=15)
        self.assertAlmostEqual(dfdt[6], expected_extra, places=15)

    def test_calc_rate_conservation(self):
        """Check CollisionTensor.calc_rate conserves mass."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        self.assertAlmostEqual(np.sum(dfdt/dfdt.max()), 0.)

    def test_calc_rate_closed(self):
        """Check CollisionTensor.calc_rate with closed boundary."""
        nb = self.num_bins
        ctens = CollisionTensor(self.grid, boundary='closed', ckern=self.ckern)
        proc = CollisionCoalescence(self.recon, ctens)
        f = np.linspace(1., nb, nb)
        dfdt = proc.calc_rate(f)
        self.assertEqual(f.shape, dfdt.shape)
        f /= self.grid.bin_widths
        # Expected value in first bin.
        idx = 0
        expected_bot = 0
        for i in range(nb):
            for j in range(ctens.nums[idx,i]):
                expected_bot -= ctens.data[j,idx,i] * f[idx] * f[i]
        # Expected value in fourth bin.
        idx = 3
        expected_middle = 0
        for i in range(nb):
            for j in range(ctens.nums[idx,i]):
                expected_middle -= ctens.data[j,idx,i] * f[idx] * f[i]
        for i in range(nb):
            for j in range(nb):
                k_idx = idx - ctens.idxs[i,j]
                num = ctens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_middle += ctens.data[k_idx,i,j] * f[i] * f[j]
        # Expected value in top bin.
        idx = 5
        expected_top = 0
        for i in range(nb-1):
            for j in range(nb):
                k_idx = idx - ctens.idxs[i,j]
                num = ctens.nums[i,j]
                if 0 <= k_idx < num:
                    expected_top += ctens.data[k_idx,i,j] * f[i] * f[j]
        self.assertAlmostEqual(dfdt[0], expected_bot, places=18)
        self.assertAlmostEqual(dfdt[3], expected_middle, places=17)
        self.assertAlmostEqual(dfdt[5], expected_top, places=16)
        self.assertAlmostEqual(np.sum(dfdt/dfdt.max()), 0.)

    def test_calc_rate_correct_sizes(self):
        """Check that CollisionTensor.calc_rate uses input vector size."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+2, nb+2)
        with self.assertRaises(ValueError):
            proc.calc_rate(f)
        with self.assertRaises(ValueError):
            proc.calc_rate(f[:nb-1])
        dfdt_1 = proc.calc_rate(f[:nb+1])
        dfdt_2 = proc.calc_rate(f[:nb])
        self.assertEqual(len(dfdt_1), nb+1)
        self.assertEqual(len(dfdt_2), nb)
        for i in range(nb):
            self.assertEqual(dfdt_1[i], dfdt_2[i])

    def test_calc_rate_no_closed_boundary_flux(self):
        """Check calc_rate produces no out-of-range mass for closed boundary."""
        nb = self.num_bins
        ctens = CollisionTensor(self.grid, boundary='closed', ckern=self.ckern)
        proc = CollisionCoalescence(self.recon, ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        self.assertEqual(dfdt[-1], 0.)

    def test_calc_rate_correct_shapes(self):
        """Check that CollisionTensor.calc_rate produces correct output shapes."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        dfdt = proc.calc_rate(f)
        # Row vector.
        f_row = np.reshape(f, (1, nb+1))
        dfdt_row = proc.calc_rate(f_row)
        self.assertArrayAlmostEqual(dfdt_row,
                                    np.reshape(dfdt, f_row.shape),
                                    places=25)
        # Column vector
        f_col = np.reshape(f, (nb+1, 1))
        dfdt_col = proc.calc_rate(f_col)
        self.assertArrayAlmostEqual(dfdt_col,
                                    np.reshape(dfdt, f_col.shape),
                                    places=25)
        f = np.linspace(1., nb, nb)
        dfdt = proc.calc_rate(f)
        # Row vector, no out-of-range bin.
        f_row = np.reshape(f, (1, nb))
        dfdt_row = proc.calc_rate(f_row)
        self.assertArrayAlmostEqual(dfdt_row,
                                    np.reshape(dfdt, f_row.shape),
                                    places=25)
        # Column vector, no out-of-range bin.
        f_col = np.reshape(f, (nb, 1))
        dfdt_col = proc.calc_rate(f_col)
        self.assertArrayAlmostEqual(dfdt_col,
                                    np.reshape(dfdt, f_col.shape),
                                    places=25)

    def test_calc_rate_does_not_change_f(self):
        """Check that CollisionTensor.calc_rate doesn't change its input."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        expected = f.copy()
        proc.calc_rate(f)
        self.assertArrayEqual(f, expected)

    def test_calc_rate_force_out_flux(self):
        """Check CollisionTensor.calc_rate affected by out_flux=True."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        actual = proc.calc_rate(f[:nb], out_flux=True)
        expected = proc.calc_rate(f[:nb+1])
        self.assertArrayEqual(actual, expected)

    def test_calc_rate_force_no_out_flux(self):
        """Check CollisionTensor.calc_rate is affected by out_flux=False."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(1., nb+1, nb+1)
        actual = proc.calc_rate(f, out_flux=False)
        expected = proc.calc_rate(f[:nb])
        self.assertArrayEqual(actual, expected)

    def test_calc_rate_derivative_fd(self):
        """Check derivative output of calc_rate against finite difference."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(2., nb+1, nb+1)
        perturb_size = 1.e-7
        rate, rate_deriv = proc.calc_rate(f, derivative=True)
        self.assertEqual(rate_deriv.shape, (nb+1, nb+1))
        # First column, perturbation in lowest bin.
        idx = 0
        f2 = f.copy()
        f2[idx] += perturb_size
        rate2 = proc.calc_rate(f2)
        expected = (rate2 - rate) / perturb_size
        self.assertArrayAlmostEqual(rate_deriv[:,0], expected)

    def test_calc_rate_derivative(self):
        """Check derivative output of calc_rate."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(2., nb+1, nb+1)
        _, rate_deriv = proc.calc_rate(f, derivative=True)
        f[:nb] /= self.grid.bin_widths
        self.assertEqual(rate_deriv.shape, (nb+1, nb+1))
        # First column, perturbation in lowest bin.
        idx = 0
        expected = np.zeros((nb+1,))
        # Effect of increased fluxes out of this bin.
        for i in range(nb):
            for j in range(self.ctens.nums[idx,i]):
                this_rate = self.ctens.data[j,idx,i] * f[i]
                expected[idx] -= this_rate
                expected[self.ctens.idxs[idx,i] + j] += this_rate
        # Effect of increased transfer of mass out of other bins.
        # (Including this one; the double counting is correct.)
        for i in range(nb):
            for j in range(self.ctens.nums[i,idx]):
                this_rate = self.ctens.data[j,i,idx] * f[i]
                expected[i] -= this_rate
                expected[self.ctens.idxs[i,idx] + j] += this_rate
        expected /= self.grid.bin_widths[idx]
        self.assertArrayAlmostEqual(rate_deriv[:,idx], expected, places=15)
        # Last column, perturbation in highest bin.
        idx = 5
        expected = np.zeros((nb+1,))
        # Effect of increased fluxes out of this bin.
        for i in range(nb):
            for j in range(self.ctens.nums[idx,i]):
                this_rate = self.ctens.data[j,idx,i] * f[i]
                expected[idx] -= this_rate
                expected[self.ctens.idxs[idx,i] + j] += this_rate
        # Effect of increased transfer of mass out of other bins.
        # (Including this one; the double counting is correct.)
        for i in range(nb):
            for j in range(self.ctens.nums[i,idx]):
                this_rate = self.ctens.data[j,i,idx] * f[i]
                expected[i] -= this_rate
                expected[self.ctens.idxs[i,idx] + j] += this_rate
        expected /= self.grid.bin_widths[idx]
        self.assertArrayAlmostEqual(rate_deriv[:,idx], expected, places=15)
        # Effect of perturbations to bottom bin should be 0.
        self.assertArrayEqual(rate_deriv[:,6], np.zeros((nb+1,)))

    def test_calc_rate_derivative_no_out_flux(self):
        """Check derivative output of calc_rate with out_flux=False."""
        nb = self.num_bins
        proc = CollisionCoalescence(self.recon, self.ctens)
        f = np.linspace(2., nb+1, nb+1)
        _, rate_deriv = proc.calc_rate(f, derivative=True, out_flux=False)
        self.assertEqual(rate_deriv.shape, (nb, nb))
        _, outflux_rate_deriv = proc.calc_rate(f, derivative=True)
        self.assertArrayAlmostEqual(rate_deriv,
                                    outflux_rate_deriv[:nb, :nb],
                                    places=25)
