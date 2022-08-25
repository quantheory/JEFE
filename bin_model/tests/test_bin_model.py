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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        ktens = KernelTensor(self.grid, kernel=kernel)
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
        self.ktens = KernelTensor(self.grid, kernel=self.kernel)
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
        self.ktens = KernelTensor(self.grid, kernel=self.kernel)
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
        self.ktens = KernelTensor(self.grid, kernel=self.kernel)
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
        ktens2 = self.NetcdfFile.read_kernel_tensor(self.grid)
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
        with self.assertRaises(ValueError):
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
