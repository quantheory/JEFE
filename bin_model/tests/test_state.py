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

"""Test state module."""

import numpy as np
import scipy.linalg as la
from scipy.special import gammainc

from bin_model import ModelConstants, LongKernel, GeometricMassGrid, \
    KernelTensor, LogTransform, DerivativeVar, PerturbedVar, \
    ModelStateDescriptor
from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.state import *
from .array_assert import ArrayTestCase


class TestModelState(ArrayTestCase):
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv('lambda')
        self.assertAlmostEqual(actual_fallout_deriv, fallout_deriv[0])
        actual_fallout_deriv = state.fallout_deriv('nu')
        self.assertAlmostEqual(actual_fallout_deriv, fallout_deriv[1])

    def test_fallout_deriv_individual_scaling(self):
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
        dsd = np.linspace(0., nb-1, nb)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                 fallout_deriv=fallout_deriv)
        state = ModelState(desc, raw)
        actual_fallout_deriv = state.fallout_deriv('lambda')
        self.assertAlmostEqual(actual_fallout_deriv, fallout_deriv[0])
        actual_fallout_deriv = state.fallout_deriv('nu')
        self.assertAlmostEqual(actual_fallout_deriv, fallout_deriv[1])

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
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter),
                      DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
            deriv_plus_fallout[:nb] = \
                deriv_vars[i].si_to_nondimensional(dsd_deriv[i,:]) \
                / dsd_scale
            deriv_plus_fallout[nb] = \
                deriv_vars[i].si_to_nondimensional(fallout_deriv[i]) \
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter),
                      DerivativeVar('nu')]
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars)
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
        for i, dvar in enumerate(deriv_vars):
            auto_deriv[i+1] = dvar.nondimensional_to_si(auto_deriv[i+1])
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
        for i, dvar in enumerate(deriv_vars):
            accr_deriv[i+1] = dvar.nondimensional_to_si(accr_deriv[i+1])
        for i in range(3):
            self.assertAlmostEqual(actual_deriv[1,i] / accr_deriv[i], 1.)

    def test_perturb_cov(self):
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), 2.*scale),
            PerturbedVar('L9', wv9, LogTransform(), 3.*scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
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
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter),
                      DerivativeVar('nu')]
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), scale),
            PerturbedVar('L9', wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
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
            deriv_plus_fallout[:nb] = \
                deriv_vars[i].si_to_nondimensional(dsd_deriv[i,:]) \
                / dsd_scale
            deriv_plus_fallout[nb] = \
                deriv_vars[i].si_to_nondimensional(fallout_deriv[i]) \
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
            wv = perturbed_vars[i].weight_vector
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
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter)]
        nvar = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), scale),
            PerturbedVar('L9', wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(nvar)
        correction_time = 5.
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
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
            deriv_plus_fallout[:nb] = \
                deriv_vars[i].si_to_nondimensional(dsd_deriv[i,:]) \
                / dsd_scale
            deriv_plus_fallout[nb] = \
                deriv_vars[i].si_to_nondimensional(fallout_deriv[i]) \
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
            wv = perturbed_vars[i].weight_vector
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
        dvn = 2
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter),
                      DerivativeVar('nu')]
        pn = 3
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        wv9 = grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), scale),
            PerturbedVar('L9', wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(pn)
        desc = ModelStateDescriptor(self.constants,
                                    self.grid,
                                    deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
                                    perturbation_rate=perturbation_rate)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(grid, lam, nu)
        dsd_deriv = np.zeros((dvn, nb))
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
        lf_jac = np.zeros((pn, dvn+1))
        for i in range(pn):
            wv = desc.perturbed_vars[i].weight_vector
            lfs[i], lf_jac[i,:] = state.linear_func_raw(wv, derivative=True,
                                                        dfdt=ddsddt_raw)
        transform_mat = np.diag(
            [desc.perturbed_vars[i].transform.derivative(lfs[i])
             for i in range(pn)]
        )
        v_to_zeta = la.pinv(transform_mat @ lf_jac)
        perturb_cov = desc.perturb_cov_raw(state.raw)
        expected = v_to_zeta @ perturb_cov @ v_to_zeta.T
        self.assertEqual(actual.shape, expected.shape)
        for i in range(len(expected.flat)):
            self.assertAlmostEqual(actual.flat[i] / expected.flat[i], 1.)
