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

"""Test time module."""

import numpy as np
import scipy.linalg as la

from bin_model import ModelConstants, LongKernel, GeometricMassGrid, \
    CollisionTensor, LogTransform, DerivativeVar, PerturbedVar, \
    ModelStateDescriptor, ModelState, StochasticPerturbation
from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.time import *
from .array_assert import ArrayTestCase


class TestIntegrator(ArrayTestCase):
    """
    Test Integrator methods.
    """
    def test_integrate_raw_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Integrator().integrate_raw(1., 2., 3.)

    def test_to_netcdf_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Integrator().to_netcdf(None)


class IntegratorTestCase(ArrayTestCase):
    """
    Base case for testing Integrator subclasses.
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
        self.ctens = CollisionTensor(self.grid, kernel=self.kernel)
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter),
                      DerivativeVar('nu')]
        self.desc = ModelStateDescriptor(self.constants,
                                         self.grid,
                                         deriv_vars=deriv_vars)
        nu = 5.
        lam = nu / 1.e-4
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(self.grid, lam, nu)
        self.raw = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        self.state = ModelState(self.desc, self.raw)
        dvn = 1
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.std_diameter)]
        pn = 3
        wv0 = self.grid.moment_weight_vector(0)
        wv6 = self.grid.moment_weight_vector(6)
        wv9 = self.grid.moment_weight_vector(9)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), scale),
            PerturbedVar('L9', wv9, LogTransform(), scale),
        ]
        error_rate = 0.5 / 60.
        perturbation_rate = error_rate**2 * np.eye(pn)
        correction_time = 5.
        self.perturb = \
            StochasticPerturbation(self.constants, perturbed_vars,
                                   perturbation_rate, correction_time)
        self.pc_desc = ModelStateDescriptor(self.constants,
                                            self.grid, deriv_vars=deriv_vars,
                                            perturbed_vars=perturbed_vars)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((dvn, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(self.grid, lam, nu)
        fallout_deriv = np.array([dsd_deriv[0,-4:].mean()])
        perturb_cov_init = (10. / np.log(10.)) \
            * (np.ones((pn, pn)) + np.eye(pn))
        self.pc_raw = self.pc_desc.construct_raw(dsd, dsd_deriv=dsd_deriv,
                                      fallout_deriv=fallout_deriv,
                                      perturb_cov=perturb_cov_init)
        self.pc_state = ModelState(self.pc_desc, self.pc_raw)


class TestRK45Integrator(IntegratorTestCase):
    """
    Test RK45Integrator methods and attributes.
    """
    def test_integrate_raw(self):
        tscale = self.constants.time_scale
        dt = 1.e-5
        num_step = 2
        integrator = RK45Integrator(self.constants, dt)
        times, actual = integrator.integrate_raw(num_step*dt / tscale,
                                                 self.state,
                                                 [self.ctens])
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
                + dt_scaled*expect_state.time_derivative_raw([self.ctens])
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
                                   [self.ctens])
        self.assertIs(exp.desc, self.state.desc)
        self.assertEqual(len(exp.proc_tens), 1)
        self.assertIs(exp.proc_tens[0], self.ctens)
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
                + dt_scaled*expect_state.time_derivative_raw([self.ctens])
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
                                   [self.ctens],
                                   self.perturb)
        for i in range(num_step+1):
            actual = exp.ddsddt[i,:]
            expected = exp.states[i].dsd_time_deriv_raw([self.ctens])[:nb]
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


class TestForwardEulerIntegrator(IntegratorTestCase):
    """
    Test ForwardEulerIntegrator methods and attributes.
    """
    def test_integrate_raw(self):
        tscale = self.constants.time_scale
        dt = 0.1
        num_step = 2
        integrator = ForwardEulerIntegrator(self.constants, dt)
        times, actual = integrator.integrate_raw(num_step*dt / tscale,
                                                 self.state,
                                                 [self.ctens])
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
                + dt_scaled*expect_state.time_derivative_raw([self.ctens])
        scale = expected.max()
        for i in range(num_step+1):
            for j in range(len(self.raw)):
                self.assertAlmostEqual(actual[i,j]/scale, expected[i,j]/scale)


class TestRK4Integrator(IntegratorTestCase):
    """
    Test RK4Integrator methods and attributes.
    """
    def test_integrate_raw(self):
        tscale = self.constants.time_scale
        dt = 0.1
        num_step = 2
        integrator = RK4Integrator(self.constants, dt)
        times, actual = integrator.integrate_raw(num_step*dt / tscale,
                                                 self.state,
                                                 [self.ctens])
        expected = np.linspace(0., num_step*dt, num_step+1) / tscale
        self.assertEqual(times.shape, (num_step+1,))
        for i in range(num_step):
            self.assertAlmostEqual(times[i], expected[i])
        self.assertEqual(actual.shape, (num_step+1, len(self.raw)))
        expected = np.zeros((num_step+1, len(self.raw)))
        dt_scaled = dt / tscale
        expected[0,:] = self.raw
        for i in range(num_step):
            stage1_state = ModelState(self.desc, expected[i,:])
            slope1 = stage1_state.time_derivative_raw([self.ctens])
            stage2_state = ModelState(self.desc,
                                      expected[i,:] + 0.5*dt_scaled*slope1)
            slope2 = stage2_state.time_derivative_raw([self.ctens])
            stage3_state = ModelState(self.desc,
                                      expected[i,:] + 0.5*dt_scaled*slope2)
            slope3 = stage3_state.time_derivative_raw([self.ctens])
            stage4_state = ModelState(self.desc,
                                      expected[i,:] + dt_scaled*slope3)
            slope4 = stage4_state.time_derivative_raw([self.ctens])
            expected[i+1,:] = expected[i,:] \
                + (dt_scaled/6.) * (slope1 + 2.*slope2 + 2.*slope3 + slope4)
        scale = expected.max()
        for i in range(num_step+1):
            for j in range(len(self.raw)):
                self.assertAlmostEqual(actual[i,j]/scale, expected[i,j]/scale)
