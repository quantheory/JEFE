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

"""Test experiment module."""

import numpy as np
import scipy.linalg as la

from bin_model import ModelConstants, LongKernel, GeometricMassGrid, \
    CollisionTensor, LogTransform, DerivativeVar, PerturbedVar, \
    ModelStateDescriptor, ModelState, RK45Integrator, StochasticPerturbation
from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.experiment import *
from .array_assert import ArrayTestCase


class TestExperiment(ArrayTestCase):
    """
    Test Experiment methods.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        nb = 30
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=nb)
        self.ckern = LongKernel(self.constants)
        self.ctens = CollisionTensor(self.grid, ckern=self.ckern)
        dvn = 2
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.diameter_scale),
                      DerivativeVar('nu')]
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
        self.desc = ModelStateDescriptor(self.constants,
                                         self.grid,
                                         deriv_vars=deriv_vars,
                                         perturbed_vars=perturbed_vars)
        nu = 5.
        lam = nu / 1.e-3
        dsd = gamma_dist_d(self.grid, lam, nu)
        dsd_deriv = np.zeros((dvn, nb))
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
        dsd_deriv = np.zeros((dvn, nb))
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
        exp = Experiment(self.desc, [self.ctens], self.integrator, times, raws)
        self.assertEqual(exp.constants.rho_air, self.constants.rho_air)
        self.assertEqual(exp.mass_grid.num_bins, self.grid.num_bins)
        self.assertEqual(exp.proc_tens[0].ckern.kc, self.ckern.kc)
        self.assertEqual(exp.proc_tens[0].data.shape, self.ctens.data.shape)
        self.assertTrue(np.all(exp.proc_tens[0].data == self.ctens.data))
        self.assertEqual(exp.times.shape, (ntimes,))
        self.assertEqual(len(exp.states), ntimes)
        for i in range(ntimes):
            self.assertEqual(exp.states[i].dsd_moment(0),
                             states[i].dsd_moment(0))
        self.assertEqual(exp.desc.deriv_var_num, 2)
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
        exp = Experiment(self.desc, [self.ctens], self.integrator, times, raws,
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
        dvn = self.desc.deriv_var_num
        zeta_cov = np.zeros((ntimes, dvn, dvn))
        for i in range(ntimes):
            zeta_cov = np.reshape(np.linspace(50. + i, 50. + (i + dvn**2-1),
                                                dvn**2),
                                    (dvn, dvn))
        exp = Experiment(self.desc, [self.ctens], self.integrator, times, raws,
                         zeta_cov=zeta_cov)
        self.assertEqual(exp.zeta_cov.shape, zeta_cov.shape)
        for i in range(len(zeta_cov.flat)):
            self.assertEqual(exp.zeta_cov.flat[i],
                             zeta_cov.flat[i])

    def test_get_moments_and_covariances(self):
        grid = self.grid
        nb = grid.num_bins
        end_time = 15.
        exp = self.integrator.integrate(end_time, self.state, [self.ctens])
        wvs = np.zeros((2, nb))
        wvs[0,:] = grid.moment_weight_vector(6)
        wvs[1,:] = grid.moment_weight_vector(3, cloud_only=True)
        mom, cov = exp.get_moments_and_covariances(wvs)
        expected_mom = np.zeros((2,2))
        expected_cov = np.zeros((2,2,2))
        for i in range(2):
            deriv = np.zeros((2, self.desc.deriv_var_num+1))
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
        exp = self.integrator.integrate(end_time, self.state, [self.ctens])
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
        exp = self.integrator.integrate(end_time, self.state, [self.ctens])
        wvs = np.zeros((2, nb))
        wvs[0,:] = grid.moment_weight_vector(6)
        wvs[1,:] = grid.moment_weight_vector(3, cloud_only=True)
        mom, cov = exp.get_moments_and_covariances(wvs, times=[1])
        expected_mom = np.zeros((1,2))
        expected_cov = np.zeros((1,2,2))
        deriv = np.zeros((2, self.desc.deriv_var_num+1))
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
        dvn = self.desc.deriv_var_num
        times = self.times
        ntimes = len(times)
        raws = np.zeros((ntimes, len(self.state.raw)))
        raws[0,:] = self.state.raw
        raws[1,:] = self.state2.raw
        states = [self.state, self.state2]
        ddsddt = np.zeros((ntimes, nb))
        for i in range(ntimes):
            ddsddt = np.linspace(-nb+i, -i, nb)
        zeta_cov = np.zeros((ntimes, dvn, dvn))
        for i in range(ntimes):
            zeta_cov = np.reshape(np.linspace(50. + i, 50. + (i + dvn**2-1),
                                                dvn**2),
                                    (dvn, dvn))
        exp = Experiment(self.desc, [self.ctens], self.integrator, times, raws,
                         ddsddt=ddsddt)
        wvs = self.grid.moment_weight_vector(6)
        with self.assertRaises(RuntimeError):
            exp.get_moments_and_covariances(wvs)
        exp = Experiment(self.desc, [self.ctens], self.integrator, times, raws,
                         zeta_cov=zeta_cov)
        with self.assertRaises(RuntimeError):
            exp.get_moments_and_covariances(wvs)
