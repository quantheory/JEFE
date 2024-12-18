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

import unittest

from bin_model import ModelConstants, LongKernel, HallKernel, \
    GeometricMassGrid, CollisionTensor, IdentityTransform, LogTransform, \
    QuadToLogTransform, DerivativeVar, PerturbedVar, ModelStateDescriptor, \
    ModelState, RK45Integrator, ForwardEulerIntegrator, RK4Integrator, \
    RadauIntegrator, BackwardEulerIntegrator, Dirk2Integrator, \
    StochasticPerturbation, make_piecewise_polynomial_basis, \
    ConstantReconstruction, CollisionCoalescence
from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.netcdf import *
from .array_assert import ArrayTestCase


class TestNetcdfFile(ArrayTestCase):
    """
    Test NetcdfFile methods.
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
        self.basis = make_piecewise_polynomial_basis(self.grid, 1)
        self.ckern = LongKernel(self.constants)
        self.ctens = CollisionTensor(self.grid, ckern=self.ckern)
        self.recon = ConstantReconstruction(self.grid)
        self.proc = CollisionCoalescence(self.recon, self.ctens)
        deriv_vars = [DerivativeVar('lambda', 1./self.constants.diameter_scale),
                      DerivativeVar('nu')]
        nvar = 3
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
        perturbation_rate = error_rate**2 * np.eye(nvar)
        correction_time = 5.
        self.perturb = \
            StochasticPerturbation(self.constants, perturbed_vars,
                                   perturbation_rate, correction_time)
        self.desc = ModelStateDescriptor(self.constants,
                                         self.grid, deriv_vars=deriv_vars,
                                         perturbed_vars=perturbed_vars)
        nu = 5.
        lam = nu / 1.e-4
        bbd = self.grid.bin_bounds_d
        dsd = gamma_dist_d(bbd, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(bbd, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(bbd, lam, nu)
        self.raw = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        self.state = ModelState(self.desc, self.raw)
        nu2 = 0.
        lam = nu / 5.e-5
        dsd = gamma_dist_d(bbd, lam, nu)
        dsd_deriv = np.zeros((2, nb))
        dsd_deriv[0,:] = gamma_dist_d_lam_deriv(bbd, lam, nu)
        dsd_deriv[1,:] = gamma_dist_d_nu_deriv(bbd, lam, nu)
        raw2 = self.desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        raws = np.zeros((2, len(self.raw)))
        raws[0,:] = self.raw
        raws[1,:] = raw2
        self.times = np.array([0., 1.])
        dt = 15.
        self.integrator = RK45Integrator(self.constants, dt)
        self.fe_integrator = ForwardEulerIntegrator(self.constants, dt)
        self.rk4_integrator = RK4Integrator(self.constants, dt)
        self.radau_integrator = RadauIntegrator(self.constants, dt)
        self.be_integrator = BackwardEulerIntegrator(self.constants, dt)
        self.dirk2_integrator = Dirk2Integrator(self.constants, dt)
        self.exp = Experiment(self.desc, [self.proc], self.integrator,
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

    def test_read_write_scalar_type(self):
        x = 25.
        self.NetcdfFile.write_scalar('x', x, 'f8', '1',
                                     'A description')
        actual = self.NetcdfFile.read_scalar('x')
        self.assertIsInstance(actual, float)

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
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            self.NetcdfFile.write_characters(
                'strings', [string, string2], ['string_len', 'string_num'],
                'A description')

    def test_read_write_characters_array_raises_for_wrong_dimension(self):
        self.NetcdfFile.write_dimension('string_len', 16)
        self.NetcdfFile.write_dimension('string_num', 3)
        string = "Hi"
        string2 = "Bye"
        with self.assertRaises(ValueError):
            self.NetcdfFile.write_characters(
                'strings', [string, string2], ['string_len', 'string_num'],
                'A description')

    def test_read_write_characters_array_raises_for_too_many_dimensions(self):
        self.NetcdfFile.write_dimension('string_len', 2)
        self.NetcdfFile.write_dimension('string_num', 2)
        self.NetcdfFile.write_dimension('lang_num', 2)
        string = "Hi"
        string2 = "By"
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            self.NetcdfFile.write_characters(
                'strings', strings, ['string_num', 'string_len'],
                'A description')

    def test_basis_io(self):
        basis = self.basis
        self.NetcdfFile.write_basis(basis)
        basis2 = self.NetcdfFile.read_basis(self.grid)
        self.assertEqual(basis.size, basis2.size)
        for i in range(basis.size):
            self.assertEqual(basis[i], basis2[i])

    def test_constants_io(self):
        const = self.constants
        self.NetcdfFile.write_constants(const)
        const2 = self.NetcdfFile.read_constants()
        self.assertEqual(const.rho_water, const2.rho_water)
        self.assertEqual(const.rho_air, const2.rho_air)
        self.assertEqual(const.diameter_scale, const2.diameter_scale)
        self.assertEqual(const.rain_d, const2.rain_d)
        self.assertEqual(const.mass_conc_scale, const2.mass_conc_scale)
        self.assertEqual(const.time_scale, const2.time_scale)

    def test_long_kernel_io(self):
        ckern = self.ckern
        self.NetcdfFile.write_collision_kernel(ckern)
        ckern2 = self.NetcdfFile.read_collision_kernel(self.constants)
        self.assertEqual(ckern.kc, ckern2.kc)
        self.assertEqual(ckern.kr, ckern2.kr)
        self.assertEqual(ckern.log_rain_m, ckern2.log_rain_m)

    def test_hall_kernel_io(self):
        ckern = HallKernel(self.constants, 'ScottChen')
        self.NetcdfFile.write_collision_kernel(ckern)
        ckern2 = self.NetcdfFile.read_collision_kernel(self.constants)
        self.assertEqual(ckern.efficiency_name, ckern2.efficiency_name)

    def test_bad_kernel_type_raises(self):
        self.NetcdfFile.write_dimension('collision_kernel_type_str_len',
                                        CollisionKernel.collision_kernel_type_str_len)
        self.NetcdfFile.write_characters('collision_kernel_type',
                                         'nonsense',
                                         'collision_kernel_type_str_len',
                                         'Type of collision kernel')
        with self.assertRaises(RuntimeError):
            self.NetcdfFile.read_collision_kernel(self.constants)

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

    def test_ctens_io(self):
        ctens = self.ctens
        self.NetcdfFile.write_mass_grid(self.grid)
        self.NetcdfFile.write_collision_tensor(ctens)
        ctens2 = self.NetcdfFile.read_collision_tensor(self.grid)
        self.assertEqual(ctens2.boundary, ctens.boundary)
        self.assertEqual(ctens2.data.shape, ctens.data.shape)
        scale = ctens.data.max()
        for i in range(len(ctens.data.flat)):
            self.assertAlmostEqual(ctens2.data.flat[i] / scale,
                                   ctens.data.flat[i] / scale)

    def test_tensor_and_metadata_io(self):
        const = self.constants
        ckern = self.ckern
        grid = self.grid
        basis = self.ctens.basis
        ctens = self.ctens
        self.NetcdfFile.write_tensor_and_metadata(ctens)
        const2, ckern2, grid2, basis2, ctens2 = self.NetcdfFile.read_tensor_and_metadata()
        self.assertEqual(const2.rho_water, const.rho_water)
        self.assertEqual(ckern2.kc, ckern.kc)
        self.assertEqual(grid2.d_min, grid.d_min)
        for i in range(len(basis)):
            self.assertEqual(basis[i], basis2[i])
        self.assertEqual(ctens2.data.shape, ctens.data.shape)
        scale = ctens.data.max()
        for i in range(len(ctens.data.flat)):
            self.assertAlmostEqual(ctens2.data.flat[i] / scale,
                                   ctens.data.flat[i] / scale)

    def test_desc_io(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        desc = self.desc
        self.NetcdfFile.write_tensor_and_metadata(self.ctens)
        self.NetcdfFile.write_descriptor(desc)
        desc2 = self.NetcdfFile.read_descriptor(const, grid)
        self.assertEqual(desc2.deriv_var_num, desc.deriv_var_num)
        for i in range(desc.deriv_var_num):
            self.assertEqual(desc2.deriv_vars[i].name,
                             desc.deriv_vars[i].name)
            self.assertEqual(desc2.deriv_vars[i].scale,
                             desc.deriv_vars[i].scale)
        self.assertEqual(desc2.perturbed_num, desc.perturbed_num)
        for i in range(desc.perturbed_num):
            self.assertEqual(desc2.perturbed_vars[i].name,
                             desc.perturbed_vars[i].name)
            for j in range(nb):
                self.assertEqual(desc2.perturbed_vars[i].weight_vector[j],
                                 desc.perturbed_vars[i].weight_vector[j])
            self.assertEqual(desc2.perturbed_vars[i].transform.transform(2.),
                             desc.perturbed_vars[i].transform.transform(2.))
            self.assertEqual(desc2.perturbed_vars[i].scale,
                             desc.perturbed_vars[i].scale)

    def test_desc_io_bad_transform_type(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        desc = self.desc
        self.NetcdfFile.write_tensor_and_metadata(self.ctens)
        self.NetcdfFile.write_descriptor(desc)
        ttsl = self.NetcdfFile.read_dimension("transform_type_str_len")
        self.NetcdfFile.nc['perturbed_transform_types'][0,:] = \
            nc4.stringtochar(np.array(['nonsense'], 'S{}'.format(ttsl)))
        with self.assertRaises(ValueError):
            self.NetcdfFile.read_descriptor(const, grid)

    def test_simple_desc_io(self):
        nb = self.grid.num_bins
        const = self.constants
        grid = self.grid
        self.NetcdfFile.write_tensor_and_metadata(self.ctens)
        desc = ModelStateDescriptor(const, grid)
        self.NetcdfFile.write_descriptor(desc)
        desc2 = self.NetcdfFile.read_descriptor(const, grid)
        self.assertEqual(desc2.deriv_var_num, desc.deriv_var_num)
        self.assertEqual(desc2.perturbed_num, desc.perturbed_num)

    def test_integrator_io(self):
        const = self.constants
        integrator = self.integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, RK45Integrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_forward_euler_integrator_io(self):
        const = self.constants
        integrator = self.fe_integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, ForwardEulerIntegrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_rk4_integrator_io(self):
        const = self.constants
        integrator = self.rk4_integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, RK4Integrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_radau_integrator_io(self):
        const = self.constants
        integrator = self.radau_integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, RadauIntegrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_backward_euler_integrator_io(self):
        const = self.constants
        integrator = self.be_integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, BackwardEulerIntegrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_dirk2_integrator_io(self):
        const = self.constants
        integrator = self.dirk2_integrator
        self.NetcdfFile.write_integrator(integrator)
        integrator2 = self.NetcdfFile.read_integrator(const)
        self.assertIsInstance(integrator2, Dirk2Integrator)
        self.assertEqual(integrator.dt, integrator2.dt)

    def test_bad_integrator_type_raises(self):
        const = self.constants
        integrator = self.integrator
        self.NetcdfFile.write_integrator(integrator)
        itsl = self.NetcdfFile.read_dimension("integrator_type_str_len")
        self.NetcdfFile.nc['integrator_type'][:] = \
            nc4.stringtochar(np.array(['nonsense'], 'S{}'.format(itsl)))
        with self.assertRaises(RuntimeError):
            integrator2 = self.NetcdfFile.read_integrator(const)

    def test_simple_experiment_io(self):
        desc = self.desc
        integrator = self.integrator
        exp = self.exp
        self.NetcdfFile.write_experiment(exp)
        exp2 = self.NetcdfFile.read_experiment(desc, [self.proc], integrator)
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
        integrator = self.integrator
        self.NetcdfFile.write_constants(const)
        self.NetcdfFile.write_mass_grid(grid)
        exp = integrator.integrate(integrator.dt*2., self.state, [self.proc],
                                   self.perturb)
        self.NetcdfFile.write_experiment(exp)
        exp2 = self.NetcdfFile.read_experiment(desc, [self.proc], integrator)
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
        integrator = self.integrator
        exp = integrator.integrate(integrator.dt*2., self.state, [self.proc],
                                   self.perturb)
        self.NetcdfFile.write_full_experiment(exp, ["k1.nc", "k2.nc"])
        exp2 = self.NetcdfFile.read_full_experiment([self.proc])
        files = self.NetcdfFile.read_characters('proc_tens_files')
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], "k1.nc")
        self.assertEqual(files[1], "k2.nc")
        self.assertEqual(exp2.desc.perturbed_num, exp.desc.perturbed_num)
        self.assertIs(exp2.proc_tens[0], self.proc)
        self.assertEqual(exp2.integrator.dt, exp.integrator.dt)
        self.assertEqual(exp2.num_time_steps, exp.num_time_steps)
