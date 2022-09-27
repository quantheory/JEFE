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

"""Test descriptor module."""

import numpy as np
import scipy.linalg as la

from bin_model import ModelConstants, GeometricMassGrid, LogTransform
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.descriptor import *
from .array_assert import ArrayTestCase


class TestDerivativeVar(ArrayTestCase):
    """
    Test DerivativeVar methods.
    """

    def setUp(self):
        self.deriv_var = DerivativeVar("lambda", 5.)

    def test_deriv_var_too_long_name(self):
        """Check error raised when DerivativeVar name is too long."""
        with self.assertRaises(ValueError):
            DerivativeVar("a" * (max_variable_name_len+1), 1.)

    def test_deriv_var_matches_name(self):
        """Check matching the name of a derivative variable."""
        self.assertTrue(self.deriv_var.matches("lambda"))
        self.assertFalse(self.deriv_var.matches("lam"))

    def test_deriv_var_si_to_nondimensional(self):
        """Check converting a derivative from SI units to nondimensional."""
        self.assertAlmostEqual(self.deriv_var.si_to_nondimensional(1.), 5.)

    def test_deriv_var_nondimensional_to_si(self):
        """Check converting a derivative from nondimensional units to SI."""
        self.assertAlmostEqual(self.deriv_var.nondimensional_to_si(1.), 0.2)

    def test_deriv_var_default_scale(self):
        """Check that derivative variable scale defaults to 1."""
        dvar = DerivativeVar("lambda")
        self.assertAlmostEqual(dvar.nondimensional_to_si(2.), 2.)
        self.assertAlmostEqual(dvar.si_to_nondimensional(2.), 2.)


class TestPerturbedVar(ArrayTestCase):
    """
    Test PerturbedVar methods.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        # Number of bins in grid.
        self.nb = 90
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=self.nb)
        self.wv = self.grid.moment_weight_vector(0)
        self.perturb_var = PerturbedVar("M0", self.wv, LogTransform, 10.)

    def test_perturb_var_too_long_name(self):
        """Check error raised when PerturbedVar name is too long."""
        with self.assertRaises(ValueError):
            PerturbedVar("a" * (max_variable_name_len+1), self.wv,
                         LogTransform, 10.)


class TestModelStateDescriptor(ArrayTestCase):
    """
    Test ModelStateDescriptor methods and attributes.
    """

    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        std_diameter=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        # Number of bins in grid.
        self.nb = 90
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-3,
                                      num_bins=self.nb)

    def dsd_only_desc(self):
        """Return a descriptor containing a DSD and no other state variables."""
        return ModelStateDescriptor(self.constants, self.grid)

    def lambda_desc(self):
        """Return a descriptor containing one derivative, lambda."""
        deriv_vars = [DerivativeVar('lambda')]
        return ModelStateDescriptor(self.constants, self.grid,
                                    deriv_vars=deriv_vars)

    def lambda_nu_desc(self):
        """Return a descriptor containing two derivatives, lambda and nu."""
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        return ModelStateDescriptor(self.constants, self.grid,
                                    deriv_vars=deriv_vars)

    def lambda_nu_scaled_desc(self):
        """Return a descriptor containing two derivatives with scaling."""
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        self.deriv_var_scales = [3., 4.]
        return ModelStateDescriptor(self.constants, self.grid,
                                    deriv_vars=deriv_vars)

    def perturb_069_desc(self):
        """Return a descriptor with moments 0, 3, and 6 perturbed."""
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        self.deriv_var_scales = [3., 4.]
        self.pn = 3
        scale = 10. / np.log(10.)
        l0 = PerturbedVar("L0", self.grid.moment_weight_vector(0),
                          LogTransform(), scale)
        l6 = PerturbedVar("L6", self.grid.moment_weight_vector(6),
                          LogTransform(), 2.*scale)
        l9 = PerturbedVar("L9", self.grid.moment_weight_vector(9),
                          LogTransform(), 3.*scale)
        self.perturb_scales = [scale, 2.*scale, 3.*scale]
        error_rate = 0.5 / 60.
        self.perturbation_rate = error_rate**2 * np.eye(self.pn)
        return ModelStateDescriptor(self.constants, self.grid,
                                    deriv_vars=deriv_vars,
                                    perturbed_vars=[l0, l6, l9],
                                    perturbation_rate=self.perturbation_rate)

    def test_state_len_dsd_only(self):
        """Check that state length is correct for dsd only descriptor."""
        desc = self.dsd_only_desc()
        self.assertEqual(desc.state_len(), self.nb+1)

    def test_dsd_loc(self):
        """Check dsd position and length."""
        desc = self.dsd_only_desc()
        self.assertEqual(desc.dsd_loc(), (0, self.nb))

    def test_dsd_loc_with_fallout(self):
        """Check dsd+fallout position and length."""
        desc = self.dsd_only_desc()
        self.assertEqual(desc.dsd_loc(with_fallout=True), (0, self.nb+1))

    def test_fallout_loc(self):
        """Check fallout position and length."""
        desc = self.dsd_only_desc()
        self.assertEqual(desc.fallout_loc(), self.nb)

    def test_dsd_raw(self):
        """Check dsd_raw."""
        desc = self.dsd_only_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        actual = desc.dsd_raw(raw)
        idx, num = desc.dsd_loc()
        self.assertArrayEqual(actual, raw[idx:idx+num])

    def test_dsd_raw_with_fallout(self):
        """Check dsd_raw with fallout."""
        desc = self.dsd_only_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        actual = desc.dsd_raw(raw, with_fallout=True)
        idx, num = desc.dsd_loc(with_fallout=True)
        self.assertArrayEqual(actual, raw[idx:idx+num])

    def test_fallout_raw(self):
        """Check fallout_raw."""
        desc = self.dsd_only_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        self.assertEqual(desc.fallout_raw(raw), raw[desc.fallout_loc()])

    def test_construct_raw(self):
        """Check construct_raw for dsd only descriptor."""
        desc = self.dsd_only_desc()
        dsd = np.linspace(0, self.nb, self.nb)
        fallout = 200.
        raw = desc.construct_raw(dsd, fallout=fallout)
        self.assertEqual(len(raw), desc.state_len())
        dsd_scale = self.constants.mass_conc_scale
        self.assertArrayAlmostEqual(desc.dsd_raw(raw), dsd / dsd_scale)
        self.assertAlmostEqual(desc.fallout_raw(raw), fallout / dsd_scale)

    def test_construct_raw_wrong_dsd_size_raises(self):
        """Check construct_raw raises an error for wrong dsd size."""
        desc = self.dsd_only_desc()
        dsd = np.ones((self.nb+1,))
        with self.assertRaises(ValueError):
            desc.construct_raw(dsd)
        dsd = np.ones((self.nb-1,))
        with self.assertRaises(ValueError):
            desc.construct_raw(dsd)

    def test_construct_raw_fallout_default_zero(self):
        """Check that construct_raw has a default fallout of 0."""
        desc = self.dsd_only_desc()
        dsd = np.ones((self.nb,))
        raw = desc.construct_raw(dsd)
        self.assertEqual(desc.fallout_raw(raw), 0.)

    def test_no_derivatives(self):
        """Check that deriv_var_num defaults to 0."""
        desc = self.dsd_only_desc()
        self.assertEqual(desc.deriv_var_num, 0)

    def test_derivatives(self):
        """Check deriv_var_num and state_len with two derivative variables."""
        desc = self.lambda_nu_desc()
        self.assertEqual(desc.deriv_var_num, 2)
        self.assertEqual(desc.state_len(), 3*self.nb+3)

    def test_find_deriv_var_index(self):
        """Check output of find_deriv_var_index."""
        desc = self.lambda_nu_desc()
        self.assertEqual(desc.find_deriv_var_index('lambda'), 0)
        self.assertEqual(desc.find_deriv_var_index('nu'), 1)

    def test_find_deriv_var_index_raises(self):
        """Check that find_deriv_var_index raises an error for invalid name."""
        desc = self.lambda_desc()
        with self.assertRaises(ValueError):
            desc.find_deriv_var_index('nonsense')

    def test_find_deriv_var(self):
        """Check output of find_deriv_var."""
        desc = self.lambda_nu_desc()
        self.assertEqual(desc.find_deriv_var('lambda').name, 'lambda')
        self.assertEqual(desc.find_deriv_var('nu').name, 'nu')

    def test_find_deriv_var_raises(self):
        """Check that find_deriv_var raises an error for invalid name."""
        desc = self.lambda_desc()
        with self.assertRaises(ValueError):
            desc.find_deriv_var('nonsense')

    def test_empty_derivatives(self):
        """Check that setting deriv_vars=[] is the same as not specifying it."""
        desc = ModelStateDescriptor(self.constants, self.grid, deriv_vars=[])
        self.assertEqual(desc.deriv_var_num, 0)

    def test_derivatives_raises_on_duplicate_name(self):
        """Check ValueError raised for duplicate derivative names."""
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('lambda')]
        with self.assertRaises(ValueError):
            desc = ModelStateDescriptor(self.constants, self.grid,
                                        deriv_vars=deriv_vars)

    def test_dsd_deriv_loc_all_without_fallout(self):
        "Check dsd_deriv_loc for with_fallout=False."
        desc = self.lambda_nu_desc()
        idxs, num = desc.dsd_deriv_loc(with_fallout=False)
        self.assertEqual(idxs, [self.nb+1, 2*self.nb+2])
        self.assertEqual(num, self.nb)

    def test_dsd_deriv_loc_all_with_fallout(self):
        """Check dsd_deriv_loc for with_fallout=True."""
        desc = self.lambda_nu_desc()
        idxs, num = desc.dsd_deriv_loc(with_fallout=True)
        self.assertEqual(idxs, [self.nb+1, 2*self.nb+2])
        self.assertEqual(num, self.nb+1)

    def test_dsd_deriv_loc_all_default_fallout(self):
        """Check dsd_deriv_loc defaults to with_fallout=False."""
        desc = self.lambda_nu_desc()
        idxs, num = desc.dsd_deriv_loc()
        idxs_expected, num_expected = desc.dsd_deriv_loc(with_fallout=False)
        self.assertEqual(idxs, idxs_expected)
        self.assertEqual(num, num_expected)

    def test_dsd_deriv_loc_individual_without_fallout(self):
        """Check dsd_deriv_loc with derivative variable specified."""
        desc = self.lambda_nu_desc()
        idx, num = desc.dsd_deriv_loc('lambda', with_fallout=False)
        self.assertEqual(idx, self.nb+1)
        self.assertEqual(num, self.nb)
        idx, num = desc.dsd_deriv_loc('nu', with_fallout=False)
        self.assertEqual(idx, 2*self.nb+2)
        self.assertEqual(num, self.nb)

    def test_dsd_deriv_loc_individual_with_fallout(self):
        """Check dsd_deriv_loc with specific derivative variable and fallout."""
        desc = self.lambda_nu_desc()
        idx, num = desc.dsd_deriv_loc('lambda', with_fallout=True)
        self.assertEqual(idx, self.nb+1)
        self.assertEqual(num, self.nb+1)
        idx, num = desc.dsd_deriv_loc('nu', with_fallout=True)
        self.assertEqual(idx, 2*self.nb+2)
        self.assertEqual(num, self.nb+1)

    def test_dsd_deriv_loc_raises_for_bad_string(self):
        """Check that dsd_deriv_loc raises for bad variable name."""
        desc = self.lambda_desc()
        with self.assertRaises(ValueError):
            desc.dsd_deriv_loc('nonsense')

    def test_fallout_deriv_loc_all(self):
        """Check fallout_deriv_loc."""
        desc = self.lambda_nu_desc()
        idxs = desc.fallout_deriv_loc()
        self.assertEqual(idxs, [2*self.nb+1, 3*self.nb+2])

    def test_fallout_deriv_loc_individual(self):
        """Check fallout_deriv_loc for specific derivative variables."""
        desc = self.lambda_nu_desc()
        idx = desc.fallout_deriv_loc('lambda')
        self.assertEqual(idx, 2*self.nb+1)
        idx = desc.fallout_deriv_loc('nu')
        self.assertEqual(idx, 3*self.nb+2)

    def test_fallout_deriv_loc_raises_for_bad_string(self):
        """Check that fallout_deriv_loc raises for bad variable name."""
        desc = self.lambda_nu_desc()
        with self.assertRaises(ValueError):
            idxs = desc.fallout_deriv_loc('nonsense')

    def test_dsd_deriv_raw_all(self):
        """Check dsd_deriv_raw."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        derivs = desc.dsd_deriv_raw(raw)
        idxs, num = desc.dsd_deriv_loc()
        expected = np.zeros((len(idxs), num))
        for i, idx in enumerate(idxs):
            expected[i,:] = raw[idx:idx+num]
        self.assertArrayEqual(derivs, expected)

    def test_dsd_deriv_raw_all_with_fallout(self):
        """Check dsd_deriv_raw for with_fallout=True."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        derivs = desc.dsd_deriv_raw(raw, with_fallout=True)
        idxs, num = desc.dsd_deriv_loc(with_fallout=True)
        expected = np.zeros((len(idxs), num))
        for i, idx in enumerate(idxs):
            expected[i,:] = raw[idx:idx+num]
        self.assertArrayEqual(derivs, expected)

    def test_dsd_deriv_raw_individual(self):
        """Check dsd_deriv_raw for specific derivative variables."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        deriv = desc.dsd_deriv_raw(raw, 'lambda')
        idx, num = desc.dsd_deriv_loc('lambda')
        expected = raw[idx:idx+num]
        self.assertArrayEqual(deriv, expected)
        deriv = desc.dsd_deriv_raw(raw, 'nu')
        idx, num = desc.dsd_deriv_loc('nu')
        expected = raw[idx:idx+num]
        self.assertArrayEqual(deriv, expected)

    def test_dsd_deriv_raw_individual_with_fallout(self):
        """Check dsd_deriv_raw with fallout for specific variables."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        deriv = desc.dsd_deriv_raw(raw, 'lambda', with_fallout=True)
        idx, num = desc.dsd_deriv_loc('lambda', with_fallout=True)
        expected = raw[idx:idx+num]
        self.assertArrayEqual(deriv, expected)
        deriv = desc.dsd_deriv_raw(raw, 'nu', with_fallout=True)
        idx, num = desc.dsd_deriv_loc('nu', with_fallout=True)
        expected = raw[idx:idx+num]
        self.assertArrayEqual(deriv, expected)

    def test_dsd_deriv_raw_raises_for_bad_string(self):
        """Check dsd_deriv_raw raises error when given bad variable name."""
        desc = self.lambda_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        with self.assertRaises(ValueError):
            desc.dsd_deriv_raw(raw, 'nonsense')

    def test_fallout_deriv_raw_all(self):
        """Check fallout_deriv_raw."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        fallout_derivs = desc.fallout_deriv_raw(raw)
        idxs = desc.fallout_deriv_loc()
        self.assertArrayEqual(fallout_derivs, raw[idxs])

    def test_fallout_deriv_raw_individual(self):
        """Check fallout_deriv_raw for individual derivative variables."""
        desc = self.lambda_nu_desc()
        raw = np.linspace(0., desc.state_len(), desc.state_len())
        fallout_deriv = desc.fallout_deriv_raw(raw, 'lambda')
        self.assertEqual(fallout_deriv, raw[desc.fallout_deriv_loc('lambda')])
        fallout_deriv = desc.fallout_deriv_raw(raw, 'nu')
        self.assertEqual(fallout_deriv, raw[desc.fallout_deriv_loc('nu')])

    def test_construct_raw_with_derivatives(self):
        """Check construct_raw with derivative input."""
        desc = self.lambda_desc()
        dsd = np.linspace(0, self.nb, self.nb)
        fallout = 200.
        dsd_deriv = dsd[None,:] + 1.
        raw = desc.construct_raw(dsd, fallout=fallout, dsd_deriv=dsd_deriv)
        self.assertEqual(len(raw), desc.state_len())
        actual_dsd_deriv = desc.dsd_deriv_raw(raw)
        self.assertArrayAlmostEqual(actual_dsd_deriv,
                                    dsd_deriv / self.constants.mass_conc_scale)

    def test_construct_raw_raises_for_missing_derivative(self):
        """Check construct_raw raises if expected derivatives not provided."""
        desc = self.lambda_desc()
        with self.assertRaises(RuntimeError):
            raw = desc.construct_raw(np.ones(self.nb))

    def test_construct_raw_raises_for_extra_derivative(self):
        """Check construct_raw raises if unexpected derivatives are provided."""
        desc = self.dsd_only_desc()
        with self.assertRaises(RuntimeError):
            raw = desc.construct_raw(np.ones(self.nb),
                                     dsd_deriv=np.ones((1, self.nb)))

    def test_construct_raw_raises_for_wrong_derivative_shape(self):
        """Check construct_raw raises if dsd_deriv is wrong shape."""
        desc = self.lambda_nu_desc()
        with self.assertRaises(ValueError):
            desc.construct_raw(np.ones(self.nb),
                               dsd_deriv=np.ones((2, self.nb+1)))
        with self.assertRaises(ValueError):
            desc.construct_raw(np.ones(self.nb),
                               dsd_deriv=np.ones((3, self.nb)))

    def test_construct_raw_allows_empty_derivative(self):
        """Check construct_raw allows unexpected empty derivatives."""
        desc = self.dsd_only_desc()
        desc.construct_raw(np.ones(self.nb), dsd_deriv=np.ones((0, self.nb)))

    def test_construct_raw_with_derivatives_scaling(self):
        """Check construct_raw with derivatives with scale factors."""
        desc = self.lambda_nu_scaled_desc()
        dsd = np.linspace(0, self.nb, self.nb)
        dsd_deriv = np.zeros((2, self.nb))
        dsd_deriv[0,:] = dsd + 1.
        dsd_deriv[1,:] = dsd + 2.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        actual_dsd_deriv = desc.dsd_deriv_raw(raw)
        scaled_dsd_deriv = dsd_deriv / self.constants.mass_conc_scale
        for i in range(2):
            scaled_dsd_deriv[i,:] *= self.deriv_var_scales[i]
        self.assertArrayAlmostEqual(actual_dsd_deriv, scaled_dsd_deriv)

    def test_construct_raw_with_fallout_derivatives(self):
        """Check construct_raw with fallout derivatives."""
        desc = self.lambda_nu_desc()
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(np.ones((self.nb,)),
                                 dsd_deriv=np.ones((2, self.nb)),
                                 fallout_deriv=fallout_deriv)
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        scaled_fallout_deriv = fallout_deriv / self.constants.mass_conc_scale
        self.assertArrayEqual(actual_fallout_deriv, scaled_fallout_deriv)

    def test_construct_raw_missing_fallout_derivative_is_zero(self):
        """Check construct_raw fallout derivative defaults to zero."""
        desc = self.lambda_nu_desc()
        raw = desc.construct_raw(np.ones((self.nb,)),
                                 dsd_deriv=np.ones((2, self.nb)))
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        self.assertArrayEqual(actual_fallout_deriv, np.zeros((2,)))

    def test_construct_raw_with_fallout_deriv_raises_for_wrong_length(self):
        """Check construct_raw raises error if fallout_deriv is wrong size."""
        desc = self.lambda_nu_desc()
        with self.assertRaises(ValueError):
            desc.construct_raw(np.ones((self.nb,)),
                               dsd_deriv=np.ones((2, self.nb)),
                               fallout_deriv=np.ones((3,)))

    def test_construct_raw_with_extra_fallout_deriv_raises(self):
        """Check construct_raw raises for unexpected fallout derivatives."""
        desc = self.dsd_only_desc()
        with self.assertRaises(RuntimeError):
            raw = desc.construct_raw(np.ones(self.nb),
                                     fallout_deriv=np.ones((1,)))

    def test_construct_raw_allows_empty_fallout(self):
        """Check construct_raw allows unexpected empty fallout derivatives."""
        desc = self.dsd_only_desc()
        desc.construct_raw(np.ones(self.nb), fallout_deriv=np.ones((0,)))

    def test_construct_raw_with_fallout_derivatives_scaling(self):
        """Check construct_raw with scaled fallout derivatives."""
        desc = self.lambda_nu_scaled_desc()
        fallout_deriv = np.array([700., 800.])
        raw = desc.construct_raw(np.ones((self.nb,)),
                                 dsd_deriv=np.ones((2, self.nb)),
                                 fallout_deriv=fallout_deriv)
        actual_fallout_deriv = desc.fallout_deriv_raw(raw)
        scaled_fallout_deriv = np.array(
            [fallout_deriv[i] * self.deriv_var_scales[i]
             / self.constants.mass_conc_scale
             for i in range(2)]
        )
        self.assertArrayEqual(actual_fallout_deriv, scaled_fallout_deriv)

    def test_perturbation_covariance(self):
        """Check state vector size when perturbed variables are provided."""
        desc = self.perturb_069_desc()
        self.assertEqual(desc.perturb_num, self.pn)
        nchol = (self.pn * (self.pn + 1)) // 2
        self.assertEqual(desc.state_len(), 3*self.nb + 3 + nchol)
        self.assertEqual(desc.perturbation_rate.shape, (self.pn, self.pn))
        for i in range(self.pn):
            for j in range(self.pn):
                self.assertAlmostEqual(desc.perturbation_rate[i,j],
                                         self.perturbation_rate[i,j] \
                                             / self.perturb_scales[i] \
                                             / self.perturb_scales[j] \
                                             * self.constants.time_scale)

    def test_perturbation_covariance_raises_for_mismatched_sizes(self):
        """Check error for mismatch in perturbed variables and rate size."""
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.)]
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        scale = 10. / np.log(10.)
        perturbed_vars = [
            PerturbedVar('L0', wv0, LogTransform(), scale),
            PerturbedVar('L6', wv6, LogTransform(), 2.*scale),
        ]
        perturbation_rate = np.eye(3)
        with self.assertRaises(ValueError):
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
                                    perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_correction_time(self):
        """Check scaling on correction_time."""
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
        deriv_vars = [DerivativeVar('lambda', 3.)]
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
        correction_time = 5.
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                    perturbed_vars=perturbed_vars,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time)
        self.assertAlmostEqual(desc.correction_time,
                               correction_time / const.time_scale)

    def test_perturb_cov_requires_correction_time_when_dims_mismatch(self):
        """Check correction time required when deriv_var_num != perturb_num."""
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
        deriv_vars = [DerivativeVar('lambda', 3.)]
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
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                        perturbed_vars=perturbed_vars,
                                        perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_correction_time_without_pv_raises(self):
        """Check correction time only allowed when perturb_num > 0."""
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.)]
        correction_time = 5.
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(self.constants, self.grid,
                                        deriv_vars=deriv_vars,
                                        correction_time=correction_time)

    def test_perturb_chol_loc(self):
        """Check perturb_chol_loc."""
        desc = self.perturb_069_desc()
        self.assertEqual(desc.perturb_chol_loc(),
                         (3*self.nb+3, (self.pn*(self.pn+1)) // 2))

    def test_perturb_chol_raw(self):
        """Check perturb_chol_raw."""
        desc = self.perturb_069_desc()
        raw = np.zeros((desc.state_len(),))
        num_chol = (self.pn * (self.pn + 1)) // 2
        raw[-num_chol:] = np.linspace(1, num_chol, num_chol)
        actual = desc.perturb_chol_raw(raw)
        expected = np.zeros((self.pn, self.pn))
        ic = 0
        for i in range(self.pn):
            for j in range(i+1):
                expected[i,j] = raw[-num_chol+ic]
                ic += 1
        self.assertArrayEqual(actual, expected)

    def test_perturb_cov_raw(self):
        """Check perturb_cov_raw."""
        desc = self.perturb_069_desc()
        raw = np.zeros((desc.state_len(),))
        num_chol = (self.pn * (self.pn + 1)) // 2
        raw[-num_chol:] = np.linspace(1, num_chol, num_chol)
        actual = desc.perturb_cov_raw(raw)
        expected = np.zeros((self.pn, self.pn))
        ic = 0
        for i in range(self.pn):
            for j in range(i+1):
                expected[i,j] = raw[-num_chol+ic]
                ic += 1
        expected = expected @ expected.T
        self.assertArrayEqual(actual, expected)

    def test_perturb_cov_construct_raw(self):
        """Check construct_raw with perturb_cov supplied."""
        desc = self.perturb_069_desc()
        orig = np.reshape(np.linspace(1, self.pn*self.pn, self.pn*self.pn),
                          (self.pn, self.pn))
        orig = orig + orig.T + 20. * np.eye(self.pn)
        raw = desc.construct_raw(np.ones((self.nb,)),
                                 dsd_deriv=np.ones((2, self.nb)),
                                 perturb_cov=orig)
        self.assertEqual(len(raw), desc.state_len())
        actual = desc.perturb_cov_raw(raw)
        expected = orig
        for i in range(self.pn):
            for j in range(self.pn):
                expected[i,j] /= self.perturb_scales[i] * self.perturb_scales[j]
        self.assertArrayAlmostEqual(actual, expected)

    def test_perturb_cov_construct_raw_raises_if_perturb_unexpected(self):
        """Check that construct_raw raises for unexpected perturb_cov."""
        desc = self.dsd_only_desc()
        with self.assertRaises(RuntimeError):
            raw = desc.construct_raw(np.ones((self.nb)),
                                     perturb_cov=np.eye(3))

    def test_perturb_cov_construct_raw_raises_if_perturb_is_wrong_size(self):
        """Check that construct_raw raises for wrong-sized perturb_cov."""
        desc = self.perturb_069_desc()
        orig = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            raw = desc.construct_raw(np.ones((self.nb,)),
                                     dsd_deriv=np.ones((2, self.nb)),
                                     perturb_cov=np.eye(2))

    def test_perturb_cov_construct_raw_defaults_to_near_zero_cov(self):
        """Check that construct_raw starts with near-zero perturb_cov."""
        desc = self.perturb_069_desc()
        raw = desc.construct_raw(np.ones((self.nb)),
                                 dsd_deriv=np.ones((2, self.nb)))
        actual = desc.perturb_cov_raw(raw)
        variance = ModelStateDescriptor.small_error_variance
        expected = variance * np.eye(self.pn)
        places = -int(np.log10(variance)) + 10
        self.assertArrayAlmostEqual(actual, expected, places=places)
