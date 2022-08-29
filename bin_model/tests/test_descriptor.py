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
            DerivativeVar("a" * 1024, 1.)

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


class TestModelStateDescriptor(ArrayTestCase):
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
        self.assertEqual(desc.deriv_var_num, 0)

    def test_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        self.assertEqual(desc.deriv_var_num, len(deriv_vars))
        self.assertEqual(desc.state_len(), 3*nb+3)

    def test_find_deriv_var_index(self):
        """Check output of find_deriv_var_index."""
        const = self.constants
        grid = self.grid
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        self.assertEqual(desc.find_deriv_var_index('lambda'), 0)
        self.assertEqual(desc.find_deriv_var_index('nu'), 1)

    def test_find_deriv_var_index_raises(self):
        """Check that find_deriv_var_index raises an error for invalid name."""
        const = self.constants
        grid = self.grid
        deriv_vars = [DerivativeVar('lambda')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        with self.assertRaises(ValueError):
            desc.find_deriv_var_index('nonsense')

    def test_find_deriv_var(self):
        """Check output of find_deriv_var."""
        const = self.constants
        grid = self.grid
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        self.assertEqual(desc.find_deriv_var('lambda').name, 'lambda')
        self.assertEqual(desc.find_deriv_var('nu').name, 'nu')

    def test_find_deriv_var_raises(self):
        """Check that find_deriv_var raises an error for invalid name."""
        const = self.constants
        grid = self.grid
        deriv_vars = [DerivativeVar('lambda')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        with self.assertRaises(ValueError):
            desc.find_deriv_var('nonsense')

    def test_empty_derivatives(self):
        const = self.constants
        grid = self.grid
        desc = ModelStateDescriptor(const, grid)
        desc = ModelStateDescriptor(const, grid, deriv_vars=[])
        self.assertEqual(desc.deriv_var_num, 0)

    def test_derivatives_raises_on_duplicate_name(self):
        """Check ValueError raised for duplicate derivative names."""
        const = self.constants
        grid = self.grid
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('lambda', 4.)]
        with self.assertRaises(ValueError):
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)

    def test_dsd_deriv_loc_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        idxs, num = desc.dsd_deriv_loc()
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_all_with_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        idxs, num = desc.dsd_deriv_loc(with_fallout=True)
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb+1)

    def test_dsd_deriv_loc_all_without_fallout(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        idxs, num = desc.dsd_deriv_loc(with_fallout=False)
        self.assertEqual(idxs, [nb+1, 2*nb+2])
        self.assertEqual(num, nb)

    def test_dsd_deriv_loc_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        with self.assertRaises(ValueError):
            desc.dsd_deriv_loc('nonsense')

    def test_fallout_deriv_loc_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        idxs = desc.fallout_deriv_loc()
        self.assertEqual(idxs, [2*nb+1, 3*nb+2])

    def test_fallout_deriv_loc_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        idx = desc.fallout_deriv_loc('lambda')
        self.assertEqual(idx, 2*nb+1)
        idx = desc.fallout_deriv_loc('nu')
        self.assertEqual(idx, 3*nb+2)

    def test_fallout_deriv_loc_raises_for_bad_string(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        with self.assertRaises(ValueError):
            idxs = desc.fallout_deriv_loc('nonsense')

    def test_dsd_deriv_raw_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        with self.assertRaises(ValueError):
            desc.dsd_deriv_raw(raw, 'nonsense')

    def test_fallout_deriv_raw_all(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        fallout_derivs = desc.fallout_deriv_raw(raw)
        self.assertEqual(len(fallout_derivs), 2)
        for i in range(2):
            self.assertEqual(fallout_derivs[i], raw[nb+(i+1)*(nb+1)])

    def test_fallout_deriv_raw_individual(self):
        const = self.constants
        grid = self.grid
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
        raw = np.linspace(0, 3*nb, 3*nb+3)
        fallout_deriv = desc.fallout_deriv_raw(raw, 'lambda')
        self.assertEqual(fallout_deriv, raw[2*nb+1])
        fallout_deriv = desc.fallout_deriv_raw(raw, 'nu')
        self.assertEqual(fallout_deriv, raw[3*nb+2])

    def test_construct_raw_with_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
                                       dsd_deriv[i,j] * deriv_vars[i].scale
                                          / dsd_scale)

    def test_construct_raw_with_fallout_derivatives(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        deriv_vars = [DerivativeVar('lambda'), DerivativeVar('nu')]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
        desc = ModelStateDescriptor(const, grid)
        dsd = np.linspace(0, nb, nb)
        fallout = 200.
        fallout_deriv = np.zeros((0,))
        raw = desc.construct_raw(dsd, fallout=fallout,
                                 fallout_deriv=fallout_deriv)

    def test_construct_raw_with_fallout_derivatives_scaling(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars)
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
                                   fallout_deriv[i] * deriv_vars[i].scale
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.)]
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
        ]
        perturbation_rate = np.eye(3)
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_raises_for_rate_without_variable(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.)]
        dsd_deriv_names = ['lambda']
        dsd_deriv_scales = np.array([3.])
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        perturbation_rate = np.eye(3)
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                    perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_allows_variables_without_rate(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.)]
        wv0 = grid.moment_weight_vector(0)
        wv6 = grid.moment_weight_vector(6)
        scale = 10. / np.log(10.)
        perturbed_variables = [
            (wv0, LogTransform(), scale),
            (wv6, LogTransform(), 2.*scale),
        ]
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.)]
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
            desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                        perturbed_variables=perturbed_variables,
                                        perturbation_rate=perturbation_rate)

    def test_perturbation_covariance_correction_time_without_pv_raises(self):
        nb = self.grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.)]
        correction_time = 5.
        with self.assertRaises(AssertionError):
            desc = ModelStateDescriptor(self.constants, self.grid,
                                        deriv_vars=deriv_vars,
                                        correction_time=correction_time)

    def test_perturb_chol_loc(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate)
        self.assertEqual(desc.perturb_chol_loc(),
                         (3*nb+3, (nvar*(nvar+1)) // 2))

    def test_perturb_chol_raw(self):
        const = self.constants
        grid = self.grid
        nb = grid.num_bins
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
        deriv_vars = [DerivativeVar('lambda', 3.), DerivativeVar('nu', 4.)]
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
        desc = ModelStateDescriptor(const, grid, deriv_vars=deriv_vars,
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
