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

"""Test perturbation module."""

import numpy as np

from bin_model import GeometricMassGrid, LogTransform, ModelConstants, \
    PerturbedVar
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.perturbation import *
from .array_assert import ArrayTestCase


class TestStochasticPerturbation(ArrayTestCase):
    """
    Test StochasticPerturbation class.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4,
                                        mass_conc_scale=1.e-3,
                                        time_scale=400.)
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=1.e-5,
                                      num_bins=5)
        scale = 10. / np.log(10.)
        self.perturbed_scales = [scale, 2.*scale, 3.*scale]
        l0 = PerturbedVar("L0", self.grid.moment_weight_vector(0),
                          LogTransform(), self.perturbed_scales[0])
        l6 = PerturbedVar("L6", self.grid.moment_weight_vector(6),
                          LogTransform(), self.perturbed_scales[1])
        l9 = PerturbedVar("L9", self.grid.moment_weight_vector(9),
                          LogTransform(), self.perturbed_scales[2])
        self.perturbed_vars = [l0, l6, l9]
        self.nvar = 3
        error_rate = 0.5 / 60.
        self.rate = error_rate**2 * np.eye(self.nvar)
        self.perturb = StochasticPerturbation(self.constants,
                                              self.perturbed_vars,
                                              self.rate)

    def test_perturbed_num_is_correct(self):
        """Check that perturbed_num matches rate matrix dimensions."""
        self.assertEqual(self.perturb.perturbed_num, self.nvar)

    def test_perturbation_rate_is_correct(self):
        """Check that input rate is scaled correctly."""
        expected_rate = self.rate * self.constants.time_scale
        for i in range(self.nvar):
            for j in range(self.nvar):
                expected_rate[i,j] /= self.perturbed_scales[i]
                expected_rate[i,j] /= self.perturbed_scales[j]
        self.assertArrayAlmostEqual(self.perturb.perturbation_rate,
                                    expected_rate)

    def test_correction_time_is_correct(self):
        """Check that correction time is scaled correctly."""
        ctime = 10.
        perturb = StochasticPerturbation(self.constants,
                                         self.perturbed_vars,
                                         self.rate,
                                         ctime)
        self.assertAlmostEqual(perturb.correction_time,
                               ctime / self.constants.time_scale)

    def test_perturbation_rate_dimensions(self):
        """Check that perturbation rate is a square matrix of correct size."""
        bad_dims_list = [
            (self.nvar, self.nvar, self.nvar),
            (self.nvar),
            (self.nvar, self.nvar+1),
            (self.nvar+1, self.nvar+1),
        ]
        for bad_dims in bad_dims_list:
            msg = f"should have rejected {bad_dims}"
            with self.assertRaises(ValueError, msg=msg):
                StochasticPerturbation(self.constants,
                                       self.perturbed_vars,
                                       np.zeros(bad_dims))

    def test_perturbation_rate_must_be_hermitian(self):
        """Check that perturbation rate matrix is Hermitian."""
        rate = np.zeros((3,3))
        rate[0,1] = 1.
        with self.assertRaises(ValueError):
            StochasticPerturbation(self.constants,
                                   self.perturbed_vars,
                                   rate)
