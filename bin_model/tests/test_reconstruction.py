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

"""Test Reconstruction classes."""

from bin_model import ModelConstants, GeometricMassGrid, \
    make_piecewise_polynomial_basis
# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.reconstruction import *

from .array_assert import ArrayTestCase

class TestConstantReconstruction(ArrayTestCase):
    """
    Test ConstantReconstruction methods.
    """
    def setUp(self):
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)
        self.num_bins = 6
        self.grid = GeometricMassGrid(self.constants,
                                      d_min=1.e-6,
                                      d_max=2.e-6,
                                      num_bins=self.num_bins)

    def test_output_basis(self):
        """Test that the reconstruction produces the expected output basis."""
        recon = ConstantReconstruction(self.grid)
        actual = recon.output_basis()
        expected = make_piecewise_polynomial_basis(self.grid, 0)
        for i, bf in enumerate(actual):
            self.assertEqual(bf, expected[i])

    def test_reconstruct(self):
        """Check the reconstruction's coefficients."""
        dsd = np.arange(self.num_bins)
        recon = ConstantReconstruction(self.grid)
        actual = recon.reconstruct(dsd)
        self.assertArrayAlmostEqual(actual, dsd / self.grid.bin_widths)
