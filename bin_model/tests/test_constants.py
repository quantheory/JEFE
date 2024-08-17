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

"""Tests for constants module."""

import unittest

# pylint: disable-next=wildcard-import
from bin_model.constants import *

class TestModelConstants(unittest.TestCase):
    """
    Test conversion methods on ModelConstants objects.
    """

    def setUp(self):
        """Set up a ModelConstants object to check."""
        self.constants = ModelConstants(rho_water=1000.,
                                        rho_air=1.2,
                                        diameter_scale=1.e-4,
                                        rain_d=1.e-4)

    def test_diameter_to_scaled_mass(self):
        """Check conversion of a diameter to scaled mass variable."""
        # Diameter of 10 microns with 100 micron diameter_scale means that the
        # scaled diameter is 1.e-1, so 1.e-3 for scaled mass.
        self.assertAlmostEqual(self.constants.diameter_to_scaled_mass(1.e-5),
                               1.e-3)

    def test_scaled_mass_to_diameter(self):
        """Check conversion of a scaled mass variable to diameter."""
        # Reverse check of diameter_to_scaled_mass.
        self.assertAlmostEqual(self.constants.scaled_mass_to_diameter(1.e-3),
                               1.e-5)

    def test_std_mass(self):
        """Check conversion of diameter_scale to std_mass."""
        self.assertAlmostEqual(self.constants.std_mass,
                               self.constants.rho_water * np.pi/6. * 1.e-12)

    def test_rain_m(self):
        """Check conversion of rain_d to rain_m."""
        self.assertEqual(self.constants.rain_m, 1.)

    def test_mass_conc_scale(self):
        """Check that the default mass_conc_scale is 1."""
        self.assertEqual(self.constants.mass_conc_scale, 1.)

    def test_time_scale(self):
        """Check that the default time_scale is 1."""
        self.assertEqual(self.constants.time_scale, 1.)
