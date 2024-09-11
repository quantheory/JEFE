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

"""Test transform module."""

import unittest

# pylint: disable-next=wildcard-import,unused-wildcard-import
from bin_model.transform import *


class TestIdentityTransform(unittest.TestCase):
    """
    Test IdentityTransform methods.
    """
    def test_identity_transform(self):
        """Check transform."""
        self.assertEqual(IdentityTransform().transform(2.), 2.)

    def test_identity_transform_deriv(self):
        """Check transform derivative."""
        self.assertEqual(IdentityTransform().derivative(2.), 1.)

    def test_identity_transform_second_over_first_derivative(self):
        """Check transform second_over_first_derivative."""
        self.assertEqual(IdentityTransform().second_over_first_derivative(2.),
                         0.)

    def test_type_string(self):
        """Check type_string."""
        self.assertEqual(IdentityTransform().type_string(), 'Identity')

    def test_get_parameters(self):
        """Check get_parameters."""
        self.assertEqual(IdentityTransform().get_parameters(), [])


class TestLogTransform(unittest.TestCase):
    """
    Test LogTransform methods.
    """
    def test_log_transform(self):
        """Check transform."""
        self.assertEqual(LogTransform().transform(2.), np.log(2.))

    def test_log_transform_deriv(self):
        """Check transform derivative."""
        self.assertEqual(LogTransform().derivative(2.), 1./2.)

    def test_log_transform_second_over_first_derivative(self):
        """Check transform second_over_first_derivative."""
        self.assertEqual(LogTransform().second_over_first_derivative(2.),
                         -1./2.)

    def test_type_string(self):
        """Check type_string."""
        self.assertEqual(LogTransform().type_string(), 'Log')

    def test_get_parameters(self):
        """Check get_parameters."""
        self.assertEqual(LogTransform().get_parameters(), [])


class TestQuadToLogTransform(unittest.TestCase):
    """
    Test QuadToLogTransform methods.
    """
    def setUp(self):
        self.trans = QuadToLogTransform(0.3)

    def test_quad_to_log_transform(self):
        """Check transform."""
        self.assertEqual(self.trans.transform(0.15), 0.875)
        self.assertEqual(self.trans.transform(2.),
                         np.log(2. / 0.3) + 1.5)

    def test_identity_transform_deriv(self):
        """Check transform derivative."""
        self.assertEqual(self.trans.derivative(0.15), 5.)
        self.assertEqual(self.trans.derivative(2.), 1./2.)

    def test_identity_transform_second_over_first_derivative(self):
        """Check transform second_over_first_derivative."""
        self.assertEqual(self.trans.second_over_first_derivative(0.15),
                         -20./9.)
        self.assertEqual(self.trans.second_over_first_derivative(2.),
                         -1./2.)

    def test_type_string(self):
        """Check type_string."""
        self.assertEqual(self.trans.type_string(), 'QuadToLog')

    def test_get_parameters(self):
        """Check get_parameters."""
        self.assertEqual(self.trans.get_parameters(), [0.3])
