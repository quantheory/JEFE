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

"""Test ArrayTestCase class."""

import unittest

import numpy as np

from .array_assert import ArrayTestCase

class TestArrayTestCase(ArrayTestCase):
    """
    Test ArrayTestCase assertions.
    """

    def test_array_equal(self):
        """Check assertArrayEqual succeeds for equal arrays."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.])
        self.assertArrayEqual(a, b)

    def test_array_equal_2d(self):
        """Check assertArrayEqual succeeds for equal 2D arrays."""
        a = np.array([[0., 1., 2.], [3., 4., 5.]])
        b = np.array([[0., 1., 2.], [3., 4., 5.]])
        self.assertArrayEqual(a, b)

    @unittest.expectedFailure
    def test_array_equal_first_is_not_an_array(self):
        """Check assertArrayEqual fails if first argument is not an array."""
        a = 1.
        b = np.array([0., 1., 2.])
        self.assertArrayEqual(a, b)

    @unittest.expectedFailure
    def test_array_equal_second_is_not_an_array(self):
        """Check assertArrayEqual fails if second argument is not an array."""
        a = np.array([0., 1., 2.])
        b = 1.
        self.assertArrayEqual(a, b)

    @unittest.expectedFailure
    def test_array_equal_first_long(self):
        """Check assertArrayEqual fails if the first array is longer."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1.])
        self.assertArrayEqual(a, b)

    @unittest.expectedFailure
    def test_array_equal_second_long(self):
        """Check assertArrayEqual fails if the second array is longer."""
        a = np.array([0., 1.])
        b = np.array([0., 1., 2.])
        self.assertArrayEqual(a, b)

    @unittest.expectedFailure
    def test_array_equal_fails(self):
        """Check assertArrayEqual fails if arrays are unequal."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.]) + 1.e-12
        self.assertArrayEqual(a, b)

    def test_array_almost_equal_passes_equal(self):
        """Check assertArrayAlmostEqual succeeds for equal arrays."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.])
        self.assertArrayAlmostEqual(a, b)

    def test_array_almost_equal_2d(self):
        """Check assertArrayAlmostEqual succeeds for equal 2D arrays."""
        a = np.array([[0., 1., 2.], [3., 4., 5.]])
        b = np.array([[0., 1., 2.], [3., 4., 5.]])
        self.assertArrayAlmostEqual(a, b)

    def test_array_almost_equal_passes_slightly_unequal(self):
        """Check assertArrayAlmostEqual succeeds for slightly equal arrays."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.]) + 1.e-12
        self.assertArrayAlmostEqual(a, b)

    @unittest.expectedFailure
    def test_array_almost_equal_first_is_not_an_array(self):
        """Check assertAlmostArrayEqual fails if first is not an array."""
        a = 0.
        b = np.array([0., 1., 2.])
        self.assertArrayAlmostEqual(a, b)

    @unittest.expectedFailure
    def test_array_almost_equal_second_is_not_an_array(self):
        """Check assertArrayAlmostEqual fails if second is not an array."""
        a = np.array([0., 1., 2.])
        b = 0.
        self.assertArrayAlmostEqual(a, b)

    @unittest.expectedFailure
    def test_array_almost_equal_first_long(self):
        """Check assertArrayAlmostEqual fails if the first array is longer."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1.])
        self.assertArrayAlmostEqual(a, b)

    @unittest.expectedFailure
    def test_array_almost_equal_second_long(self):
        """Check assertArrayAlmostEqual fails if the second array is longer."""
        a = np.array([0., 1.])
        b = np.array([0., 1., 2.])
        self.assertArrayAlmostEqual(a, b)

    @unittest.expectedFailure
    def test_array_almost_equal(self):
        """Check assertArrayAlmostEqual fails if arrays are unequal."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.]) + 1.e-6
        self.assertArrayAlmostEqual(a, b)

    def test_array_almost_equal_places(self):
        """Check assertArrayAlmostEqual is more permissive for fewer places."""
        a = np.array([0., 1., 2.])
        b = np.array([0., 1., 2.]) + 1.e-6
        self.assertArrayAlmostEqual(a, b, places=4)
