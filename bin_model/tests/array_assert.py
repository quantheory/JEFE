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

"""Subclass unittest.TestCase to add assertions for numpy arrays."""

import unittest

import numpy as np

# Suppress traceback through this module.
# pylint: disable-next=invalid-name
__unittest = True

class ArrayTestCase(unittest.TestCase):
    """
    Provide assertArrayEqual and assertArrayAlmostEqual assertions.
    """

    def _assert_array_shapes_equal(self, a, b):
        """Assert that arguments are both arrays and are equal."""
        self.assertIsInstance(a, np.ndarray,
                              msg="first argument is not a numpy array")
        self.assertIsInstance(b, np.ndarray,
                              msg="second argument is not a numpy array")
        if a.shape != b.shape:
            self.fail(msg=f"Mismatched array shapes: {a.shape} != {b.shape}")

    # pylint: disable-next=invalid-name
    def assertArrayEqual(self, a, b):
        """Assert that inputs are two numpy arrays that are equal."""
        self._assert_array_shapes_equal(a, b)
        for i, a_i, b_i in zip(range(len(a.flat)), a.flat, b.flat):
            self.assertEqual(a_i, b_i,
                             msg=f"first failure in entry {i}")

    # pylint: disable-next=invalid-name
    def assertArrayAlmostEqual(self, a, b, places=7):
        """Assert that inputs are two numpy arrays that are almost equal.

        The `places` parameter is passed directly to unittest.assertAlmostEqual.
        """
        self._assert_array_shapes_equal(a, b)
        for i, a_i, b_i in zip(range(len(a.flat)), a.flat, b.flat):
            self.assertAlmostEqual(a_i, b_i, places=places,
                                   msg=f"first failure in entry {i}")
