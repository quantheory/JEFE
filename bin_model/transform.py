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

"""Moment transforms used for JEFE."""

from abc import ABC, abstractmethod

import numpy as np


class Transform(ABC):
    """
    Represent a transformation of a prognostic variable.
    """

    transform_type_str_len = 32
    "Length of a transform type string on file."

    @abstractmethod
    def transform(self, x):
        """Transform the variable."""

    @abstractmethod
    def derivative(self, x):
        """Calculate the first derivative of the transformation."""

    @abstractmethod
    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""

    @abstractmethod
    def type_string(self):
        """Get string representing type of transform."""

    @abstractmethod
    def get_parameters(self):
        """Get parameters of this transform as a list."""

    @classmethod
    def from_params(cls, type_str, params):
        """Construct a Transform from type string and parameters."""
        if type_str == "Identity":
            return IdentityTransform()
        if type_str == "Log":
            return LogTransform()
        if type_str == "QuadToLog":
            return QuadToLogTransform(params[0])
        raise ValueError("transform type string not recognized")


class IdentityTransform(Transform):
    """
    Transform a prognostic variable by doing nothing.
    """
    def transform(self, x):
        """Transform the variable."""
        return x

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        return 1.

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        return 0.

    def type_string(self):
        """Get string representing type of transform."""
        return "Identity"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return []


class LogTransform(Transform):
    """
    Transform a prognostic variable using the natural logarithm.
    """
    def transform(self, x):
        """Transform the variable."""
        return np.log(x)

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        return 1./x

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        return -1./x

    def type_string(self):
        """Get string representing type of transform."""
        return "Log"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return []


class QuadToLogTransform(Transform):
    """
    Transform a prognostic variable using a mix of a quadratic and logarithm.

    The transform represented by this class uses a quadratic near 0, and the
    natural logarithm for larger values. The logarithm is offset, and quadratic
    chosen so that the first and second derivatives are continuous.

    Initialization arguments:
    x0 - Length scale at which quadratic to logarithm transition occurs.
    """
    def __init__(self, x0):
        self.x0 = x0

    def transform(self, x):
        """Transform the variable."""
        xs = x / self.x0
        if xs >= 1.:
            return np.log(xs) + 1.5
        return -0.5 * (xs)**2 + 2. * xs

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        xs = x / self.x0
        if xs >= 1.:
            return 1. / x
        return (-xs + 2.) / self.x0

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        xs = x / self.x0
        if xs >= 1.:
            return -1. / x
        return -1. / ((-xs + 2.) * self.x0)

    def type_string(self):
        """Get string representing type of transform."""
        return "QuadToLog"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return [self.x0]
