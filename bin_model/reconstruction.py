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

"""Classes for reconstructing DSDs in terms of basis functions."""

from abc import ABC, abstractmethod

import numba as nb
import numpy as np

from bin_model.basis import make_piecewise_polynomial_basis

class Reconstruction(ABC):
    """
    Base class for all reconstruction methods in the bin model.
    """

    @abstractmethod
    def output_basis(self):
        """Get the basis functions that will be used for reconstruction output.

        Returns the Basis corresponding to the output subspace.
        """

    @abstractmethod
    def reconstruct(self, vars):
        """Reconstruct the basis function view from prognostic variables.

        The returned variable is an array of weights corresponding to the input
        basis functions.
        """

class ConstantReconstruction(Reconstruction):
    """
    Reconstruct the mass-weighted DSD as piecewise-constant over log(mass).

    Initialization arguments:
    grid - The MassGrid defining the prognostic mass variables.
    """
    def __init__(self, grid):
        self.grid = grid

    def output_basis(self):
        return make_piecewise_polynomial_basis(self.grid, 0)

    def reconstruct(self, vars):
        return vars / self.grid.bin_widths
