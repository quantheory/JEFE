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

"""Classes for processes in JEFE's bin model."""

from abc import ABC, abstractmethod

import numba as nb
import numpy as np

class Process(ABC):
    """
    Base class for all processes in the bin model.
    """

    @abstractmethod
    def calc_rate(self, f, out_flux=None, derivative=False):
        """Calculate rate of change of prognostic variables due to this process.

        Arguments:
        f - Representation of DSD on this grid.
        out_flux (optional) - Whether to force output of mass leaving the box.
        derivative (optional) - Whether to return the Jacobian of the rate
                                calculation as well.

        ``f'' must be an array of total size num_bins or num_bins+1, either a
        1-D array, a row vector, or column vector. If of size num_bins+1, the
        last bin is assumed to hold the mass that has been removed from the
        grid cell, which is ignored.

        If out_flux is not specified, the output is an array with the same
        shape as f, containing the rate of change of f over time due to
        this process. If f is of size num_bins+1, the final element of
        the output is the amount of mass that leaves the box (e.g. due to
        sedimentation or advection).

        If out_flux is specified and True, then the output is the same as if f
        had been of shape (nb+1,), and if it is False, then the output is the
        same as if f was of shape (nb,).

        If derivative is True, the return value is a tuple, where the first
        element is the output described above, and the second element is a
        square matrix with the same size (on each size) as the first output.
        This matrix contains the Jacobian of this output with respect to the
        DSD (+ fallout if included in the output).
        """

class CollisionCoalescence(Process):
    """
    Represents the process of collision-coalescence.

    Initialization arguments:
    recon - A Reconstruction argument for transforming to bin space.
    ctens - A CollisionTensor object that uses the reconstruction to predict the
            process rate.
    """

    def __init__(self, recon, ctens):
        self.recon = recon
        self.grid = recon.grid
        self.ctens = ctens

    def calc_rate(self, f, out_flux=None, derivative=False):
        """Calculate process rate for collision-coalescence.

        Arguments:
        f - Representation of DSD on this grid.
        out_flux (optional) - Whether to force output of mass leaving the box.
        derivative (optional) - Whether to return the Jacobian of the rate
                                calculation as well.

        ``f'' must be an array of total size num_bins or num_bins+1, either a
        1-D array, a row vector, or column vector. If of size num_bins+1, the
        last bin is assumed to hold the mass that has been removed from the
        grid cell, which is ignored.

        If out_flux is not specified, the output is an array with the same
        shape as f, containing the rate of change of f over time due to
        coalescence. If f is of size num_bins+1, the final element of
        the output is the amount of mass that leaves the box (e.g. due to
        assuming large particles fall out of the box).

        If out_flux is specified and True, then the output is the same as if f
        had been of shape (nb+1,), and if it is False, then the output is the
        same as if f was of shape (nb,).

        If derivative is True, the return value is a tuple, where the first
        element is the output described above, and the second element is a
        square matrix with the same size (on each size) as the first output.
        This matrix contains the Jacobian of this output with respect to the
        DSD (+ fallout if included in the output).
        """
        nb = self.grid.num_bins
        f_len = np.product(f.shape)
        if not nb <= f_len < nb+2:
            raise ValueError("invalid f length: "+str(f_len))
        if out_flux is None:
            out_flux = f_len == nb + 1
            out_len = f_len
            out_shape = f.shape
        else:
            out_len = nb + 1 if out_flux else nb
            out_shape = (out_len,)
        f_shaped = np.reshape(f.copy(), (f_len, 1))
        f_shaped[:nb,0] = self.recon.reconstruct(f_shaped[:nb,0])
        rate = self.ctens._calc_rate(f_shaped, out_flux, out_len)
        output = np.reshape(rate, out_shape)
        if derivative:
            deriv = self.ctens._calc_deriv(f_shaped, out_flux, out_len)
            for i in range(nb):
                deriv[:,i] /= self.grid.bin_widths[i]
            return output, deriv
        return output
