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

import numpy as np

class KernelTensor():
    """
    Represent a collision kernel evaluated on a particular mass grid.

    Initialization arguments:
    kernel - A Kernel object representing the collision kernel.
    grid - A MassGrid object defining the bins.
    boundary (optional) - Upper boundary condition. If 'open', then particles
                          that are created larger than the largest bin size
                          "fall out" of the box. If 'closed', these particles
                          are placed in the largest bin. Defaults to 'open'.
    data (optional) - Precalculated kernel tensor data.

    Attributes:
    grid - Stored reference to corresponding grid.
    scaling - Effect of the kernel has been scaled down by this amount.
    boundary - Upper boundary condition for this kernel.
    idxs, nums, max_num - Outputs of `MassGrid.construct_sparsity_structure`.
    data - Data corresponding to nonzero elements of the tensor kernel.
           This is represented by an array of shape:
               (num_bins, num_bins, max_num)

    Methods:
    calc_rate
    to_netcdf

    Class Methods:
    from_netcdf
    """

    boundary_str_len = 16
    """Length of string specifying boundary condition for largest bin."""

    def __init__(self, kernel, grid, boundary=None, data=None):
        self.kernel = kernel
        self.grid = grid
        self.const = grid.constants
        self.scaling = self.const.std_mass \
            / (self.const.mass_conc_scale * self.const.time_scale)
        if boundary is None:
            boundary = 'open'
        self.boundary = boundary
        idxs, nums, max_num = \
            grid.construct_sparsity_structure(boundary=boundary)
        self.idxs = idxs
        self.nums = nums
        self.max_num = max_num
        nb = grid.num_bins
        bb = grid.bin_bounds
        if data is not None:
            self.data = data
            return
        self.data = np.zeros((nb, nb, max_num))
        if boundary == 'closed':
            high_bin = nb - 1
        else:
            high_bin = nb
        for i in range(nb):
            for j in range(nb):
                idx = idxs[i,j]
                for k in range(nums[i,j]):
                    zidx = idx + k
                    if zidx == high_bin:
                        top_bound = np.inf
                    else:
                        top_bound = bb[zidx+1]
                    self.data[i,j,k] = kernel.integrate_over_bins(
                        (bb[i], bb[i+1]), (bb[j], bb[j+1]),
                        (bb[zidx], top_bound))
        self.data /= self.scaling

    def calc_rate(self, f, out_flux=None, derivative=False):
        """Calculate rate of change of f due to collision-coalescence.

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
        collision-coalescence. If f is of size num_bins+1, the final element of
        the output is the amount of mass that leaves the box due to collision-
        coalescence (i.e. due to becoming too large).

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
        if not (nb <= f_len < nb+2):
            raise ValueError("invalid f length: "+str(f_len))
        if out_flux is None:
            out_flux = f_len == nb + 1
            out_len = f_len
            out_shape = f.shape
        else:
            out_len = nb + 1 if out_flux else nb
            out_shape = (out_len,)
        f = np.reshape(f, (f_len, 1))
        f_outer = np.dot(f, np.transpose(f))
        output = np.zeros((out_len,))
        if derivative:
            rate_deriv = np.zeros((out_len, out_len))
        for i in range(nb):
            for j in range(nb):
                idx = self.idxs[i,j]
                fprod = f_outer[i,j]
                for k in range(self.nums[i,j]):
                    dfdt_term = fprod * self.data[i,j,k]
                    output[i] -= dfdt_term
                    if out_flux or idx+k < nb:
                        output[idx+k] += dfdt_term
                    if derivative:
                        deriv_i = self.data[i,j,k] * f[j]
                        deriv_j = self.data[i,j,k] * f[i]
                        rate_deriv[i,i] -= deriv_i
                        rate_deriv[i,j] -= deriv_j
                        if out_flux or idx+k < nb:
                            rate_deriv[idx+k,i] += deriv_i
                            rate_deriv[idx+k,j] += deriv_j
        output[:nb] /= self.grid.bin_widths
        output = np.reshape(output, out_shape)
        if derivative:
            for i in range(out_len):
                rate_deriv[:nb,i] /= self.grid.bin_widths
            return output, rate_deriv
        else:
            return output

    def to_netcdf(self, netcdf_file):
        netcdf_file.write_dimension('boundary_str_len',
                                    self.boundary_str_len)
        netcdf_file.write_characters('boundary', self.boundary,
                                     'boundary_str_len',
                                     'Largest bin boundary condition')
        netcdf_file.write_dimension('kernel_sparsity_dim', self.max_num)
        netcdf_file.write_array('kernel_tensor_data', self.data,
            'f8', ('num_bins', 'num_bins', 'kernel_sparsity_dim'), '1',
            'Nondimensionalized kernel tensor data')

    @classmethod
    def from_netcdf(cls, netcdf_file, kernel, grid):
        boundary = netcdf_file.read_characters('boundary')
        data = netcdf_file.read_array('kernel_tensor_data')
        return KernelTensor(kernel, grid, boundary=boundary, data=data)
