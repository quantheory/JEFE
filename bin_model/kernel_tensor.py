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

"""Class for the collision-coalescence kernel tensor in JEFE's bin model."""

import numpy as np

class KernelTensor():
    """
    Represent a collision kernel evaluated on a particular mass grid.

    Initialization arguments:
    grid - A MassGrid object defining the bins.
    boundary (optional) - Upper boundary condition. If 'open', then particles
                          that are created larger than the largest bin size
                          "fall out" of the box. If 'closed', these particles
                          are placed in the largest bin. Defaults to 'open'.
    kernel (optional) - A Kernel object representing the collision kernel.
    data (optional) - Precalculated kernel tensor data.

    Exactly one of the kernel or data arguments must be supplied.

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

    def __init__(self, grid, boundary=None, kernel=None, data=None):
        self.kernel = kernel
        self.grid = grid
        if boundary is None:
            boundary = 'open'
        self.boundary = boundary
        idxs, nums, max_num = \
            grid.construct_sparsity_structure(boundary=boundary)
        self.idxs = idxs
        self.nums = nums
        self.max_num = max_num
        if data is not None:
            if kernel is not None:
                raise RuntimeError("cannot supply both kernel and data to"
                                   " KernelTensor constructor")
            self.data = data
            return
        if kernel is None:
            raise RuntimeError("must provide either kernel or data to"
                               " construct a KernelTensor")
        integrals = self._calc_kernel_integrals(kernel)
        const = grid.constants
        scaling = const.mass_conc_scale * const.time_scale / const.std_mass
        self.data = scaling * integrals

    def _calc_kernel_integrals(self, kernel):
        """Integrate kernel to get contributions to entries in self.data."""
        nb = self.grid.num_bins
        bb = self.grid.bin_bounds
        integrals = np.zeros((nb, nb, self.max_num))
        # Largest bin for output is last in-range bin for closed boundary, but
        # is the out-of-range "bin" going to infinity for open boundary.
        if self.boundary == 'closed':
            high_bin = nb - 1
        else:
            high_bin = nb
        for i in range(nb):
            for j in range(nb):
                idx = self.idxs[i,j]
                for k in range(self.nums[i,j]):
                    zidx = idx + k
                    if zidx == high_bin:
                        top_bound = np.inf
                    else:
                        top_bound = bb[zidx+1]
                    integrals[i,j,k] = kernel.integrate_over_bins(
                        (bb[i], bb[i+1]), (bb[j], bb[j+1]),
                        (bb[zidx], top_bound))
        return integrals

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
        if not nb <= f_len < nb+2:
            raise ValueError("invalid f length: "+str(f_len))
        if out_flux is None:
            out_flux = f_len == nb + 1
            out_len = f_len
            out_shape = f.shape
        else:
            out_len = nb + 1 if out_flux else nb
            out_shape = (out_len,)
        f = np.reshape(f, (f_len, 1))
        rate = self._calc_rate(f, out_flux, out_len)
        output = np.reshape(rate, out_shape)
        if derivative:
            return output, self._calc_deriv(f, out_flux, out_len)
        return output

    def _calc_rate(self, f, out_flux, out_len):
        """Do tensor contraction to calculate rate for calc_rate."""
        nb = self.grid.num_bins
        f_outer = np.dot(f, np.transpose(f))
        rate = np.zeros((out_len,))
        for i in range(nb):
            for j in range(nb):
                idx = self.idxs[i,j]
                fprod = f_outer[i,j]
                for k in range(self.nums[i,j]):
                    zidx = idx + k
                    dfdt_term = fprod * self.data[i,j,k]
                    rate[i] -= dfdt_term
                    if zidx < nb or out_flux:
                        rate[zidx] += dfdt_term
        rate[:nb] /= self.grid.bin_widths
        return rate

    def _calc_deriv(self, f, out_flux, out_len):
        """Calculate derivative for calc_rate."""
        nb = self.grid.num_bins
        rate_deriv = np.zeros((out_len, out_len))
        for i in range(nb):
            for j in range(nb):
                idx = self.idxs[i,j]
                for k in range(self.nums[i,j]):
                    zidx = idx + k
                    deriv_i = self.data[i,j,k] * f[j]
                    deriv_j = self.data[i,j,k] * f[i]
                    rate_deriv[i,i] -= deriv_i
                    rate_deriv[i,j] -= deriv_j
                    if zidx < nb or out_flux:
                        rate_deriv[zidx,i] += deriv_i
                        rate_deriv[zidx,j] += deriv_j
        for i in range(out_len):
            rate_deriv[:nb,i] /= self.grid.bin_widths
        return rate_deriv

    def to_netcdf(self, netcdf_file):
        """Write kernel tensor data to netCDF file."""
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
    def from_netcdf(cls, netcdf_file, grid):
        """Retrieve kernel tensor data from netCDF file."""
        boundary = netcdf_file.read_characters('boundary')
        data = netcdf_file.read_array('kernel_tensor_data')
        return KernelTensor(grid, boundary=boundary, data=data)
