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

"""Class for the collision-coalescence tensor in JEFE's bin model."""

import numpy as np

from bin_model.basis import PolynomialOnInterval

class CollisionTensor():
    """
    Represent a collision kernel evaluated on a particular mass grid.

    Initialization arguments:
    grid - A MassGrid object defining the bins.
    boundary (optional) - Upper boundary condition. If 'open', then particles
                          that are created larger than the largest bin size
                          "fall out" of the box. If 'closed', these particles
                          are placed in the largest bin. Defaults to 'open'.
    ckern (optional) - A CollisionKernel object to discretize.
    data (optional) - Precalculated collision tensor data.

    Exactly one of the ckern or data arguments must be supplied.
    """

    boundary_str_len = 16
    """Length of string specifying boundary condition for largest bin."""

    def __init__(self, grid, boundary=None, ckern=None, data=None):
        self.ckern = ckern
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
            if ckern is not None:
                raise RuntimeError("cannot supply both ckern and data to"
                                   " CollisionTensor constructor")
            self.data = data
            return
        if ckern is None:
            raise RuntimeError("must provide either ckern or data to"
                               " construct a CollisionTensor")
        integrals = self._calc_kernel_integrals(ckern)
        const = grid.constants
        scaling = const.mass_conc_scale * const.time_scale / const.std_mass
        self.data = scaling * integrals

    def _calc_kernel_integrals(self, ckern):
        """Integrate kernel to get contributions to entries in self.data."""
        nb = self.grid.num_bins
        bb = self.grid.bin_bounds
        integrals = np.zeros((self.max_num, nb, nb))
        # Largest bin for output is last in-range bin for closed boundary, but
        # is the out-of-range "bin" going to infinity for open boundary.
        if self.boundary == 'closed':
            high_bin = nb - 1
        else:
            high_bin = nb
        for k in range(nb):
            basis_x = PolynomialOnInterval(bb[k], bb[k+1], 0)
            for l in range(nb):
                basis_y = PolynomialOnInterval(bb[l], bb[l+1], 0)
                idx = self.idxs[k,l]
                for i in range(self.nums[k,l]):
                    zidx = idx + i
                    if zidx == high_bin:
                        top_bound = np.inf
                    else:
                        top_bound = bb[zidx+1]
                    integrals[i,k,l] = ckern.integrate_over_bins(
                        basis_x, basis_y, (bb[zidx], top_bound))
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
        f_shaped = np.reshape(f.copy(), (f_len, 1))
        f_shaped[:nb] /= np.reshape(self.grid.bin_widths, (nb, 1))
        rate = self._calc_rate(f_shaped, out_flux, out_len)
        output = np.reshape(rate, out_shape)
        if derivative:
            deriv = self._calc_deriv(f_shaped, out_flux, out_len)
            for i in range(nb):
                deriv[:,i] /= self.grid.bin_widths[i]
            return output, deriv
        return output

    def _calc_rate(self, f, out_flux, out_len):
        """Do tensor contraction to calculate rate for calc_rate."""
        nb = self.grid.num_bins
        f_outer = np.dot(f[:nb], np.transpose(f[:nb]))
        rate = np.zeros((out_len+self.max_num,))
        for i in range(self.max_num):
            dfdt_term = f_outer * self.data[i,:,:]
            # Removal terms.
            rate[:nb] -= np.sum(dfdt_term,axis=1)
            # Production terms.
            np.add.at(rate, self.idxs + i, dfdt_term)
        if out_flux:
            rate[out_len-1] = rate[out_len-1:].sum()
        return rate[:out_len]

    def _calc_deriv(self, f, out_flux, out_len):
        """Calculate derivative for calc_rate."""
        nb = self.grid.num_bins
        rate_deriv = np.zeros((out_len+self.max_num, out_len))
        for i in range(self.max_num):
            for k in range(nb):
                deriv_k = self.data[i,k,:nb] * f.flat[:nb]
                deriv_l = self.data[i,k,:] * f[k]
                rate_deriv[k,k] -= deriv_k.sum()
                rate_deriv[k,:nb] -= deriv_l
                for l in range(nb):
                    zidx = self.idxs[k,l] + i
                    rate_deriv[zidx,k] += deriv_k[l]
                    rate_deriv[zidx,l] += deriv_l[l]
        if out_flux:
            rate_deriv[out_len-1,:] = rate_deriv[out_len-1:,:].sum(axis=0)
        return rate_deriv[:out_len,:].copy() # Copy to make data contiguous.

    def to_netcdf(self, netcdf_file):
        """Write collision tensor data to netCDF file."""
        netcdf_file.write_dimension('boundary_str_len',
                                    self.boundary_str_len)
        netcdf_file.write_characters('boundary', self.boundary,
                                     'boundary_str_len',
                                     'Largest bin boundary condition')
        netcdf_file.write_dimension('tensor_sparsity_dim', self.max_num)
        netcdf_file.write_array('collision_tensor_data', self.data,
            'f8', ('tensor_sparsity_dim', 'num_bins', 'num_bins'), '1',
            'Nondimensionalized collision tensor data')

    @classmethod
    def from_netcdf(cls, netcdf_file, grid):
        """Retrieve collision tensor data from netCDF file."""
        boundary = netcdf_file.read_characters('boundary')
        data = netcdf_file.read_array('collision_tensor_data')
        return CollisionTensor(grid, boundary=boundary, data=data)
