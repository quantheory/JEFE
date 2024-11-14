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

import numba as nb
import numpy as np

from bin_model.basis import make_piecewise_polynomial_basis
from bin_model.collision_kernel import CoalescenceKernel

@nb.njit(nogil=True, cache=True)
def add_at(rate, idxs, dfdt_term, nb):
    """Numba-optimized function used to replace np.add.at."""
    for k in range(nb):
        for l in range(nb):
            rate[idxs[k,l]] += dfdt_term[k,l]

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

    def __init__(self, grid, basis=None, boundary=None, ckern=None, data=None):
        self.ckern = ckern
        self.grid = grid
        if boundary is None:
            boundary = 'open'
        self.boundary = boundary
        if basis is None:
            basis = make_piecewise_polynomial_basis(grid, 0)
        self.basis = basis
        idxs, nums, max_num = \
            CoalescenceKernel.construct_sparsity_structure(basis, grid,
                                                           boundary=boundary)
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
        nbasis = self.basis.size
        nb = self.grid.num_bins
        bb = self.grid.bin_bounds
        integrals = np.zeros((self.max_num, nbasis, nbasis))
        # Largest bin for output is last in-range bin for closed boundary, but
        # is the out-of-range "bin" going to infinity for open boundary.
        if self.boundary == 'closed':
            high_bin = nb - 1
        else:
            high_bin = nb
        for k in range(nbasis):
            basis_x = self.basis[k]
            for l in range(nbasis):
                basis_y = self.basis[l]
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
            add_at(rate, self.idxs + i, dfdt_term, nb)
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
        return CollisionTensor(grid, boundary=boundary, data=np.array(data))
