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

import numpy as np

class PiecewisePolyBasis():
    """
    Represent a piecewise polynomial basis defined sparsely over a mass grid.

    Initialization arguments:
    grid - A MassGrid object defining the bins.
    degree - The maximum polynomial degree of the bins.

    Attributes:
    size - Number of basis functions.
    """
    def __init__(self, grid, degree):
        self.grid = grid
        self.degree = degree
        self.size = (degree + 1) * grid.num_bins
        self._funcs = []
        def create_basis_function(bin_idx, degree):
            def func(x):
                if not \
                   (grid.bin_bounds[bin_idx] < x <= grid.bin_bounds[bin_idx+1]):
                    return 0.
                scaled_x = (x - grid.bin_bounds[bin_idx]) \
                    / grid.bin_widths[bin_idx]
                return scaled_x**degree
            return func
        for i in range(grid.num_bins):
            deg_funcs = []
            for j in range(degree+1):
                deg_funcs.append(create_basis_function(i,j))
            self._funcs.append(deg_funcs)

    def __getitem__(self, idx):
        return self._funcs[idx]

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_scalar('polynomial_basis_degree', self.degree,
                                 'i4', '1',
                                 'Maximum degree of piecewise polynomial basis.')

    @classmethod
    def from_netcdf(cls, netcdf_file, grid):
        degree = netcdf_file.read_scalar('polynomial_basis_degree')
        return PiecewisePolyBasis(grid, degree)
