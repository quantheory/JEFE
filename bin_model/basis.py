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

class PolynomialOnInterval():
    """
    Represent a polynomial basis function with one root on a bounded interval.

    On the given integral, the function is: f(x) = (x-root)**(degree)

    Initialization arguments:
    lower_bound, upper_bound - Bounds over which the function is non-zero.
    degree - The degree (maximum exponent) of the polynomial.
    root - The sole zero of the polynomial.
    """
    def __init__(self, lower_bound, upper_bound, degree, root=None, scale=1.):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.degree = degree
        self.root = root
        self.scale = scale
        if degree == 0:
            if root is not None:
                raise RuntimeError("cannot specify the root of a constant"
                                   " function")
        else:
            if root is None:
                raise RuntimeError("must specify root of a non-constant"
                                   " polynomial")

    def __call__(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            if self.degree == 0: # Avoid 0**0 indeterminate form.
                return self.scale
            else:
                return self.scale * (x-self.root)**self.degree
        else:
            return 0.


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
        def create_basis_function(degree, bin_idx):
            lower = grid.bin_bounds[bin_idx]
            upper = grid.bin_bounds[bin_idx+1]
            if degree == 0:
                root = None
            else:
                root = lower
            scale = (upper-lower)**(-degree)
            return PolynomialOnInterval(lower, upper, degree, root=root,
                                        scale=scale)
        for i in range(degree+1):
            for j in range(grid.num_bins):
                self._funcs.append(create_basis_function(i, j))

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
