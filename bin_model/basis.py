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

from abc import ABC, abstractmethod

class BasisFunction(ABC):
    """
    Represent a polynomial basis function.

    Attributes:
    lower_bound, upper_bound - Bounds over which the function is non-zero.
    """

    basis_function_type_string_len = 32
    """Length of a basis function type string on file."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __eq__(self, other):
        type_string = self.type_string()
        if type_string == "PolynomialOnInterval":
            return (type_string == other.type_string()) \
                and (self.lower_bound == other.lower_bound) \
                and (self.upper_bound == other.upper_bound) \
                and (self.degree == other.degree) \
                and (self.root == other.root) \
                and (self.scale == other.scale)
        elif type_string == "DiracDelta":
            return (type_string == other.type_string()) \
                and (self.location == other.location) \
                and (self.scale == other.scale)
        else:
            raise NotImplementedError("equality testing not implemented for type '{}'"
                                      .format(type_string))

    @classmethod
    def from_parameters(self, type_string, integer_parameters, real_parameters):
        if type_string == "PolynomialOnInterval":
            if len(integer_parameters) != 1:
                raise ValueError("wrong number of integer parameters: {}"
                                 .format(len(integer_parameters)))
            degree = integer_parameters[0]
            if (degree == 0 and len(real_parameters) != 3) \
               or (degree != 0 and len(real_parameters) != 4):
                raise ValueError("wrong number of real parameters for degree {}: {}"
                                 .format(degree, len(real_parameters)))
            lower_bound = real_parameters[0]
            upper_bound = real_parameters[1]
            if degree == 0:
                root = None
                scale = real_parameters[2]
            else:
                root = real_parameters[2]
                scale = real_parameters[3]
            return PolynomialOnInterval(lower_bound, upper_bound, degree, root,
                                        scale=scale)
        elif type_string == "DiracDelta":
            if len(integer_parameters) != 0:
                raise ValueError("wrong number of integer parameters: {}"
                                 .format(len(integer_parameters)))
            if len(real_parameters) != 2:
                raise ValueError("wrong number of real parameters: {}"
                                 .format(len(real_parameters)))
            location = real_parameters[0]
            scale = real_parameters[1]
            return DiracDelta(location, scale=scale)
        else:
            raise ValueError("type name '{}' not recognized".format(type_string))

    @abstractmethod
    def __call__(self, x):
        """Retrieve the value of the basis function at a given point."""

    @abstractmethod
    def type_string(self):
        """Get string representing the type of this basis function."""

    @abstractmethod
    def integer_parameters(self):
        """Get integer parameters of this basis function as a list."""

    @abstractmethod
    def real_parameters(self):
        """Get floating-point parameters of this basis function as a list."""


class PolynomialOnInterval(BasisFunction):
    """
    Represent a polynomial basis function with one root on a bounded interval.

    On the given interval, the function is: f(x) = scale * (x-root)**(degree)

    Initialization arguments:
    lower_bound, upper_bound - Bounds over which the function is non-zero.
    degree - The degree (maximum exponent) of the polynomial.
    root - The sole zero of the polynomial.
    """
    def __init__(self, lower_bound, upper_bound, degree, root=None, scale=1.):
        super().__init__(lower_bound, upper_bound)
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

    def type_string(self):
        """Get string representing the type of this basis function."""
        return "PolynomialOnInterval"

    def integer_parameters(self):
        """Get integer parameters of this basis function as a list."""
        return [self.degree]

    def real_parameters(self):
        """Get floating-point parameters of this basis function as a list."""
        if self.degree == 0:
            return [self.lower_bound, self.upper_bound, self.scale]
        else:
            return [self.lower_bound, self.upper_bound, self.root, self.scale]

class DiracDelta(BasisFunction):
    """
    Represent a Dirac delta function with a given location and scaling.

    Initialization arguments:
    location - Location where the delta function has non-zero weight.
    scale - Quantity to scale the delta function by, i.e. the result of any
            definite integral over a region that encompasses the location.

    Calling this object raises a TypeError, since any code handling a DiracDelta
    should be treating it as a special case, rather than blindly evaluating it
    as a black-box function.
    """
    def __init__(self, location, scale=1.):
        super().__init__(location, location)
        self.location = location
        self.scale = scale

    def __call__(self, x):
        raise TypeError("cannot call a DiracDelta object")

    def type_string(self):
        """Get string representing the type of this basis function."""
        return "DiracDelta"

    def integer_parameters(self):
        """Get integer parameters of this basis function as a list."""
        return []

    def real_parameters(self):
        """Get floating-point parameters of this basis function as a list."""
        return [self.location, self.scale]


def make_piecewise_polynomial_basis(grid, degree):
    """
    Create a piecewise polynomial basis defined sparsely over a mass grid.

    Arguments:
    grid - A MassGrid object defining the bins.
    degree - The maximum polynomial degree of the basis functions.
    """
    funcs = []
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
            funcs.append(create_basis_function(i, j))
    return Basis(funcs)

def make_delta_on_bounds_basis(grid):
    """
    Create a basis consisting of delta functions at each bin boundary.

    Arguments:
    grid - A MassGrid object defining the bins.

    The basis includes delta functions at the grid boundaries, i.e. the number
    of returned basis functions is equal to the number of bins plus one.
    """
    funcs = [DiracDelta(location) for location in grid.bin_bounds]
    return Basis(funcs)


class Basis():
    """
    Represent a basis defined sparsely over a mass grid.

    Initialization arguments:
    funcs - A list of the BasisFunction objects forming the basis.

    Attributes:
    size - Number of basis functions.

    The basis functions can be accessed as if in a list, e.g. `basis[0]` is the
    first basis function in `basis`.
    """
    def __init__(self, funcs):
        self.size = len(funcs)
        self._funcs = funcs

    def __getitem__(self, idx):
        return self._funcs[idx]

    def __add__(self, other):
        return Basis(self._funcs + other._funcs)

    def __len__(self):
        return self.size

    def indices_of_subset(self, other):
        """Find the indices of another basis's functions within this basis.

        If the other basis is determined to not be a subset of this basis, then
        this function returns None.
        """
        try:
            indices = [self._funcs.index(f) for f in other]
            return indices
        except ValueError:
            return None

    def to_netcdf(self, netcdf_file):
        """Write internal state to netCDF file."""
        netcdf_file.write_dimension('num_basis_functions', self.size)
        type_list = [f.type_string() for f in self._funcs]
        netcdf_file.write_dimension('basis_function_type_string_len',
                                    BasisFunction.basis_function_type_string_len)
        netcdf_file.write_characters('basis_function_type_strings',
                                     type_list,
                                     ('num_basis_functions', 'basis_function_type_string_len'),
                                     "Types of basis functions")
        integer_param_list = [f.integer_parameters() for f in self._funcs]
        integer_num_params = np.array([len(params) for params in integer_param_list])
        netcdf_file.write_dimension('max_basis_integer_parameters',
                                    max(integer_num_params))
        netcdf_file.write_array('num_basis_integer_parameters',
                                integer_num_params,
                                'i4', ('num_basis_functions',),
                                '1', "Number of integer parameters defining"
                                " each basis function")
        integer_params = np.zeros((self.size, max(integer_num_params)))
        for i in range(self.size):
            nparam = integer_num_params[i]
            integer_params[i,:nparam] = integer_param_list[i]
        netcdf_file.write_array('basis_integer_parameters', integer_params,
                                'i4',
                                ('num_basis_functions', 'max_basis_integer_parameters'),
                                '1', "Integer parameters for each basis function")
        real_param_list = [f.real_parameters() for f in self._funcs]
        real_num_params = np.array([len(params) for params in real_param_list])
        netcdf_file.write_dimension('max_basis_real_parameters',
                                    max(real_num_params))
        netcdf_file.write_array('num_basis_real_parameters', real_num_params,
                                'i4', ('num_basis_functions',),
                                '1', "Number of real parameters defining"
                                " each basis function")
        real_params = np.zeros((self.size, max(real_num_params)))
        for i in range(self.size):
            nparam = real_num_params[i]
            real_params[i,:nparam] = real_param_list[i]
        netcdf_file.write_array('basis_real_parameters', real_params,
                                'f8',
                                ('num_basis_functions', 'max_basis_real_parameters'),
                                '1', "Real parameters for each basis function.")

    @classmethod
    def from_netcdf(cls, netcdf_file, grid):
        size = netcdf_file.read_dimension('num_basis_functions')
        type_list = netcdf_file.read_characters('basis_function_type_strings')
        integer_num_params = netcdf_file.read_array("num_basis_integer_parameters")
        integer_params = netcdf_file.read_array("basis_integer_parameters")
        real_num_params = netcdf_file.read_array("num_basis_real_parameters")
        real_params = netcdf_file.read_array("basis_real_parameters")
        funcs = []
        for i in range(size):
            funcs.append(BasisFunction.from_parameters(
                type_list[i],
                integer_params[i,:integer_num_params[i]],
                real_params[i,:real_num_params[i]]
            ))
        return Basis(funcs)
