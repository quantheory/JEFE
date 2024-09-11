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

"""Bin model I/O using netCDF."""

import numpy as np
import netCDF4 as nc4

from bin_model.constants import ModelConstants
from bin_model.collision_kernel import CollisionKernel
from bin_model.mass_grid import MassGrid
from bin_model.collision_tensor import CollisionTensor
from bin_model.descriptor import ModelStateDescriptor
from bin_model.time import Integrator
from bin_model.experiment import Experiment


class NetcdfFile:
    """
    Read/write model objects from/to a netCDF file.

    Initialization arguments:
    dataset - netCDF4 Dataset corresponding to the file open for I/O.
    """
    def __init__(self, dataset):
        self.nc = dataset

    def variable_is_present(self, name):
        """Test if a variable with the given name is present on the file."""
        return name in self.nc.variables

    def write_scalar(self, name, value, dtype, units, description):
        """Write a scalar to a netCDF file.

        Arguments:
        name - Name of variable on file.
        value - Value to write.
        dtype - netCDF datatype of variable.
        units - String describing units of variable.
        description - Human-readable description of variable's meaning.
        """
        var = self.nc.createVariable(name, dtype)
        var.units = units
        var.description = description
        var[...] = value

    def read_scalar(self, name):
        """Read the named variable from a netCDF file."""
        if np.isnan(self.nc[name]):
            return None
        else:
            return self.nc[name][...].flat[0]

    def write_dimension(self, name, length):
        """Create a new dimension on a netCDF file.

        Arguments:
        name - Name of dimension.
        length - Size of dimension.
        """
        self.nc.createDimension(name, length)

    def read_dimension(self, name):
        """Retrieve the size of the named dimension on a netCDF file."""
        return len(self.nc.dimensions[name])

    def write_characters(self, name, value, dims, description):
        """Write a string or array of strings to netCDF as a character array.

        Arguments:
        name - Name of variable on file.
        value - Value to write.
        dims - List of strings naming the dimensions of array on the file.
               The last dimension will be used as the string length.
               If a scalar string needs to be written, this can be a string
               rather than a list.
        description - Human-readable description of variable's meaning.
        """
        if isinstance(dims, str):
            dims = [dims]
        dim_lens = []
        for dim in dims:
            dim_lens.append(self.read_dimension(dim))
        def assert_correct_shape(dim_lens, value):
            if len(dim_lens) > 1:
                if isinstance(value, str):
                    raise ValueError("too many dimensions provided for string"
                                     " iterable")
                if dim_lens[0] != len(value):
                    raise ValueError("input string iterable is wrong shape for"
                                     " given dimensions")
                for i in range(dim_lens[0]):
                    assert_correct_shape(dim_lens[1:], value[i])
            else:
                if not isinstance(value, str):
                    raise ValueError("too few dimensions provided for string"
                                     " iterable or at least one value is not a"
                                     " string")
                if dim_lens[0] < len(value):
                    raise ValueError("some input strings are too long for given"
                                     " array dimension")
        assert_correct_shape(dim_lens, value)
        var = self.nc.createVariable(name, 'S1', dims)
        var.description = description
        var[:] = nc4.stringtochar(np.array(value,
                                           'S{}'.format(dim_lens[-1])))

    def read_characters(self, name):
        """Read a string stored in a netCDF file as a character array."""
        shape = self.nc[name].shape
        def build_lists(shape, array):
            if len(shape) > 1:
                return [build_lists(shape[1:], array[i])
                        for i in range(shape[0])]
            else:
                return str(nc4.chartostring(array[:]))
        return build_lists(shape, self.nc[name])

    def write_array(self, name, value, dtype, dims, units, description):
        """Write an array to a netCDF file.

        Arguments:
        name - Name of variable on file.
        value - Value to write.
        dtype - netCDF datatype of variable.
        dims - List of strings naming the dimensions of array on the file.
        units - String describing units of variable.
        description - Human-readable description of variable's meaning.
        """
        if value.shape != tuple([self.read_dimension(dim) for dim in dims]):
            raise ValueError("value.shape does not match dimensions of array on"
                             " file")
        var = self.nc.createVariable(name, dtype, dims)
        var.units = units
        var.description = description
        var[...] = value

    def read_array(self, name):
        """Read an array from a netCDF file."""
        return self.nc[name][...]

    def write_constants(self, constants):
        """Write a ModelConstants object to a netCDF file."""
        constants.to_netcdf(self)

    def read_constants(self):
        """Read a ModelConstants object from a netCDF file."""
        return ModelConstants.from_netcdf(self)

    def write_collision_kernel(self, ckern):
        """Write a CollisionKernel object to a netCDF file."""
        ckern.to_netcdf(self)

    def read_collision_kernel(self, constants):
        """Read a CollisionKernel object from a netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the CollisionKernel.
        """
        return CollisionKernel.from_netcdf(self, constants)

    def write_mass_grid(self, mass_grid):
        """Write a MassGrid object to a netCDF file."""
        mass_grid.to_netcdf(self)

    def read_mass_grid(self, constants):
        """Read a MassGrid object from a netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the MassGrid.
        """
        return MassGrid.from_netcdf(self, constants)

    def write_collision_tensor(self, ctens):
        """Write a CollisionTensor object to a netCDF file."""
        ctens.to_netcdf(self)

    def read_collision_tensor(self, grid):
        """Read a CollisionTensor object from a netCDF file.

        Arguments:
        grid - MassGrid object to use in constructing the MassGrid.
        """
        return CollisionTensor.from_netcdf(self, grid)

    def write_ckgt(self, ctens):
        """Write constants, kernel, grid, and tensor data to netCDF file.

        Arguments:
        ctens - CollisionTensor object created with the ModelConstants,
                CollisionKernel, and MassGrid objects that are to be stored.
        """
        self.write_constants(ctens.grid.constants)
        self.write_collision_kernel(ctens.ckern)
        self.write_mass_grid(ctens.grid)
        self.write_collision_tensor(ctens)

    def read_ckgt(self):
        """Read constants, kernel, grid, and tensor data from netCDF file.

        Returns the tuple

            (constants, ckern, grid, ctens)

        with types

            (ModelConstants, CollisionKernel, MassGrid, CollisionTensor)
        """
        constants = self.read_constants()
        ckern = self.read_collision_kernel(constants)
        grid = self.read_mass_grid(constants)
        ctens = self.read_collision_tensor(grid)
        return constants, ckern, grid, ctens

    def write_descriptor(self, desc):
        """Write ModelStateDescriptor to netCDF file."""
        desc.to_netcdf(self)

    def read_descriptor(self, constants, mass_grid):
        """Read ModelStateDescriptor from netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the
                    descriptor.
        mass_grid - MassGrid object to use in constructing the descriptor.
        """
        return ModelStateDescriptor.from_netcdf(self, constants, mass_grid)

    def write_integrator(self, integrator):
        """Write Integrator to netCDF file."""
        integrator.to_netcdf(self)

    def read_integrator(self, constants):
        """Read Integrator from netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the
                    integrator.
        """
        return Integrator.from_netcdf(self, constants)

    def write_experiment(self, exp):
        """Write Experiment to netCDF file."""
        exp.to_netcdf(self)

    def read_experiment(self, desc, proc_tens, integrator):
        """Read Experiment from netCDF file.

        Arguments:
        desc - ModelStateDescriptor object used to construct Experiment.
        proc_tens - Process tensor list used to construct Experiment.
        integrator - Integrator used to construct Experiment.
        """
        return Experiment.from_netcdf(self, desc, proc_tens, integrator)

    def write_full_experiment(self, exp, proc_tens_files):
        """Write experiment with all data to netCDF file.

        Arguments:
        exp - The Experiment to write.
        proc_tens_files - A list of strings containing the location of files
                          with process tensor data.
        """
        nfile = len(proc_tens_files)
        self.write_dimension('proc_tens_file_num', nfile)
        max_len = 0
        for i in range(nfile):
            max_len = max(max_len, len(proc_tens_files[i]))
        self.write_dimension('proc_tens_file_str_len', max_len)
        self.write_characters('proc_tens_files', proc_tens_files,
            ['proc_tens_file_num', 'proc_tens_file_str_len'],
            "Files containing process tensors used in integration")
        desc = exp.desc
        self.write_constants(desc.constants)
        self.write_mass_grid(desc.mass_grid)
        self.write_descriptor(desc)
        self.write_integrator(exp.integrator)
        exp.to_netcdf(self)

    def read_full_experiment(self, proc_tens):
        """Read all experiment data except process tensors from netCDF file.

        Arguments:
        proc_tens - Process tensor list used to construct Experiment.
        """
        const = self.read_constants()
        grid = self.read_mass_grid(const)
        desc = self.read_descriptor(const, grid)
        integrator = self.read_integrator(const)
        return self.read_experiment(desc, proc_tens, integrator)
