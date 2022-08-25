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

"""A bin model with associated adjoint model used by JEFE."""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import netCDF4 as nc4

from bin_model.constants import ModelConstants
from bin_model.descriptor import ModelStateDescriptor
from bin_model.kernel import Kernel, LongKernel, HallKernel
from bin_model.kernel_tensor import KernelTensor
from bin_model.mass_grid import MassGrid, GeometricMassGrid
from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
from bin_model.state import ModelState
from bin_model.transform import Transform, IdentityTransform, LogTransform, \
    QuadToLogTransform


class Integrator:
    """
    Integrate a model state in time.

    Methods:
    integrate_raw
    integrate
    to_netcdf

    Class methods:
    from_netcdf
    """

    integrator_type_str_len = 64
    """Length of integrator_type string on file."""

    def integrate_raw(self, t_len, state, proc_tens):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        raise NotImplementedError

    def integrate(self, t_len, state, proc_tens):
        """Integrate the state and return an Experiment.

        Arguments:
        t_len - Length of time to integrate over (seconds).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.

        Returns an Experiment object summarizing the integration and all inputs
        to it.
        """
        tscale = self.constants.time_scale
        desc = state.desc
        times, raws = self.integrate_raw(t_len / tscale, state, proc_tens)
        times = times * tscale
        ddn = desc.dsd_deriv_num
        if ddn > 0:
            nb = desc.mass_grid.num_bins
            num_step = len(times) - 1
            ddsddt = np.zeros((num_step+1, nb))
            states = [ModelState(desc, raws[i,:]) for i in range(num_step+1)]
            for i in range(num_step+1):
                ddsddt[i,:] = states[i].dsd_time_deriv_raw(proc_tens)[:nb]
            pn = desc.perturb_num
            if pn > 0:
                zeta_cov = np.zeros((num_step+1, ddn+1, ddn+1))
                for i in range(num_step+1):
                    zeta_cov[i,:,:] = states[i].zeta_cov_raw(ddsddt[i,:])
            else:
                zeta_cov = None
        else:
            ddsddt = None
            zeta_cov = None
        return Experiment(desc, proc_tens, self, times, raws,
                          ddsddt=ddsddt, zeta_cov=zeta_cov)

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        raise NotImplementedError

    @classmethod
    def from_netcdf(self, netcdf_file, constants):
        """Read Integrator from netCDF file.

        Arguments:
        constants - The ModelConstants object.
        """
        integrator_type = netcdf_file.read_characters("integrator_type")
        if integrator_type == "RK45":
            dt = netcdf_file.read_scalar("dt")
            return RK45Integrator(constants, dt)
        else:
            assert False, "integrator_type on file not recognized"


class RK45Integrator(Integrator):
    """
    Integrate a model state in time using the SciPy RK45 implementation.

    Initialization arguments:
    constants - The ModelConstants object.
    dt - Max time step at which to calculate the results.

    Methods:
    integrate_raw
    """
    def __init__(self, constants, dt):
        self.constants = constants
        self.dt = dt
        self.dt_raw = dt / constants.time_scale

    def integrate_raw(self, t_len, state, proc_tens):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        dt = self.dt_raw
        tol = dt * 1.e-10
        num_step = int(t_len / dt)
        if t_len - (num_step * dt) > tol:
            num_step = num_step + 1
        times = np.linspace(0., t_len, num_step+1)
        raw_len = len(state.raw)
        rate_fun = lambda t, raw: \
            ModelState(state.desc, raw).time_derivative_raw(proc_tens)
        atol = 1.e-6 * np.ones(raw_len)
        pn = state.desc.perturb_num
        pcidx, _ = state.desc.perturb_chol_loc()
        for i in range(pn):
            offset = (((i+1)*(i+2)) // 2) - 1
            atol[pcidx+offset] = 1.e-300
        solbunch = solve_ivp(rate_fun, (times[0], times[-1]), state.raw,
                             method='RK45', t_eval=times, max_step=self.dt,
                             atol=atol)
        assert solbunch.status == 0, \
            "integration failed: " + solbunch.message
        if state.desc.perturb_num > 0:
            for i in range(num_step+1):
                pc = state.desc.perturb_cov_raw(solbunch.y[:,i])
                assert np.all(la.eigvalsh(pc) >= 0.), \
                        "negative covariance occurred at: " \
                        + str(solbunch.t[i])
        output = np.transpose(solbunch.y)
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "RK45",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")


class Experiment:
    """
    Collect all data produced by a particular model integration.

    Initialization arguments:
    desc - The ModelStateDescriptor.
    proc_tens - Process tensors used to perform an integration.
    integrator - Integrator that produced the integration.
    times - Times at which snapshot data is output.
            num_time_steps will be 1 less than the length of this array.
    raws - A 2-D array of raw state vectors, where the first dimension is the
           number of output times and the second dimension is the length of the
           state vector for each time.
    ddsddt (optional) - Raw derivative of DSD data.
                        Shape is `(num_time_steps, num_bins)`.
    zeta_cov - Raw covariance of DSD derivative variables (including time).
               Shape is `(num_time_steps, dsd_deriv_num+1, dsd_deriv_num+1)`.

    Attributes:
    num_time_steps - Number of time steps in integration.

    Methods:
    get_moments_and_covariances
    to_netcdf

    Class methods:
    from_netcdf
    """
    def __init__(self, desc, proc_tens, integrator, times, raws,
                 ddsddt=None, zeta_cov=None):
        self.constants = desc.constants
        self.mass_grid = desc.mass_grid
        self.desc = desc
        self.proc_tens = proc_tens
        self.integrator = integrator
        self.times = times
        self.num_time_steps = len(times)
        self.raws = raws
        self.states = [ModelState(self.desc, raws[i,:])
                       for i in range(self.num_time_steps)]
        self.ddsddt = ddsddt
        self.zeta_cov = zeta_cov

    def get_moments_and_covariances(self, wvs, times=None):
        """Calculate moments and their error covariances.

        Arguments:
        wvs - An array where each row is a weight vector.
        times (optional) - Array of times to sample. If not specified, all
                           times in this experiment will be returned.

        Returns a tuple `(lfs, lf_cov)`, where lfs is an array of moments of
        shape `(num_time_steps, lf_num)`, where lf_num is the number of moments
        requested, and lf_cov is an array of shape
        `(num_time_steps, lf_num, lf_num)`, which gives the covariance matrix
        at each time of the requested moments.
        """
        assert (self.ddsddt is not None) and (self.zeta_cov is not None), \
            "experiment did not produce covariance data to calculate with"
        need_reshape = len(wvs.shape) == 1
        if need_reshape:
            wvs = wvs[None,:]
        lf_num = wvs.shape[0]
        if times is None:
            nt = self.num_time_steps
        else:
            nt = len(times)
        lfs = np.zeros((nt, lf_num))
        lf_cov = np.zeros((nt, lf_num, lf_num))
        for i in range(nt):
            if times is None:
                it = i
            else:
                it = times[i]
            deriv = np.zeros((lf_num, self.desc.dsd_deriv_num+1))
            for j in range(lf_num):
                lfs[i,j], deriv[j,:] = \
                    self.states[it].linear_func_raw(wvs[j,:],
                                                    derivative=True,
                                                    dfdt=self.ddsddt[it,:])
            lf_cov[i,:,:] = np.dot(np.dot(deriv, self.zeta_cov[it,:,:]),
                                   deriv.T)
        if need_reshape:
            return lfs[:,0], lf_cov[:,0,0]
        else:
            return lfs, lf_cov

    def to_netcdf(self, netcdf_file):
        """Write Experiment to netCDF file."""
        netcdf_file.write_dimension('time', self.num_time_steps)
        netcdf_file.write_array("time", self.times,
            "f8", ['time'], "s",
            "Model time elapsed since start of integration")
        netcdf_file.write_dimension('raw_state_len', self.raws.shape[1])
        netcdf_file.write_array("raw_state_data", self.raws,
            "f8", ['time', 'raw_state_len'], "1",
            "Raw, unstructured, nondimensionalized model state information")
        if self.ddsddt is not None:
            netcdf_file.write_array("ddsddt", self.ddsddt,
                "f8", ['time', 'num_bins'], "1",
                "Nondimensionalized time derivative of DSD")
        if self.zeta_cov is not None:
            netcdf_file.write_dimension("dsd_deriv_num+1",
                                        self.desc.dsd_deriv_num+1)
            netcdf_file.write_array("zeta_cov", self.zeta_cov,
                "f8", ['time', 'dsd_deriv_num+1', 'dsd_deriv_num+1'], "1",
                "Nondimensionalized error covariance of dsd_deriv variables "
                "and time given the perturbed variable covariance in the "
                "corresponding state")

    @classmethod
    def from_netcdf(self, netcdf_file, desc, proc_tens, integrator):
        """Read Experiment from netCDF file.

        Arguments:
        desc - ModelStateDescriptor object used to construct Experiment.
        proc_tens - Process tensor list used to construct Experiment.
        integrator - Integrator used to construct Experiment.
        """
        times = netcdf_file.read_array("time")
        raws = netcdf_file.read_array("raw_state_data")
        if netcdf_file.variable_is_present("ddsddt"):
            ddsddt = netcdf_file.read_array("ddsddt")
        else:
            ddsddt = None
        if netcdf_file.variable_is_present("zeta_cov"):
            zeta_cov = netcdf_file.read_array("zeta_cov")
        else:
            zeta_cov = None
        return Experiment(desc, proc_tens, integrator, times, raws,
                          ddsddt=ddsddt, zeta_cov=zeta_cov)


class NetcdfFile:
    """
    Read/write model objects from/to a netCDF file.

    Initialization arguments:
    dataset - netCDF4 Dataset corresponding to the file open for I/O.

    Methods:
    variable_is_present
    write_scalar
    read_scalar
    write_dimension
    read_dimension
    write_characters
    read_characters
    write_array
    read_array
    write_constants
    read_constants
    write_kernel
    read_kernel
    write_grid
    read_grid
    write_kernel_tensor
    read_kernel_tensor
    write_cgk
    read_cgk
    write_descriptor
    read_descriptor
    write_integrator
    read_integrator
    write_experiment
    read_experiment
    write_full_experiment
    read_full_experiment
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
            return self.nc[name][...]

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
                assert not isinstance(value, str), \
                    "too many dimensions provided for string iterable"
                assert dim_lens[0] == len(value), \
                    "input string iterable is wrong shape for given " \
                    "dimensions"
                for i in range(dim_lens[0]):
                    assert_correct_shape(dim_lens[1:], value[i])
            else:
                assert isinstance(value, str), \
                    "too few dimensions provided for string iterable " \
                    "or at least one value is not a string"
                assert dim_lens[0] >= len(value), \
                    "some input strings are too long for given array dimension"
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
        assert value.shape == \
            tuple([self.read_dimension(dim) for dim in dims]), \
            "value shape does not match dimensions we were given to write to"
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

    def write_kernel(self, kernel):
        """Write a Kernel object to a netCDF file."""
        kernel.to_netcdf(self)

    def read_kernel(self, constants):
        """Read a Kernel object from a netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the Kernel.
        """
        return Kernel.from_netcdf(self, constants)

    def write_mass_grid(self, mass_grid):
        """Write a MassGrid object to a netCDF file."""
        mass_grid.to_netcdf(self)

    def read_mass_grid(self, constants):
        """Read a MassGrid object from a netCDF file.

        Arguments:
        constants - ModelConstants object to use in constructing the MassGrid.
        """
        return MassGrid.from_netcdf(self, constants)

    def write_kernel_tensor(self, ktens):
        """Write a KernelTensor object to a netCDF file."""
        ktens.to_netcdf(self)

    def read_kernel_tensor(self, grid):
        """Read a KernelTensor object from a netCDF file.

        Arguments:
        grid - MassGrid object to use in constructing the MassGrid.
        """
        return KernelTensor.from_netcdf(self, grid)

    def write_cgk(self, ktens):
        """Write constants, grid, kernel, and tensor data to netCDF file.

        Arguments:
        ktens - KernelTensor object created with the ModelConstants, MassGrid,
                and Kernel objects that are to be stored.
        """
        self.write_constants(ktens.grid.constants)
        self.write_kernel(ktens.kernel)
        self.write_mass_grid(ktens.grid)
        self.write_kernel_tensor(ktens)

    def read_cgk(self):
        """Read constants, grid, kernel, and tensor data from netCDF file.

        Returns the tuple

            (constants, kernel, grid, ktens)

        with types

            (ModelConstants, Kernel, MassGrid, KernelTensor)
        """
        constants = self.read_constants()
        kernel = self.read_kernel(constants)
        grid = self.read_mass_grid(constants)
        ktens = self.read_kernel_tensor(grid)
        return constants, kernel, grid, ktens

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
