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

from bin_model.math_utils import gamma_dist_d, gamma_dist_d_lam_deriv, \
    gamma_dist_d_nu_deriv
from bin_model.constants import ModelConstants
from bin_model.kernel import Kernel, LongKernel, HallKernel
from bin_model.mass_grid import MassGrid, GeometricMassGrid


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
        assert nb <= f_len < nb + 2, "invalid f length: "+str(f_len)
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


class Transform:
    """
    Represent a transformation of a prognostic variable.

    Methods:
    transform
    derivative
    second_over_first_derivative
    type_string
    get_parameters

    Class methods:
    from_params
    """

    transform_type_str_len = 32
    "Length of a transform type string on file."

    def transform(self, x):
        """Transform the variable."""
        raise NotImplementedError

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        raise NotImplementedError

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        raise NotImplementedError

    def type_string(self):
        """Get string representing type of transform."""
        raise NotImplementedError

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        raise NotImplementedError

    @classmethod
    def from_params(self, type_str, params):
        if type_str == "Identity":
            return IdentityTransform()
        elif type_str == "Log":
            return LogTransform()
        elif type_str == "QuadToLog":
            return QuadToLogTransform(params[0])
        else:
            assert False, "transform type string not recognized"


class IdentityTransform(Transform):
    """
    Transform a prognostic variable by doing nothing.
    """
    def transform(self, x):
        """Transform the variable."""
        return x

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        return 1.

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        return 0.

    def type_string(self):
        """Get string representing type of transform."""
        return "Identity"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return []


class LogTransform(Transform):
    """
    Transform a prognostic variable using the natural logarithm.
    """
    def transform(self, x):
        """Transform the variable."""
        return np.log(x)

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        return 1./x

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        return -1./x

    def type_string(self):
        """Get string representing type of transform."""
        return "Log"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return []


class QuadToLogTransform(Transform):
    """
    Transform a prognostic variable using a mix of a quadratic and logarithm.

    The transform represented by this class uses a quadratic near 0, and the
    natural logarithm for larger values. The logarithm is offset, and quadratic
    chosen so that the first and second derivatives are continuous.

    Initialization arguments:
    x0 - Length scale at which quadratic to logarithm transition occurs.
    """
    def __init__(self, x0):
        self.x0 = x0

    def transform(self, x):
        """Transform the variable."""
        xs = x / self.x0
        if xs >= 1.:
            return np.log(xs) + 1.5
        else:
            return -0.5 * (xs)**2 + 2. * xs

    def derivative(self, x):
        """Calculate the first derivative of the transformation."""
        xs = x / self.x0
        if xs >= 1.:
            return 1. / x
        else:
            return (-xs + 2.) / self.x0

    def second_over_first_derivative(self, x):
        """Calculate the second derivative divided by the first."""
        xs = x / self.x0
        if xs >= 1.:
            return -1. / x
        else:
            return -1. / ((-xs + 2.) * self.x0)

    def type_string(self):
        """Get string representing type of transform."""
        return "QuadToLog"

    def get_parameters(self):
        """Get parameters of this transform as a list."""
        return [self.x0]


class ModelStateDescriptor:
    """
    Describe the state variables contained in a ModelState.

    Initialization arguments:
    constants - A ModelConstants object.
    mass_grid - A MassGrid object defining the bins.
    dsd_deriv_names (optional) - List of strings naming variables with respect
                                 to which DSD derivatives are prognosed.
    dsd_deriv_scales (optional) - List of scales for the derivatives of the
                                  named variable. These scales will be applied
                                  in addition to mass_conc_scale. They default
                                  to 1.
    perturbed_variables (optional) - A list of tuples, with each tuple
        containing a weight vector, a transform, and a scale, in that order.
    perturbation_rate (optional) - A covariance matrix representing the error
        introduced to the perturbed variables per second.
    correction_time (optional) - Time scale over which the error covariance is
        nudged toward a corrected value.
    scale_inputs (optional) - Whether to scale the input variables. Default is
                              True. Setting this to False is mainly intended
                              for testing and I/O utility code.

    Attributes:
    constants - ModelConstants object used by this model.
    mass_grid - The grid of the DSD used by this model.
    dsd_deriv_num - Number of variables with respect to which the derivative of
                    the DSD is tracked.
    dsd_deriv_names - Names of variables with tracked derivatives.
    dsd_deriv_scales - Scales of variables with tracked derivatives.
                       These scales are applied on top of mass_conc_scale.

    Methods:
    state_len
    construct_raw
    dsd_loc
    fallout_loc
    dsd_deriv_loc
    fallout_deriv_loc
    perturb_chol_loc
    dsd_raw
    fallout_raw
    dsd_deriv_raw
    fallout_deriv_raw
    perturb_cov_raw
    to_netcdf

    Class Methods:
    from_netcdf
    """

    dsd_deriv_name_str_len = 64
    """Length of dsd derivative variable name strings on file."""

    def __init__(self, constants, mass_grid,
                 dsd_deriv_names=None, dsd_deriv_scales=None,
                 perturbed_variables=None, perturbation_rate=None,
                 correction_time=None, scale_inputs=None):
        if scale_inputs is None:
            scale_inputs = True
        self.constants = constants
        self.mass_grid = mass_grid
        if dsd_deriv_names is not None:
            self.dsd_deriv_num = len(dsd_deriv_names)
            assert len(set(dsd_deriv_names)) == self.dsd_deriv_num, \
                "duplicate derivatives found in list"
            self.dsd_deriv_names = dsd_deriv_names
            if dsd_deriv_scales is None:
                dsd_deriv_scales = np.ones((self.dsd_deriv_num,))
            assert len(dsd_deriv_scales) == self.dsd_deriv_num, \
                "dsd_deriv_scales length does not match dsd_deriv_names"
            # Convert to array to allow user to specify a list here...
            self.dsd_deriv_scales = np.array(dsd_deriv_scales)
        else:
            assert dsd_deriv_scales is None, \
                "cannot specify dsd_deriv_scales without dsd_deriv_names"
            self.dsd_deriv_num = 0
            self.dsd_deriv_names = []
            self.dsd_deriv_scales = np.zeros((0,))
        if perturbed_variables is not None:
            pn = len(perturbed_variables)
            nb = mass_grid.num_bins
            self.perturb_num = pn
            self.perturb_wvs = np.zeros((pn, nb))
            for i in range(pn):
                self.perturb_wvs[i,:] = perturbed_variables[i][0]
            self.perturb_transforms = [t[1] for t in perturbed_variables]
            self.perturb_scales = np.array([t[2] for t in perturbed_variables])
            self.perturbation_rate = np.zeros((pn, pn))
            if perturbation_rate is not None:
                assert perturbation_rate.shape == (pn, pn), \
                    "perturbation_rate is the wrong shape, should be " \
                    + str((pn, pn))
                for i in range(pn):
                    for j in range(pn):
                        self.perturbation_rate[i,j] = perturbation_rate[i,j]
                        if scale_inputs:
                            self.perturbation_rate[i,j] *= constants.time_scale \
                                / perturbed_variables[i][2] \
                                / perturbed_variables[j][2]
            if correction_time is not None:
                self.correction_time = correction_time
                if scale_inputs:
                    self.correction_time /= constants.time_scale
            else:
                assert pn == self.dsd_deriv_num + 1, \
                    "must specify correction time unless perturb_num is " \
                    "equal to dsd_deriv_num+1"
                self.correction_time = None
        else:
            assert perturbation_rate is None, \
                "cannot specify perturbation_rate without perturbed_variables"
            assert correction_time is None, \
                "cannot specify correction_time without perturbed_variables"
            self.perturb_num = 0

    def state_len(self):
        """Return the length of the state vector."""
        idx, num = self.perturb_chol_loc()
        return idx + num

    def construct_raw(self, dsd, fallout=None, dsd_deriv=None,
                      fallout_deriv=None, perturb_cov=None):
        """Construct raw state vector from individual variables.

        Arguments:
        dsd - Drop size distribution.
        fallout (optional) - Amount of third moment that has fallen out of the
                             model. If not specified, defaults to zero.
        dsd_deriv - DSD derivatives. Mandatory if dsd_deriv_num is not zero.
        fallout_deriv - Fallout derivatives. If not specified, defaults to
                        zero.
        perturb_cov - Covariance matrix for Gaussian perturbation. If not
                      specified, defaults to zero.

        Returns a 1-D array of size given by state_len().
        """
        raw = np.zeros((self.state_len(),))
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        pn = self.perturb_num
        mc_scale = self.constants.mass_conc_scale
        assert len(dsd) == nb, "dsd of wrong size for this descriptor's grid"
        idx, num = self.dsd_loc()
        raw[idx:idx+num] = dsd / self.constants.mass_conc_scale
        if fallout is None:
            fallout = 0.
        raw[self.fallout_loc()] = fallout / mc_scale
        if ddn > 0:
            assert dsd_deriv is not None, \
                "dsd_deriv input is required, but missing"
            assert (dsd_deriv.shape == (ddn, nb)), \
                "dsd_deriv input is the wrong shape"
            if fallout_deriv is None:
                fallout_deriv = np.zeros((ddn,))
            assert len(fallout_deriv) == ddn, \
                "fallout_deriv input is wrong length"
            for i in range(ddn):
                idx, num = self.dsd_deriv_loc(self.dsd_deriv_names[i])
                raw[idx:idx+num] = dsd_deriv[i,:] / self.dsd_deriv_scales[i] \
                                    / mc_scale
                idx = self.fallout_deriv_loc(self.dsd_deriv_names[i])
                raw[idx] = fallout_deriv[i] / self.dsd_deriv_scales[i] \
                                    / mc_scale
        else:
            assert dsd_deriv is None or len(dsd_deriv.flat) == 0, \
                "no dsd derivatives should be specified for this descriptor"
            assert fallout_deriv is None or len(fallout_deriv) == 0, \
                "no fallout derivatives should be specified " \
                "for this descriptor"
        if pn > 0:
            if perturb_cov is not None:
                assert (perturb_cov.shape == (pn, pn)), \
                    "perturb_cov input is the wrong shape"
                perturb_cov = perturb_cov.copy()
                for i in range(pn):
                    for j in range(pn):
                        perturb_cov[i,j] /= \
                            self.perturb_scales[i] * self.perturb_scales[j]
            else:
                perturb_cov = 1.e-50 * np.eye(pn)
            idx, _ = self.perturb_chol_loc()
            chol = la.cholesky(perturb_cov, lower=True)
            ic = 0
            for i in range(pn):
                for j in range(i+1):
                    raw[idx+ic] = chol[i,j]
                    ic += 1
        else:
            assert perturb_cov is None, \
                "no perturbation covariance should be specified " \
                "for this descriptor"
        return raw

    def dsd_loc(self, with_fallout=None):
        """Return location of the DSD data in the state vector.

        Arguments:
        with_fallout (optional) - Include fallout at the end of DSD data.
                                  Defaults to False.

        Returns a tuple (idx, num), where idx is the location of the first DSD
        entry and num is the number of entries.
        """
        if with_fallout is None:
            with_fallout = False
        add = 1 if with_fallout else 0
        return (0, self.mass_grid.num_bins + add)

    def fallout_loc(self):
        """Return index of fallout scalar in the state vector."""
        idx, num = self.dsd_loc()
        return idx + num

    def dsd_deriv_loc(self, var_name=None, with_fallout=None):
        """Return location of DSD derivative data in the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.
        with_fallout (optional) - Include fallout derivative at the end of DSD
                                  derivative data. Defaults to False.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.

        Returns a tuple (idx, num), where idx is the location of the first
        entry and num is the number of entries. If all derivative information
        is returned, idx is a list of integers, while num is a scalar that is
        the size of each contiguous block (since all will be the same size).
        """
        if with_fallout is None:
            with_fallout = False
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        st_idx, st_num = self.dsd_loc(with_fallout=True)
        start = st_idx + st_num
        num = nb+1 if with_fallout else nb
        if ddn == 0:
            return [start], 0
        if var_name is None:
            return [start + i*(nb+1) for i in range(ddn)], num
        else:
            idx = self.dsd_deriv_names.index(var_name)
            return start + idx*(nb+1), num

    def fallout_deriv_loc(self, var_name=None):
        """Return location of fallout derivative data in the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.
        """
        idx, num = self.dsd_deriv_loc(var_name, with_fallout=False)
        if var_name is None:
            return [i+num for i in idx]
        else:
            return idx+num

    def perturb_chol_loc(self):
        """Return location of perturbation covariance Cholesky decomposition.

        Returns a tuple (idx, num), where idx is the location of the first
        element and num is the number of elements.
        """
        idx, num = self.dsd_deriv_loc(with_fallout=True)
        pn = self.perturb_num
        return idx[-1] + num, (pn*(pn+1)) // 2

    def dsd_raw(self, raw, with_fallout=None):
        """Return raw DSD data from the state vector.

        Arguments:
        raw - Raw state vector.
        with_fallout (optional) - Include fallout at the end of DSD data, if
                                  present. Defaults to False.
        """
        idx, num = self.dsd_loc(with_fallout)
        return raw[idx:idx+num]

    def fallout_raw(self, raw):
        """Return raw fallout data from the state vector."""
        return raw[self.fallout_loc()]

    def dsd_deriv_raw(self, raw, var_name=None, with_fallout=None):
        """Return raw DSD derivative data from the state vector.

        Arguments:
        raw - Raw state vector.
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 2D array of size
        `(dsd_deriv_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        nb = self.mass_grid.num_bins
        ddn = self.dsd_deriv_num
        idx, num = self.dsd_deriv_loc(var_name, with_fallout)
        if var_name is None:
            output = np.zeros((ddn, num))
            for i in range(ddn):
                output[i,:] = raw[idx[i]:idx[i]+num]
            return output
        else:
            return raw[idx:idx+num]

    def fallout_deriv_raw(self, raw, var_name=None):
        """Return raw fallout derivative data from the state vector.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, information for all derivatives is
        returned. If var_name is provided, information for just that derivative
        is returned.
        """
        ddn = self.dsd_deriv_num
        idx = self.fallout_deriv_loc(var_name)
        if var_name is None:
            output = np.zeros((ddn,))
            for i in range(ddn):
                output[i] = raw[idx[i]]
            return output
        else:
            return raw[idx]

    def perturb_chol_raw(self, raw):
        """Return perturbation covariance Cholesky decomposition from state."""
        idx, _ = self.perturb_chol_loc()
        pn = self.perturb_num
        pc = np.zeros((pn, pn))
        ic = 0
        for i in range(pn):
            for j in range(i+1):
                pc[i,j] = raw[idx+ic]
                ic += 1
        return pc

    def perturb_cov_raw(self, raw):
        """Return raw perturbation covariance matrix from the state vector."""
        pc = self.perturb_chol_raw(raw)
        return pc @ pc.T

    def to_netcdf(self, netcdf_file):
        netcdf_file.write_dimension("dsd_deriv_num", self.dsd_deriv_num)
        netcdf_file.write_dimension("dsd_deriv_name_str_len",
                                    self.dsd_deriv_name_str_len)
        netcdf_file.write_characters("dsd_deriv_names", self.dsd_deriv_names,
            ['dsd_deriv_num', 'dsd_deriv_name_str_len'],
            "Names of variables with respect to which we evolve the "
            "derivative of the drop size distribution")
        netcdf_file.write_array("dsd_deriv_scales", self.dsd_deriv_scales,
            "f8", ["dsd_deriv_num"], "1",
            "Scaling factors used for nondimensionalization of drop size "
            "distribution derivatives")
        pn = self.perturb_num
        netcdf_file.write_dimension("perturb_num", pn)
        if pn == 0:
            return
        netcdf_file.write_array("perturb_wvs", self.perturb_wvs,
            "f8", ["perturb_num", "num_bins"], "1",
            "Weight vectors defining perturbed variables to evolve over time")
        netcdf_file.write_dimension("transform_type_str_len",
                                    Transform.transform_type_str_len)
        transform_types = [t.type_string() for t in self.perturb_transforms]
        transform_params = [t.get_parameters()
                            for t in self.perturb_transforms]
        netcdf_file.write_characters(
            "perturb_transform_types", transform_types,
            ["perturb_num", "transform_type_str_len"],
            "Types of transforms used for perturbed variables")
        max_param_num = 0
        for params in transform_params:
            max_param_num = max(max_param_num, len(params))
        netcdf_file.write_dimension("max_transform_param_num",
                                    max_param_num)
        param_array = np.zeros((pn, max_param_num))
        for i in range(pn):
            params = transform_params[i]
            for j in range(len(params)):
                param_array[i,j] = params[j]
        netcdf_file.write_array("transform_params", param_array,
            "f8", ["perturb_num", "max_transform_param_num"], "1",
            "Parameters for transforms for perturbed variables")
        netcdf_file.write_array("perturb_scales", self.perturb_scales,
            "f8", ["perturb_num"], "1",
            "Scaling factors used for nondimensionalization of perturbed "
            "variables")
        netcdf_file.write_array("perturbation_rate", self.perturbation_rate,
            "f8", ["perturb_num", "perturb_num"], "1",
            "Matrix used to grow perturbation covariance over time")
        netcdf_file.write_scalar("correction_time", self.correction_time,
            "f8", "1",
            "Nondimensionalized time scale for nudging error covariance "
            "toward the manifold to which the true solution is confined")

    @classmethod
    def from_netcdf(cls, netcdf_file, constants, mass_grid):
        dsd_deriv_names = netcdf_file.read_characters('dsd_deriv_names')
        dsd_deriv_scales = netcdf_file.read_array('dsd_deriv_scales')
        pn = netcdf_file.read_dimension("perturb_num")
        if pn == 0:
            return ModelStateDescriptor(constants, mass_grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    scale_inputs=False)
        wvs = netcdf_file.read_array("perturb_wvs")
        transform_types = \
            netcdf_file.read_characters("perturb_transform_types")
        transform_params = netcdf_file.read_array("transform_params")
        transforms = [Transform.from_params(transform_types[i],
                                            transform_params[i,:])
                      for i in range(pn)]
        perturb_scales = netcdf_file.read_array("perturb_scales")
        perturbed_variables = []
        for i in range(pn):
            perturbed_variables.append((wvs[i,:], transforms[i],
                                        perturb_scales[i]))
        perturbation_rate = netcdf_file.read_array("perturbation_rate")
        correction_time = netcdf_file.read_scalar("correction_time")
        return ModelStateDescriptor(constants, mass_grid,
                                    dsd_deriv_names=dsd_deriv_names,
                                    dsd_deriv_scales=dsd_deriv_scales,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time,
                                    scale_inputs=False)


class ModelState:
    """
    Describe a state of the model at a moment in time.

    Initialization arguments:
    desc - The ModelStateDescriptor object corresponding to this object.
    raw - A 1-D vector containing raw state data.

    Attributes:
    constants - ModelConstants object used by this model.
    mass_grid - The grid of the DSD used by this model.
    desc - ModelStateDescriptor associated with this state.
    raw - A single large vector used to handle the state (mainly for use in
          time integration).

    Methods:
    dsd
    dsd_moment
    fallout
    dsd_deriv
    fallout_deriv
    perturb_cov
    dsd_time_deriv_raw
    time_derivative_raw
    linear_func_raw
    linear_func_rate_raw
    zeta_cov_raw
    rain_prod_breakdown
    """
    def __init__(self, desc, raw):
        self.constants = desc.constants
        self.mass_grid = desc.mass_grid
        self.desc = desc
        self.raw = raw

    def dsd(self):
        """Droplet size distribution associated with this state.

        The units are those of M3 for the distribution."""
        return self.desc.dsd_raw(self.raw) * self.constants.mass_conc_scale

    def dsd_moment(self, n, cloud_only=None, rain_only=None):
        """Calculate a moment of the DSD.

        Arguments:
        n - Moment to calculate (can be any real number).
        cloud_only (optional) - Only count cloud-sized drops.
        rain_only (optional) - Only count rain-sized drops.
        """
        const = self.constants
        grid = self.mass_grid
        m3_dsd = self.dsd() / const.std_mass
        weight_vector = grid.moment_weight_vector(n, cloud_only, rain_only)
        return const.std_diameter**(n) * np.dot(weight_vector, m3_dsd)

    def fallout(self):
        """Return amount of third moment that has fallen out of the model."""
        return self.desc.fallout_raw(self.raw) * self.constants.mass_conc_scale

    def dsd_deriv(self, var_name=None):
        """Return derivatives of the DSD with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 2D array of size
        `(dsd_deriv_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        dsd_deriv = self.desc.dsd_deriv_raw(self.raw, var_name).copy()
        if var_name is None:
            for i in range(self.desc.dsd_deriv_num):
                dsd_deriv[i,:] *= self.desc.dsd_deriv_scales[i]
        else:
            idx = self.desc.dsd_deriv_names.index(var_name)
            dsd_deriv *= self.desc.dsd_deriv_scales[idx]
        dsd_deriv *= self.constants.mass_conc_scale
        return dsd_deriv

    def fallout_deriv(self, var_name=None):
        """Return derivative of fallout with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 1D array of length dsd_deriv_num is
        returned, with all derivatives in it. If var_name is provided, a
        single scalar is returned for that derivative.
        """
        if var_name is None:
            output = self.desc.fallout_deriv_raw(self.raw)
            for i in range(self.desc.dsd_deriv_num):
                output[i] *= self.desc.dsd_deriv_scales[i]
            return output * self.constants.mass_conc_scale
        else:
            idx = self.desc.dsd_deriv_names.index(var_name)
            return self.desc.fallout_deriv_raw(self.raw) \
                * self.constants.mass_conc_scale \
                * self.desc.dsd_deriv_scales[idx]

    def perturb_cov(self):
        """Return perturbation covariance matrix."""
        output = self.desc.perturb_cov_raw(self.raw).copy()
        pn = self.desc.perturb_num
        pscales = self.desc.perturb_scales
        for i in range(pn):
            for j in range(pn):
                output[i,j] *= pscales[i] * pscales[j]
        return output

    def dsd_time_deriv_raw(self, proc_tens):
        """Time derivative of the raw dsd using the given process tensors.

        Arguments:
        proc_tens - List of process tensors, the rates of which sum to give the
                    time derivative.
        """
        dsd_raw = self.desc.dsd_raw(self.raw, with_fallout=True)
        dfdt = np.zeros(dsd_raw.shape)
        for pt in proc_tens:
            dfdt += pt.calc_rate(dsd_raw, out_flux=True)
        return dfdt

    def time_derivative_raw(self, proc_tens):
        """Time derivative of the state using the given process tensors.

        Arguments:
        proc_tens - List of process tensors, the rates of which sum to give the
                    time derivative.
        """
        desc = self.desc
        nb = self.mass_grid.num_bins
        ddn = desc.dsd_deriv_num
        pn = desc.perturb_num
        dfdt = np.zeros((len(self.raw),))
        dsd_raw = desc.dsd_raw(self.raw)
        dsd_deriv_raw = desc.dsd_deriv_raw(self.raw, with_fallout=True)
        didx, dnum = desc.dsd_loc(with_fallout=True)
        dridxs, drnum = desc.dsd_deriv_loc(with_fallout=True)
        if pn > 0:
            double_time_deriv = np.zeros((nb+1))
        for pt in proc_tens:
            if ddn > 0:
                rate, derivative = pt.calc_rate(dsd_raw, out_flux=True,
                                                derivative=True)
                dfdt[didx:didx+dnum] += rate
                for i in range(ddn):
                    dfdt[dridxs[i]:dridxs[i]+drnum] += \
                        derivative @ dsd_deriv_raw[i,:]
                if pn > 0:
                    double_time_deriv += derivative @ rate
            else:
                dfdt[didx:didx+dnum] += pt.calc_rate(dsd_raw, out_flux=True)
        if pn > 0:
            ddsddt = desc.dsd_raw(dfdt)
            ddsddt_deriv = np.zeros((ddn+1,nb))
            ddsddt_deriv[0,:] = double_time_deriv[:nb]
            ddsddt_deriv[1:,:] = desc.dsd_deriv_raw(dfdt)
            perturb_cov_raw = desc.perturb_cov_raw(self.raw)
            lfs = np.zeros((pn,))
            lf_jac = np.zeros((pn, ddn+1))
            lf_rates = np.zeros((pn,))
            lf_rate_jac = np.zeros((pn, ddn+1))
            for i in range(pn):
                wv = self.desc.perturb_wvs[i]
                lfs[i], lf_jac[i,:] = \
                    self.linear_func_raw(wv, derivative=True,
                                         dfdt=ddsddt)
                lf_rates[i], lf_rate_jac[i,:] = \
                    self.linear_func_rate_raw(wv, ddsddt,
                                              dfdt_deriv=ddsddt_deriv)
            transform_mat = np.zeros((pn, pn))
            transform_mat2 = np.zeros((pn,))
            for i in range(pn):
                transform = desc.perturb_transforms[i]
                transform_mat[i,i] = transform.derivative(lfs[i])
                transform_mat2[i] = \
                    transform.second_over_first_derivative(lfs[i])
            zeta_to_v = transform_mat @ lf_jac
            jacobian = transform_mat @ lf_rate_jac @ la.pinv(zeta_to_v)
            jacobian += np.diag(lf_rates * transform_mat2)
            if self.desc.correction_time is None:
                perturb_cov_projected = perturb_cov_raw
            else:
                error_cov_inv = la.inv(desc.perturbation_rate)
                projection = la.inv(zeta_to_v.T @ error_cov_inv
                                        @ zeta_to_v)
                projection = zeta_to_v @ projection @ zeta_to_v.T \
                                @ error_cov_inv
                perturb_cov_projected = projection @ perturb_cov_raw \
                                            @ projection.T
            cov_rate = jacobian @ perturb_cov_projected
            cov_rate += cov_rate.T
            cov_rate += desc.perturbation_rate
            if self.desc.correction_time is not None:
                cov_rate += (perturb_cov_projected - perturb_cov_raw) \
                                / self.desc.correction_time
            # Convert this rate to a rate on the Cholesky decomposition.
            pcidx, pcnum = desc.perturb_chol_loc()
            perturb_chol = desc.perturb_chol_raw(self.raw)
            chol_rate = la.solve(perturb_chol, cov_rate)
            chol_rate = np.transpose(la.solve(perturb_chol, chol_rate.T))
            for i in range(pn):
                chol_rate[i,i] = 0.5 * chol_rate[i,i]
                for j in range(i+1, pn):
                    chol_rate[i,j] = 0.
            chol_rate = perturb_chol @ chol_rate
            chol_rate_flat = np.zeros((pcnum,))
            ic = 0
            for i in range(pn):
                for j in range(i+1):
                    chol_rate_flat[ic] = chol_rate[i,j]
                    ic += 1
            dfdt[pcidx:pcidx+pcnum] = chol_rate_flat
        return dfdt

    def linear_func_raw(self, weight_vector, derivative=None, dfdt=None):
        """Calculate a linear functional of the DSD (nondimensionalized units).

        Arguments:
        weight_vector - Weighting function defining the integral over the DSD.
                        (E.g. obtained from `MassGrid.moment_weight_vector`.)
        derivative (optional) - If True, return derivative information as well.
                                Defaults to False.
        dfdt (optional) - The raw derivative of the DSD with respect to time.
                          If included and derivative is True, the time
                          derivative of the functional will be prepended to the
                          derivative output.

        If derivative is False, this returns a scalar representing the
        nondimensional moment or other linear functional requested. If
        derivative is True, returns a tuple where the first element is the
        requested functional and the second element is the gradient of the
        functional with to the variables listed in self.desc.

        If dfdt is specified and derivative is True, the 0-th element of the
        gradient will be the derivative with respect to time.
        """
        if derivative is None:
            derivative = False
        dsd_raw = self.desc.dsd_raw(self.raw)
        nb = self.mass_grid.num_bins
        if derivative:
            ddn = self.desc.dsd_deriv_num
            offset = 1 if dfdt is not None else 0
            dsd_deriv_raw = np.zeros((ddn+offset, nb))
            if dfdt is not None:
                dsd_deriv_raw[0,:] = dfdt
            dsd_deriv_raw[offset:,:] = self.desc.dsd_deriv_raw(self.raw)
            grad = dsd_deriv_raw @ weight_vector
            return np.dot(dsd_raw, weight_vector), grad
        else:
            return np.dot(dsd_raw, weight_vector)

    def linear_func_rate_raw(self, weight_vector, dfdt, dfdt_deriv=None):
        """Calculate rate of change of a linear functional of the DSD.

        Arguments:
        weight_vector - Weighting function defining the integral over the DSD.
                        (E.g. obtained from `MassGrid.moment_weight_vector`.)
        dfdt - The raw derivative of the DSD with respect to time.
        dfdt_deriv (optional) - If not None, return derivative information as
                                well. Defaults to None.

        If dfdt_deriv is None, this returns a scalar representing the time
        derivative of the nondimensional moment or other linear functional
        requested. If dfdt_deriv contains DSD derivative information, returns a
        tuple where the first element is the requested time derivative and the
        second element is the gradient of the derivative with respect to the
        variables for which the DSD derivative is provided.

        If dfdt_deriv is not None, it should be an array of shape
            (ddn, num_bins)
        where ddn is the number of derivatives that will be returned in the
        second argument.
        """
        if dfdt_deriv is None:
            return np.dot(dfdt, weight_vector)
        else:
            return np.dot(dfdt, weight_vector), dfdt_deriv @ weight_vector

    def zeta_cov_raw(self, ddsddt):
        """Find the raw error covariance of dsd_deriv variables and time.

        Arguments:
        ddsddt - Time derivative of raw DSD, e.g. the first num_bins elements
                 of dsd_time_deriv_raw.
        """
        desc = self.desc
        ddn = desc.dsd_deriv_num
        pn = desc.perturb_num
        lfs = np.zeros((pn,))
        lf_jac = np.zeros((pn, ddn+1))
        for i in range(pn):
            wv = desc.perturb_wvs[i,:]
            lfs[i], lf_jac[i,:] = self.linear_func_raw(wv, derivative=True,
                                                       dfdt=ddsddt)
        transform_mat = np.diag([desc.perturb_transforms[i].derivative(lfs[i])
                                 for i in range(pn)])
        v_to_zeta = la.pinv(transform_mat @ lf_jac)
        # We are assuming here that perturb_cov does not need the "correction"
        # for pn > ddn + 1.
        perturb_cov_raw = desc.perturb_cov_raw(self.raw)
        return v_to_zeta @ perturb_cov_raw @ v_to_zeta.T

    def rain_prod_breakdown(self, ktens, cloud_vector, derivative=None):
        """Calculate autoconversion and accretion rates.

        Arguments:
        ktens - Collision KernelTensor.
        cloud_vector - A vector of values between 0 and 1, representing the
                       percentage of mass in a bin that should be considered
                       cloud rather than rain.
        derivative (optional) - If True, returns Jacobian information.
                                Defaults to False.

        If derivative is False, the return value is an array of length 2
        containing only the autoconversion and accretion rates. If derivative
        is True, the return value is a tuple, with the first entry containing
        the process rates and the second entry containing the Jacobian of those
        rates with respect to time and the dsd_deriv variables listed in desc.
        """
        if derivative is None:
            derivative = False
        rate_scale = self.constants.mass_conc_scale / self.constants.time_scale
        grid = self.mass_grid
        nb = grid.num_bins
        m3_vector = grid.moment_weight_vector(3)
        dsd_raw = self.desc.dsd_raw(self.raw)
        total_inter = ktens.calc_rate(dsd_raw, out_flux=True,
                                      derivative=derivative)
        if derivative:
            save_deriv = total_inter[1]
            total_inter = total_inter[0]
            ddn = self.desc.dsd_deriv_num
            dsd_deriv_raw = np.zeros((ddn+1, nb+1))
            dsd_deriv_raw[0,:] = total_inter
            dsd_deriv_raw[1:,:] = self.desc.dsd_deriv_raw(self.raw,
                                                          with_fallout=True)
            total_deriv = save_deriv @ dsd_deriv_raw.T
        cloud_dsd_raw = dsd_raw * cloud_vector
        cloud_inter = ktens.calc_rate(cloud_dsd_raw, out_flux=True,
                                      derivative=derivative)
        if derivative:
            cloud_dsd_deriv = np.transpose(dsd_deriv_raw).copy()
            for i in range(3):
                cloud_dsd_deriv[:nb,i] *= cloud_vector
                cloud_dsd_deriv[nb,i] = 0.
            cloud_deriv = cloud_inter[1] @ cloud_dsd_deriv
            cloud_inter = cloud_inter[0]
        rain_vector = 1. - cloud_vector
        auto = np.dot(cloud_inter[:nb]*rain_vector, m3_vector) \
            + cloud_inter[nb]
        auto *= rate_scale
        no_csc_or_auto = total_inter - cloud_inter
        accr = np.dot(-no_csc_or_auto[:nb]*cloud_vector, m3_vector)
        accr *= rate_scale
        rates = np.array([auto, accr])
        if derivative:
            rate_deriv = np.zeros((2, ddn+1))
            rate_deriv[0,:] = (m3_vector * rain_vector) @ cloud_deriv[:nb,:] \
                + cloud_deriv[nb,:]
            no_soa_deriv = total_deriv - cloud_deriv
            rate_deriv[1,:] = -(m3_vector * cloud_vector) @ no_soa_deriv[:nb,:]
            rate_deriv *= rate_scale
            rate_deriv[:,0] /= self.constants.time_scale
            for i in range(ddn):
                rate_deriv[:,1+i] *= self.desc.dsd_deriv_scales[i]
            return rates, rate_deriv
        else:
            return rates


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

    def read_kernel_tensor(self, kernel, grid):
        """Read a KernelTensor object from a netCDF file.

        Arguments:
        kernel - Kernel object to use in constructing the KernelTensor.
        grid - MassGrid object to use in constructing the MassGrid.
        """
        return KernelTensor.from_netcdf(self, kernel, grid)

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
        ktens = self.read_kernel_tensor(kernel, grid)
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
