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

"""Bin model state descriptor for JEFE."""

import numpy as np
import scipy.linalg as la

from bin_model.transform import Transform

max_variable_name_len = 64
"""Maximum length of variable name strings on file."""

class DerivativeVar:
    """
    Describe a variable that we track the DSD derivative with respect to.

    Initialization arguments:
    name - Variable name.
    scale - Used for nondimensionalization of derivatives.
    """

    def __init__(self, name, scale=1.):
        if len(name) > max_variable_name_len:
            bad_len = len(name)
            raise ValueError(f"derivative variable name too long: {name}"
                             f" (length was {bad_len}, max is"
                             f" {max_variable_name_len})")
        self.name = name
        self.scale = scale

    def matches(self, name):
        """Check whether input name matches the name of this variable."""
        return name == self.name

    def si_to_nondimensional(self, derivative):
        """Convert SI unit derivative to nondimensional representation."""
        return derivative * self.scale

    def nondimensional_to_si(self, derivative):
        """Convert unitless derivative to SI units."""
        return derivative / self.scale

class PerturbedVar:
    """
    Describe a variable that we are tracking a perturbation to.

    Currently, perturbed variables must be expressible as a function of a single
    linear functional of the drop size distribution (e.g. moments or simple
    transformations of moments).

    Initialization arguments:
    name - Variable name.
    weight_vector - Vector used to convert drop size distribution to a linear
                    functional.
    transform - Transform object used to convert linear functional to the
                desired variable.
    scale - Scale used to nondimensionalize variable.
    """

    def __init__(self, name, weight_vector, transform, scale):
        if len(name) > max_variable_name_len:
            bad_len = len(name)
            raise ValueError(f"perturbed variable name too long: {name}"
                             f" (length was {bad_len}, max is"
                             f" {max_variable_name_len})")
        self.name = name
        self.weight_vector = weight_vector
        self.transform = transform
        self.scale = scale


class ModelStateDescriptor:
    """
    Describe the state variables contained in a ModelState.

    Initialization arguments:
    constants - A ModelConstants object.
    mass_grid - A MassGrid object defining the bins.
    deriv_vars (optional) - List of DerivativeVars describing variables with
                            respect to which the state derivatives are
                            prognosed.
    perturbed_variables (optional) - A list of tuples, with each tuple
        containing a weight vector, a transform, and a scale, in that order.
    perturbation_rate (optional) - A covariance matrix representing the error
        introduced to the perturbed variables per second.
    correction_time (optional) - Time scale over which the error covariance is
        nudged toward a corrected value.
    scale_inputs (optional) - Whether to scale the input variables. Default is
                              True. Setting this to False is mainly intended
                              for testing and I/O utility code.
    """

    small_error_variance = 1.e-50
    """Used to initialize variables starting with near-zero error covariance."""

    def __init__(self, constants, mass_grid,
                 deriv_vars=None,
                 perturbed_variables=None, perturbation_rate=None,
                 correction_time=None, scale_inputs=None):
        if scale_inputs is None:
            scale_inputs = True
        self.constants = constants
        self.mass_grid = mass_grid
        if deriv_vars is not None:
            self.deriv_var_num = len(deriv_vars)
            num_unique_names = len(set(dvar.name for dvar in deriv_vars))
            if num_unique_names != self.deriv_var_num:
                raise ValueError("duplicate derivative variable names in state"
                                 " descriptor")
            self.deriv_vars = deriv_vars
        else:
            self.deriv_var_num = 0
            self.deriv_vars = []
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
                if perturbation_rate.shape != (pn, pn):
                    raise ValueError("perturbation_rate is shape"
                                     f" {perturbation_rate.shape} but should be"
                                     f" ({pn}, {pn})")
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
                assert pn == self.deriv_var_num + 1, \
                    "must specify correction time unless perturb_num is " \
                    "equal to deriv_var_num+1"
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

    def find_deriv_var_index(self, name):
        """Return index of derivative variable corresponding to the given name.

        This function returns the index of the variable in self.deriv_vars (i.e.
        0 for the first variable, 1 for the second...), not information about
        the location of the variable in state data.
        """
        for i, dvar in enumerate(self.deriv_vars):
            if dvar.matches(name):
                return i
        raise ValueError(f"variable '{name}' is not a derivative variable"
                         " listed in the state descriptor")

    def find_deriv_var(self, name):
        """Return derivative variable corresponding to the given name."""
        return self.deriv_vars[self.find_deriv_var_index(name)]

    def construct_raw(self, dsd, fallout=None, dsd_deriv=None,
                      fallout_deriv=None, perturb_cov=None):
        """Construct raw state vector from individual variables.

        Arguments:
        dsd - Drop size distribution.
        fallout (optional) - Amount of third moment that has fallen out of the
                             model. If not specified, defaults to zero.
        dsd_deriv - DSD derivatives. Mandatory if deriv_var_num is not zero.
        fallout_deriv - Fallout derivatives. If not specified, defaults to
                        zero.
        perturb_cov - Covariance matrix for Gaussian perturbation. If not
                      specified, defaults to zero.

        Returns a 1-D array of size given by state_len().
        """
        raw = np.zeros((self.state_len(),))
        nb = self.mass_grid.num_bins
        dvn = self.deriv_var_num
        pn = self.perturb_num
        mc_scale = self.constants.mass_conc_scale
        if len(dsd) != nb:
            in_size = len(dsd)
            raise ValueError(f"input dsd is size {in_size} but the descriptor"
                             f" grid size is {nb}")
        idx, num = self.dsd_loc()
        raw[idx:idx+num] = dsd / self.constants.mass_conc_scale
        if fallout is None:
            fallout = 0.
        raw[self.fallout_loc()] = fallout / mc_scale
        if dvn > 0:
            if dsd_deriv is None:
                raise RuntimeError("dsd_deriv input is required, but missing")
            if dsd_deriv.shape != (dvn, nb):
                raise ValueError(f"dsd_deriv input is shape {dsd_deriv.shape}"
                                 f" but expected shape ({dvn}, {nb}) array")
            if fallout_deriv is None:
                fallout_deriv = np.zeros((dvn,))
            if len(fallout_deriv.flat) != dvn:
                raise ValueError("fallout_deriv input is shape"
                                 f" {fallout_deriv.shape}, but expected size"
                                 f" {dvn} array")
            for i, dvar in enumerate(self.deriv_vars):
                idx, num = self.dsd_deriv_loc(dvar.name)
                raw[idx:idx+num] = \
                    dvar.si_to_nondimensional(dsd_deriv[i,:]) / mc_scale
                idx = self.fallout_deriv_loc(self.deriv_vars[i].name)
                raw[idx] = \
                    dvar.si_to_nondimensional(fallout_deriv[i]) / mc_scale
        else:
            if dsd_deriv is not None and len(dsd_deriv.flat) != 0:
                raise RuntimeError("no dsd derivatives should be specified for"
                                   " this descriptor")
            if fallout_deriv is not None and len(fallout_deriv) != 0:
                raise RuntimeError("no fallout derivatives should be specified"
                                   " for this descriptor")
        if pn > 0:
            if perturb_cov is not None:
                if perturb_cov.shape != (pn, pn):
                    raise ValueError("perturb_cov input is shape"
                                     f" {perturb_cov.shape} but expected shape "
                                     f"({pn}, {pn}) array")
                perturb_cov = perturb_cov.copy()
                for i in range(pn):
                    for j in range(pn):
                        perturb_cov[i,j] /= \
                            self.perturb_scales[i] * self.perturb_scales[j]
            else:
                perturb_cov = self.small_error_variance * np.eye(pn)
            idx, _ = self.perturb_chol_loc()
            chol = la.cholesky(perturb_cov, lower=True)
            ic = 0
            for i in range(pn):
                for j in range(i+1):
                    raw[idx+ic] = chol[i,j]
                    ic += 1
        else:
            if perturb_cov is not None:
                raise RuntimeError("no perturbation covariance should be "
                                   "specified for this descriptor")
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
        dvn = self.deriv_var_num
        st_idx, st_num = self.dsd_loc(with_fallout=True)
        start = st_idx + st_num
        num = nb+1 if with_fallout else nb
        if dvn == 0:
            return [start], 0
        if var_name is None:
            return [start + i*(nb+1) for i in range(dvn)], num
        else:
            idx = self.find_deriv_var_index(var_name)
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
        `(deriv_var_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        nb = self.mass_grid.num_bins
        dvn = self.deriv_var_num
        idx, num = self.dsd_deriv_loc(var_name, with_fallout)
        if var_name is None:
            output = np.zeros((dvn, num))
            for i in range(dvn):
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
        dvn = self.deriv_var_num
        idx = self.fallout_deriv_loc(var_name)
        if var_name is None:
            output = np.zeros((dvn,))
            for i in range(dvn):
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
        netcdf_file.write_dimension("deriv_var_num", self.deriv_var_num)
        netcdf_file.write_dimension("variable_name_length",
                                    max_variable_name_len)
        netcdf_file.write_characters("deriv_var_names",
            [dvar.name for dvar in self.deriv_vars],
            ['deriv_var_num', 'variable_name_length'],
            "Names of variables with respect to which we evolve the "
            "derivative of the state")
        netcdf_file.write_array("deriv_var_scales",
            np.array([dvar.scale for dvar in self.deriv_vars]),
            "f8", ["deriv_var_num"], "1",
            "Scaling factors used for nondimensionalization of derivatives")
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
        deriv_var_names = netcdf_file.read_characters('deriv_var_names')
        deriv_var_scales = netcdf_file.read_array('deriv_var_scales')
        deriv_vars = [DerivativeVar(name, scale)
                      for name, scale in zip(deriv_var_names, deriv_var_scales)]
        pn = netcdf_file.read_dimension("perturb_num")
        if pn == 0:
            return ModelStateDescriptor(constants, mass_grid,
                                        deriv_vars=deriv_vars,
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
                                    deriv_vars=deriv_vars,
                                    perturbed_variables=perturbed_variables,
                                    perturbation_rate=perturbation_rate,
                                    correction_time=correction_time,
                                    scale_inputs=False)
