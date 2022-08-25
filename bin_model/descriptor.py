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
