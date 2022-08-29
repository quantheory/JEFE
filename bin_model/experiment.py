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

"""Class representing an experiment using the bin model."""

import numpy as np

from bin_model.state import ModelState


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
               Shape is `(num_time_steps, deriv_var_num+1, deriv_var_num+1)`.
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
            deriv = np.zeros((lf_num, self.desc.deriv_var_num+1))
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
            netcdf_file.write_dimension("deriv_var_num+1",
                                        self.desc.deriv_var_num+1)
            netcdf_file.write_array("zeta_cov", self.zeta_cov,
                "f8", ['time', 'deriv_var_num+1', 'deriv_var_num+1'], "1",
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
