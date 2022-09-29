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

"""Classes used for bin model time integration."""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

from bin_model.experiment import Experiment
from bin_model.state import ModelState


class Integrator:
    """
    Integrate a model state in time.
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
        dvn = desc.deriv_var_num
        if dvn > 0:
            nb = desc.mass_grid.num_bins
            num_step = len(times) - 1
            ddsddt = np.zeros((num_step+1, nb))
            states = [ModelState(desc, raws[i,:]) for i in range(num_step+1)]
            for i in range(num_step+1):
                ddsddt[i,:] = states[i].dsd_time_deriv_raw(proc_tens)[:nb]
            pn = desc.perturbed_num
            if pn > 0:
                zeta_cov = np.zeros((num_step+1, dvn+1, dvn+1))
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
        raise RuntimeError(f"integrator_type '{integrator_type}' on file not"
                           " recognized")


class RK45Integrator(Integrator):
    """
    Integrate a model state in time using the SciPy RK45 implementation.

    Initialization arguments:
    constants - The ModelConstants object.
    dt - Max time step at which to calculate the results.
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
        pn = state.desc.perturbed_num
        pcidx, _ = state.desc.perturb_chol_loc()
        for i in range(pn):
            offset = (((i+1)*(i+2)) // 2) - 1
            atol[pcidx+offset] = 1.e-300
        solbunch = solve_ivp(rate_fun, (times[0], times[-1]), state.raw,
                             method='RK45', t_eval=times, max_step=self.dt,
                             atol=atol)
        if solbunch.status != 0:
            raise RuntimeError("integration failed: " + solbunch.message)
        if state.desc.perturbed_num > 0:
            for i in range(num_step+1):
                pc = state.desc.perturb_cov_raw(solbunch.y[:,i])
                if np.any(la.eigvalsh(pc) < 0.):
                    raise RuntimeError("negative covariance occurred at: " \
                                       + str(solbunch.t[i]))
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
