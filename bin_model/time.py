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

    def integrate_raw(self, t_len, state, proc_tens, perturb=None):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        raise NotImplementedError

    def integrate(self, t_len, state, proc_tens, perturb=None):
        """Integrate the state and return an Experiment.

        Arguments:
        t_len - Length of time to integrate over (seconds).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.

        Returns an Experiment object summarizing the integration and all inputs
        to it.
        """
        tscale = self.constants.time_scale
        desc = state.desc
        times, raws = self.integrate_raw(t_len / tscale, state, proc_tens,
                                         perturb)
        times = times * tscale
        if state.desc.perturbed_num > 0:
            for i in range(raws.shape[0]):
                pc = state.desc.perturb_cov_raw(raws[i,:])
                if np.any(la.eigvalsh(pc) <= 0.):
                    raise RuntimeError("nonpositive covariance occurred at: " \
                                       + str(times[i]))
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
        elif integrator_type == "ForwardEuler":
            dt = netcdf_file.read_scalar("dt")
            return ForwardEulerIntegrator(constants, dt)
        elif integrator_type == "RK4":
            dt = netcdf_file.read_scalar("dt")
            return RK4Integrator(constants, dt)
        raise RuntimeError(f"integrator_type '{integrator_type}' on file not"
                           " recognized")

    def _get_times_util(self, t_len, dt):
        """Get array of times given time length and time step."""
        tol = dt * 1.e-10
        num_step = int(t_len / dt)
        if t_len - (num_step * dt) > tol:
            num_step = num_step + 1
        return np.linspace(0., t_len, num_step+1)


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

    def integrate_raw(self, t_len, state, proc_tens, perturb=None):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        raw_len = len(state.raw)
        rate_fun = lambda t, raw: \
            ModelState(state.desc, raw).time_derivative_raw(proc_tens, perturb)
        nb = state.mass_grid.num_bins
        atol = 1.e-6 * np.ones(raw_len)
        dsd_idx, _ = state.desc.dsd_loc()
        atol[dsd_idx:dsd_idx+nb] = 1.e-6 * state.mass_grid.bin_widths
        pn = state.desc.perturbed_num
        pcidx, _ = state.desc.perturb_chol_loc()
        for i in range(pn):
            offset = (((i+1)*(i+2)) // 2) - 1
            atol[pcidx+offset] = 1.e-300
        solbunch = solve_ivp(rate_fun, (times[0], times[-1]), state.raw,
                             method='RK45', t_eval=times, max_step=self.dt_raw,
                             atol=atol)
        if solbunch.status != 0:
            raise RuntimeError("integration failed: " + solbunch.message)
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


class ForwardEulerIntegrator(Integrator):
    """
    Integrate a model state in time using the Forward Euler method.

    Initialization arguments:
    constants - The ModelConstants object.
    dt - Max time step at which to calculate the results.
    """
    def __init__(self, constants, dt):
        self.constants = constants
        self.dt = dt
        self.dt_raw = dt / constants.time_scale

    def integrate_raw(self, t_len, state, proc_tens, perturb=None):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        num_step = len(times) - 1
        raw_len = len(state.raw)
        output = np.zeros((num_step+1, raw_len))
        output[0,:] = state.raw
        for i in range(num_step):
            step_state = ModelState(state.desc, output[i,:])
            output[i+1,:] = output[i,:] + \
                dt * step_state.time_derivative_raw(proc_tens, perturb)
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "ForwardEuler",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")


class RK4Integrator(Integrator):
    """
    Integrate a model state in time using the traditional RK4 method.

    Initialization arguments:
    constants - The ModelConstants object.
    dt - Max time step at which to calculate the results.
    """
    def __init__(self, constants, dt):
        self.constants = constants
        self.dt = dt
        self.dt_raw = dt / constants.time_scale

    def integrate_raw(self, t_len, state, proc_tens, perturb=None):
        """Integrate the state and return raw state data.

        Arguments:
        t_len - Length of time to integrate over (nondimensionalized units).
        state - Initial state.
        proc_tens - List of process tensors to calculate state process rates
                    each time step.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.

        Returns a tuple `(times, raws)`, where times is an array of times at
        which the output is provided, and raws is an array for which each row
        is the raw state vector at a different time.
        """
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        num_step = len(times) - 1
        raw_len = len(state.raw)
        output = np.zeros((num_step+1, raw_len))
        output[0,:] = state.raw
        num_stages = 4
        for i in range(num_step):
            slopes = np.zeros((num_stages, raw_len))
            add_next = np.zeros((raw_len,))
            coefs = [0.5, 0.5, 1., 0.]
            for j in range(num_stages):
                stage_state = ModelState(state.desc, output[i,:]+add_next)
                slopes[j,:] = stage_state.time_derivative_raw(proc_tens,
                                                              perturb)
                add_next = coefs[j] * dt * slopes[j,:]
            weights = (dt / 6.) * np.array([1., 2., 2., 1.])
            output[i+1,:] = output[i,:] + weights @ slopes
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "RK4",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")
