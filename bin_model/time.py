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

"""Classes used for bin model time integration."""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.optimize import root

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
        dt = netcdf_file.read_scalar("dt")
        if integrator_type == "RK45":
            return RK45Integrator(constants, dt)
        elif integrator_type == "ForwardEuler":
            return ForwardEulerIntegrator(constants, dt)
        elif integrator_type == "RK4":
            return RK4Integrator(constants, dt)
        elif integrator_type == "Radau":
            return RadauIntegrator(constants, dt)
        elif integrator_type == "BackwardEuler":
            return BackwardEulerIntegrator(constants, dt)
        elif integrator_type == "DIRK2":
            return Dirk2Integrator(constants, dt)
        raise RuntimeError(f"integrator_type '{integrator_type}' on file not"
                           " recognized")

    def _get_times_util(self, t_len, dt):
        """Get array of times given time length and time step."""
        tol = dt * 1.e-10
        num_step = int(t_len / dt)
        if t_len - (num_step * dt) > tol:
            num_step = num_step + 1
        return np.linspace(0., t_len, num_step+1)

    # Issue to be cleaned up: If calc_deriv is to be called here, it should
    # arguably have the leading underscore removed and be placed under unit test
    # directly. Arguably this whole functionality should just be moved to
    # CollisionTensor or State.
    def _get_jac_util(self, state, proc_tens):
        """Calculate Jacobian of the time derivative of a state.

        Does not currently support states with any DerivativeVar.
        """
        nb = state.mass_grid.num_bins
        bw = state.mass_grid.bin_widths
        raw_len = len(state.raw)
        f = state.desc.dsd_raw(state.raw)[:nb] / bw
        f_shaped = np.reshape(f[:nb], (nb, 1))
        output = np.zeros((raw_len, raw_len))
        for pt in proc_tens:
            deriv = pt._calc_deriv(f_shaped, out_flux=True, out_len=nb+1)
            for i in range(nb):
                deriv[:,i] /= bw[i]
            output += deriv
        return output


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


class RadauIntegrator(Integrator):
    """
    Integrate a model state in time using the SciPy Radau implementation.

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

        Note that this class does not support the linear tangent model
        functionality that the explicit integrators allow, so the number of
        derivative variables and perturbed moment variables must both be zero.
        """
        if state.desc.deriv_var_num > 0:
            raise RuntimeError("implicit integrator not compatible with linear"
                               " tangent functionality")
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        raw_len = len(state.raw)
        nb = state.mass_grid.num_bins
        rate_fun = lambda t, raw: \
            ModelState(state.desc, raw).time_derivative_raw(proc_tens, perturb)
        rate_jac = lambda t, raw: \
            self._get_jac_util(ModelState(state.desc, raw), proc_tens)
        atol = 1.e-6 * np.ones(raw_len)
        dsd_idx, _ = state.desc.dsd_loc()
        atol[dsd_idx:dsd_idx+nb] = 1.e-6 * state.mass_grid.bin_widths
        pn = state.desc.perturbed_num
        pcidx, _ = state.desc.perturb_chol_loc()
        for i in range(pn):
            offset = (((i+1)*(i+2)) // 2) - 1
            atol[pcidx+offset] = 1.e-300
        solbunch = solve_ivp(rate_fun, (times[0], times[-1]), state.raw,
                             method='Radau', t_eval=times, max_step=self.dt_raw,
                             atol=atol, jac=rate_jac)
        if solbunch.status != 0:
            raise RuntimeError("integration failed: " + solbunch.message)
        output = np.transpose(solbunch.y)
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "Radau",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")


class BackwardEulerIntegrator(Integrator):
    """
    Integrate a model state in time using the Backward Euler method.

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
        if state.desc.deriv_var_num > 0:
            raise RuntimeError("implicit integrator not compatible with linear"
                               " tangent functionality")
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        num_step = len(times) - 1
        raw_len = len(state.raw)
        output = np.zeros((num_step+1, raw_len))
        output[0,:] = state.raw
        for i in range(num_step):
            step_state = ModelState(state.desc, output[i,:])
            def residual_f(raw):
                new_state = ModelState(state.desc, raw)
                tend = new_state.time_derivative_raw(proc_tens, perturb)
                return new_state.raw - step_state.raw - dt * tend
            def residual_j(raw):
                new_state = ModelState(state.desc, raw)
                jac = self._get_jac_util(new_state, proc_tens)
                return np.eye(len(raw)) - dt * jac
            scale = np.ones(raw_len)
            scale[:state.mass_grid.num_bins] = 1./state.mass_grid.bin_widths
            sol = root(residual_f, step_state.raw, jac=residual_j, tol=1.e-6,
                       options={'diag': scale})
            if not sol.success:
                raise RuntimeError("integration failed: " + sol.message)
            output[i+1,:] = sol.x
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "BackwardEuler",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")


class Dirk2Integrator(Integrator):
    """
    Integrate a model state in time using a second-order DIRK method.

    The method used is a second-order Diagonally Implicit Runge-Kutta (DIRK)
    method with Butcher tableau:

    1 |    1   0
    0 |   -1   1
    ------------
      |  1/2 1/2

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
        if state.desc.deriv_var_num > 0:
            raise RuntimeError("implicit integrator not compatible with linear"
                               " tangent functionality")
        dt = self.dt_raw
        times = self._get_times_util(t_len, dt)
        num_step = len(times) - 1
        raw_len = len(state.raw)
        output = np.zeros((num_step+1, raw_len))
        output[0,:] = state.raw
        scale = np.ones(raw_len)
        scale[:state.mass_grid.num_bins] = 1./state.mass_grid.bin_widths
        for i in range(num_step):
            step_state = ModelState(state.desc, output[i,:])
            def residual_f(raw):
                new_state = ModelState(state.desc, raw)
                tend = new_state.time_derivative_raw(proc_tens, perturb)
                return new_state.raw - step_state.raw - dt * tend
            def residual_j(raw):
                new_state = ModelState(state.desc, raw)
                jac = self._get_jac_util(new_state, proc_tens)
                return np.eye(len(raw)) - dt * jac
            sol = root(residual_f, step_state.raw, jac=residual_j, tol=1.e-6,
                       options={'diag': scale})
            if not sol.success:
                raise RuntimeError("integration failed: " + sol.message)
            be = sol.x
            be_tend = ModelState(state.desc, sol.x) \
                .time_derivative_raw(proc_tens, perturb)
            def residual_f2(raw):
                new_state = ModelState(state.desc, raw)
                tend = new_state.time_derivative_raw(proc_tens, perturb)
                return new_state.raw - step_state.raw - dt * (tend - be_tend)
            sol = root(residual_f2, step_state.raw, jac=residual_j, tol=1.e-6,
                       options={'diag': scale})
            if not sol.success:
                raise RuntimeError("integration failed: " + sol.message)
            stage2_tend = ModelState(state.desc, sol.x) \
                .time_derivative_raw(proc_tens, perturb)
            output[i+1,:] = output[i,:] + 0.5 * dt * (be_tend + stage2_tend)
        return times, output

    def to_netcdf(self, netcdf_file):
        """Write Integrator to netCDF file."""
        netcdf_file.write_dimension("integrator_type_str_len",
                                    self.integrator_type_str_len)
        netcdf_file.write_characters("integrator_type", "DIRK2",
                                     "integrator_type_str_len",
                                     "Type of time integration used")
        netcdf_file.write_scalar("dt", self.dt,
                                 "f8", "s",
                                 "Maximum time step used by integrator")
