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

"""State for bin model used by JEFE."""

import numpy as np
import scipy.linalg as la


class ModelState:
    """
    Describe a state of the model at a moment in time.

    Initialization arguments:
    desc - The ModelStateDescriptor object corresponding to this object.
    raw - A 1-D vector containing raw state data.
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
        return const.diameter_scale**(n) * np.dot(weight_vector, m3_dsd)

    def fallout(self):
        """Return amount of third moment that has fallen out of the model."""
        return self.desc.fallout_raw(self.raw) * self.constants.mass_conc_scale

    def dsd_deriv(self, var_name=None):
        """Return derivatives of the DSD with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 2D array of size
        `(deriv_var_num, num_bins)` is returned, with all derivatives in it.
        If var_name is provided, a 1D array of size num_bins is returned.
        """
        dsd_deriv = self.desc.dsd_deriv_raw(self.raw, var_name).copy()
        if var_name is None:
            for i, dvar in enumerate(self.desc.deriv_vars):
                dsd_deriv[i,:] = dvar.nondimensional_to_si(dsd_deriv[i,:])
        else:
            dvar = self.desc.find_deriv_var(var_name)
            dsd_deriv = dvar.nondimensional_to_si(dsd_deriv)
        dsd_deriv *= self.constants.mass_conc_scale
        return dsd_deriv

    def fallout_deriv(self, var_name=None):
        """Return derivative of fallout with respect to different variables.

        Arguments:
        var_name (optional) - Return information for derivative with respect to
                              the variable named by this string.

        If var_name is not provided, a 1D array of length deriv_var_num is
        returned, with all derivatives in it. If var_name is provided, a
        single scalar is returned for that derivative.
        """
        fallout_deriv_raw = self.desc.fallout_deriv_raw(self.raw, var_name)
        if var_name is None:
            output = np.zeros(fallout_deriv_raw.shape)
            for i, dvar in enumerate(self.desc.deriv_vars):
                output[i] = dvar.nondimensional_to_si(fallout_deriv_raw[i])
        else:
            dvar = self.desc.find_deriv_var(var_name)
            output = dvar.nondimensional_to_si(fallout_deriv_raw)
        return output * self.constants.mass_conc_scale

    def perturb_cov(self):
        """Return perturbation covariance matrix."""
        output = self.desc.perturb_cov_raw(self.raw).copy()
        pn = self.desc.perturbed_num
        pvars = self.desc.perturbed_vars
        for i in range(pn):
            for j in range(pn):
                output[i,j] *= pvars[i].scale * pvars[j].scale
        return output

    def dsd_time_deriv_raw(self, proc_tens):
        """Time derivative of the raw dsd using the given processes.

        Arguments:
        proc_tens - List of processes, the rates of which sum to give the
                    time derivative.
        """
        dsd_raw = self.desc.dsd_raw(self.raw, with_fallout=True)
        dfdt = np.zeros(dsd_raw.shape)
        for proc in proc_tens:
            dfdt += proc.calc_rate(dsd_raw, out_flux=True)
        return dfdt

    # Better to just disable this check than to do a relatively "unnatural"
    # refactoring just to reduce the number of short names given to variables.
    def _apply_proc_tens(self, proc_tens, double_time_deriv):
        """Apply proc_tens to self, returning raw vector time derivative.

        Arguments:
        proc_tens - List of processes, the rates of which sum to give the
                    time derivative.
        double_time_deriv - Array accumulating the second derivative of the dsd
                            with respect to time. If input is None, then this
                            calculation is not performed.
        """
        desc = self.desc
        dvn = desc.deriv_var_num
        dfdt = np.zeros((len(self.raw),))
        dsd_raw = desc.dsd_raw(self.raw)
        didx, dnum = desc.dsd_loc(with_fallout=True)
        for proc in proc_tens:
            if dvn > 0:
                dfdt += self._apply_proc_tens_with_deriv(dsd_raw, proc,
                                                         double_time_deriv)
            else:
                dfdt[didx:didx+dnum] += proc.calc_rate(dsd_raw, out_flux=True)
        return dfdt

    def _apply_proc_tens_with_deriv(self, dsd_raw, proc, double_time_deriv):
        """Apply single process to self assuming dvn > 0.

        Arguments:
        dsd_raw - Raw dsd vector.
        proc - The process.
        double_time_deriv - Array accumulating the second derivative of the dsd
                            with respect to time. If input is None, then this
                            calculation is not performed.
        """
        desc = self.desc
        dfdt = np.zeros((len(self.raw),))
        dsd_deriv_raw = desc.dsd_deriv_raw(self.raw, with_fallout=True)
        didx, dnum = desc.dsd_loc(with_fallout=True)
        dridxs, drnum = desc.dsd_deriv_loc(with_fallout=True)
        rate, derivative = proc.calc_rate(dsd_raw, out_flux=True,
                                        derivative=True)
        dfdt[didx:didx+dnum] = rate
        for i, dridx in enumerate(dridxs):
            dfdt[dridx:dridx+drnum] = derivative @ dsd_deriv_raw[i,:]
        if double_time_deriv is not None:
            double_time_deriv += derivative @ rate
        return dfdt

    def time_derivative_raw(self, proc_tens, perturb=None):
        """Time derivative of the state using the given processes.

        Arguments:
        proc_tens - List of processes, the rates of which sum to give the
                    time derivative.
        perturb (optional) - StochasticPerturbation affecting perturbed
                             variables.
        """
        desc = self.desc
        nb = self.mass_grid.num_bins
        dvn = desc.deriv_var_num
        pn = desc.perturbed_num
        if dvn+1 < pn and (perturb is None or perturb.correction_time is None):
            raise ValueError("perturbation correction_time is not set, but"
                             f" the dimension of perturbation ({pn}) exceeds"
                             f" the dimension of derivative set ({dvn+1})")
        if pn > 0:
            double_time_deriv = np.zeros((nb+1))
        else:
            # We don't need to calculate this without perturbations to apply.
            double_time_deriv = None
        dfdt = self._apply_proc_tens(proc_tens, double_time_deriv)
        if pn > 0:
            if perturb is None:
                perturbation_rate = np.zeros((pn, pn))
                correction_time = None
            else:
                perturbation_rate = perturb.perturbation_rate
                correction_time = perturb.correction_time
            ddsddt = desc.dsd_raw(dfdt)
            ddsddt_deriv = np.zeros((dvn+1,nb))
            ddsddt_deriv[0,:] = double_time_deriv[:nb]
            ddsddt_deriv[1:,:] = desc.dsd_deriv_raw(dfdt)
            perturb_cov_raw = desc.perturb_cov_raw(self.raw)
            lfs = np.zeros((pn,))
            lf_jac = np.zeros((pn, dvn+1))
            lf_rates = np.zeros((pn,))
            lf_rate_jac = np.zeros((pn, dvn+1))
            for i in range(pn):
                wv = self.desc.perturbed_vars[i].weight_vector
                lfs[i], lf_jac[i,:] = \
                    self.linear_func_raw(wv, derivative=True,
                                         dfdt=ddsddt)
                lf_rates[i], lf_rate_jac[i,:] = \
                    self.linear_func_rate_raw(wv, ddsddt,
                                              dfdt_deriv=ddsddt_deriv)
            transform_mat = np.zeros((pn, pn))
            transform_mat2 = np.zeros((pn,))
            for i in range(pn):
                transform = desc.perturbed_vars[i].transform
                transform_mat[i,i] = transform.derivative(lfs[i])
                transform_mat2[i] = \
                    transform.second_over_first_derivative(lfs[i])
            zeta_to_v = transform_mat @ lf_jac
            jacobian = transform_mat @ lf_rate_jac @ la.pinv(zeta_to_v)
            jacobian += np.diag(lf_rates * transform_mat2)
            if correction_time is None:
                perturb_cov_projected = perturb_cov_raw
            else:
                error_cov_inv = la.inv(perturbation_rate)
                projection = la.inv(zeta_to_v.T @ error_cov_inv
                                        @ zeta_to_v)
                projection = zeta_to_v @ projection @ zeta_to_v.T \
                                @ error_cov_inv
                perturb_cov_projected = projection @ perturb_cov_raw \
                                            @ projection.T
            cov_rate = jacobian @ perturb_cov_projected
            cov_rate += cov_rate.T
            cov_rate += perturbation_rate
            if correction_time is not None:
                cov_rate += (perturb_cov_projected - perturb_cov_raw) \
                                / correction_time
            # Convert this rate to a rate on the Cholesky decomposition.
            # Say that PHI(A) is a function returning the lower-triangular
            # matrix B such that B + B^T = A. Then derivatives of a matrix
            # S = L L^T can be propagated to the Cholesky decomposition L using:
            #     dL/dt = L PHI(L^{-1} dS/dt L^{-T})
            # This can be derived by applying the matrix product rule to
            # S = L L^T, multiplying by the inverses used above on the left and
            # right sides, and noting that the result has the form of a matrix
            # added to its transpose, allowing for simplification using PHI.
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
        lf = np.dot(dsd_raw, weight_vector)
        if not derivative:
            return lf
        dvn = self.desc.deriv_var_num
        offset = 1 if dfdt is not None else 0
        dsd_deriv_raw = np.zeros((dvn+offset, nb))
        if dfdt is not None:
            dsd_deriv_raw[0,:] = dfdt
        dsd_deriv_raw[offset:,:] = self.desc.dsd_deriv_raw(self.raw)
        grad = dsd_deriv_raw @ weight_vector
        return lf, grad

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
            (dvn, num_bins)
        where dvn is the number of derivatives that will be returned in the
        second argument.
        """
        dlfdt = np.dot(dfdt, weight_vector)
        if dfdt_deriv is None:
            return dlfdt
        return dlfdt, dfdt_deriv @ weight_vector

    def zeta_cov_raw(self, ddsddt):
        """Find the raw error covariance of dsd_deriv variables and time.

        Arguments:
        ddsddt - Time derivative of raw DSD, e.g. the first num_bins elements
                 of dsd_time_deriv_raw.
        """
        desc = self.desc
        dvn = desc.deriv_var_num
        pn = desc.perturbed_num
        lfs = np.zeros((pn,))
        lf_jac = np.zeros((pn, dvn+1))
        for i in range(pn):
            wv = desc.perturbed_vars[i].weight_vector
            lfs[i], lf_jac[i,:] = self.linear_func_raw(wv, derivative=True,
                                                       dfdt=ddsddt)
        transform_mat = np.diag(
            [desc.perturbed_vars[i].transform.derivative(lfs[i])
             for i in range(pn)]
        )
        v_to_zeta = la.pinv(transform_mat @ lf_jac)
        # We are assuming here that perturb_cov does not need the "correction"
        # for pn > dvn + 1.
        perturb_cov_raw = desc.perturb_cov_raw(self.raw)
        return v_to_zeta @ perturb_cov_raw @ v_to_zeta.T

    def rain_prod_breakdown(self, proc, cloud_vector, derivative=None):
        """Calculate autoconversion and accretion rates.

        Arguments:
        proc - Process.
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
        total_inter = proc.calc_rate(dsd_raw, out_flux=True,
                                      derivative=derivative)
        if derivative:
            save_deriv = total_inter[1]
            total_inter = total_inter[0]
            dvn = self.desc.deriv_var_num
            dsd_deriv_raw = np.zeros((dvn+1, nb+1))
            dsd_deriv_raw[0,:] = total_inter
            dsd_deriv_raw[1:,:] = self.desc.dsd_deriv_raw(self.raw,
                                                          with_fallout=True)
            total_deriv = save_deriv @ dsd_deriv_raw.T
        cloud_dsd_raw = dsd_raw * cloud_vector
        cloud_inter = proc.calc_rate(cloud_dsd_raw, out_flux=True,
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
        if not derivative:
            return rates
        rate_deriv = np.zeros((2, dvn+1))
        rate_deriv[0,:] = (m3_vector * rain_vector) @ cloud_deriv[:nb,:] \
            + cloud_deriv[nb,:]
        no_soa_deriv = total_deriv - cloud_deriv
        rate_deriv[1,:] = -(m3_vector * cloud_vector) @ no_soa_deriv[:nb,:]
        rate_deriv *= rate_scale
        rate_deriv[:,0] /= self.constants.time_scale
        for i in range(dvn):
            dvar = self.desc.deriv_vars[i]
            rate_deriv[:,1+i] = dvar.nondimensional_to_si(rate_deriv[:,1+i])
        return rates, rate_deriv
