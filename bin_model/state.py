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
