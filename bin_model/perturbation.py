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
"""Description of perturbations used to drive JEFE."""

import numpy as np

# Allow this to be something of a struct for now.
# pylint: disable-next=too-few-public-methods
class StochasticPerturbation:
    """
    Describe perturbation of variables due to a simple stochastic process.

    The perturbations in this case are assumed to result from a Wiener
    process. Given that we assume a linear response of the system to
    perturbation, this means that the effects of small errors are modeled using
    a linear stochastic differential equation. Thus the evolution over time of
    the covariance of the error PDF can be readily modeled.

    Initialization arguments:
    constants - A ModelConstants object.
    perturbed_vars - List of PerturbedVars.
    perturbation_rate - A covariance matrix representing the error
        introduced to the perturbed variables per second.
    correction_time (optional) - Time scale over which the error covariance is
        nudged toward a corrected value. Defaults to None.

    Note that input arguments are converted to nondimensional representation for
    use in this object.

    Other attributes:
    perturbed_num - Number of perturbed variables.
    """

    def __init__(self, constants, perturbed_vars, perturbation_rate,
                 correction_time=None):
        pn = len(perturbed_vars)
        self.perturbed_num = pn
        if len(perturbation_rate.shape) != 2:
            raise ValueError("perturbation rate is not a matrix"
                             f" (shape {perturbation_rate.shape})")
        if perturbation_rate.shape != (pn, pn):
            raise ValueError("perturbation rate has wrong dimensions"
                             f" (shape {perturbation_rate.shape},"
                             f" expected ({pn}, {pn}))")
        if np.any(perturbation_rate != np.transpose(perturbation_rate)):
            raise ValueError("perturbation rate is not a Hermitian matrix")
        self.perturbation_rate = perturbation_rate * constants.time_scale
        for i in range(pn):
            for j in range(pn):
                self.perturbation_rate[i,j] /= \
                    perturbed_vars[i].scale * perturbed_vars[j].scale
        self.correction_time = correction_time
        if self.correction_time is not None:
            self.correction_time /= constants.time_scale
