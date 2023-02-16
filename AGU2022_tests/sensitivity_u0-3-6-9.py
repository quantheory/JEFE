#!/usr/bin/env python3

import os
import traceback

import numpy as np
from scipy.special import gamma
import netCDF4 as nc4

import bin_model as bm

os.makedirs('sensitivity_experiments', exist_ok=True)

MOMENTS = [0, 3, 6, 9]

KERNEL_FILE_NAME = \
    os.path.join("kernels", "Hall_ScottChen_kernel_nb168.nc")

EXP_FILE_NAME_TEMPLATE = \
    os.path.join("sensitivity_experiments",
                 "experiment_aguset_u{}-{}-{}-{}.nc".format(
                     MOMENTS[0], MOMENTS[1], MOMENTS[2], MOMENTS[3],
))

# Initial conditions
INITIAL_MASS = 1.e-3 # Initial mass concentration (kg/m^3)
INITIAL_NC = 220. # Initial number concentration (cm^-3)
NU = 10. # Shape parameter for initial condition

# Run length in seconds
END_TIME = 3600.

# Time step in seconds
DT = 80.

# Read kernel file.
with nc4.Dataset(KERNEL_FILE_NAME, "r") as nc:
    netcdf_file = bm.NetcdfFile(nc)
    const, kernel, grid, ktens = netcdf_file.read_cgk()

# Set up initial conditions on this grid.
m3_init = INITIAL_MASS / (const.rho_water * np.pi/6.) # m^3 / kg 3rd moment
m0_init = INITIAL_NC * 1.e6 / const.rho_air # kg^-1 number concentration
lambda_init = ( m0_init * gamma(NU + 3)
    / (m3_init * gamma(NU)) )**(1./3.) # m^-1 scale parameter
print(f"Initial condition: mass={INITIAL_MASS}, lambda={lambda_init}, nu={NU}")
dsd_init = bm.gamma_dist_d(grid, lambda_init, NU)
dsd_init *= INITIAL_MASS / np.dot(dsd_init, grid.bin_widths)

# Define initial condition variables.
deriv_vars = [
    bm.DerivativeVar('lambda', 1. / const.std_diameter),
    bm.DerivativeVar('nu'),
    bm.DerivativeVar('liquid_mass', const.mass_conc_scale),
]
# Perturbed variables in decibels, except for small values.
db_scale = 10. / np.log(10.)
db_error_rate = 1. # db after one hour
m0_scale = const.mass_conc_scale / const.std_mass
abs_error_scale = 2. * db_scale / (db_error_rate * m0_scale)
# Define perturbed variables.
perturbed_vars = []
for mom in MOMENTS:
    wv = grid.moment_weight_vector(mom)
    # Magnitude of error when moments are small is similar to:
    #  - For cloud: 100 extra drops of size 20 microns per cubic meter
    #  - For rain: 1 extra drop of size 100 microns per cubic meter
    # in an hour.
    # For full moments, add both together.
    x0 = 100. * (20.e-6 / const.std_diameter)**mom
    x0 += 1. * (100.e-6 / const.std_diameter)**mom
    x0 *= abs_error_scale
    transform = bm.QuadToLogTransform(x0)
    perturbed_vars.append(bm.PerturbedVar(f'M{mom}', wv, transform, db_scale))
# Rate of error growth.
perturbation_rate = db_error_rate**2 * np.eye(len(perturbed_vars)) / 3600.
perturb = bm.StochasticPerturbation(const, perturbed_vars, perturbation_rate)
# Form state descriptor from variables.
desc = bm.ModelStateDescriptor(const, grid,
                               deriv_vars=deriv_vars,
                               perturbed_vars=perturbed_vars)

# Define integrator
integrator = bm.RK45Integrator(const, DT)

# Initial derivative values.
dsd_deriv = np.zeros((desc.deriv_var_num, grid.num_bins))
dsd_deriv[0,:] = bm.gamma_dist_d_lam_deriv(grid, lambda_init, NU)
dsd_deriv[1,:] = bm.gamma_dist_d_nu_deriv(grid, lambda_init, NU)
dsd_deriv[2,:] = dsd_init / INITIAL_MASS

# Construct initial state.
raw = desc.construct_raw(dsd_init, dsd_deriv=dsd_deriv)
initial_state = bm.ModelState(desc, raw)
# Run experiment.
print("Starting integration.")
try:
    exp = integrator.integrate(END_TIME, initial_state, [ktens], perturb)
except ValueError as err:
    print(err)
    traceback.print_tb(err.__traceback__)
    print("Integration failed for d=", d_micron, " microns, nu = ", nu)
    print("Continuing after the above error...", flush=True)
except AssertionError as err:
    print(err)
    traceback.print_tb(err.__traceback__)
    print("Integration failed for d=", d_micron, " microns, nu = ", nu)
    print("Continuing after the above error...", flush=True)
# Save output.
exp_file_name = EXP_FILE_NAME_TEMPLATE
with nc4.Dataset(exp_file_name, "w") as nc:
    netcdf_file = bm.NetcdfFile(nc)
    netcdf_file.write_full_experiment(exp, [KERNEL_FILE_NAME])
