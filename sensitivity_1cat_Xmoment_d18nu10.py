#!/usr/bin/env python3

import os
import sys
import traceback

import numpy as np
from scipy.special import gamma
import netCDF4 as nc4

import bin_model as bm

os.makedirs('sensitivity_experiments', exist_ok=True)

moments = sorted([int(sys.argv[i]) for i in range(1, len(sys.argv))])
moments_str = '-'.join([str(m) for m in moments])

KERNEL_FILE_NAME = \
    os.path.join("kernels", "Hall_ScottChen_kernel_nb336.nc")

EXP_FILE_NAME_TEMPLATE = \
    os.path.join("sensitivity_experiments",
                 "experiment_d{:02d}nu{:02d}_f{}.nc")

# Initial conditions
INITIAL_MASS = 1.e-3 # Initial mass concentration (kg/m^3)
# Initial mean diameters (micron)
INITIAL_DS_MICRONS = [18]
# Shape parameter for initial condition
INITIAL_NUS = [10]

# Run length in seconds
END_TIME = 7200.

# Time step in seconds
DT = 1.

# Read kernel file.
with nc4.Dataset(KERNEL_FILE_NAME, "r") as nc:
    netcdf_file = bm.NetcdfFile(nc)
    const, kernel, grid, ktens = netcdf_file.read_cgk()

# Define initial condition variables.
dsd_deriv_names = ['lambda', 'nu', 'M3']
dsd_deriv_scales = [const.std_diameter, 1., 1. / const.mass_conc_scale]
# Define perturbed variables and error growth rates.
# Rate of error growth in decibels for large errors.
db_scale = 10. / np.log(10.)
db_error_rate = 1. # db after one hour
# For small errors, we have absolute rather than relative errors.
m0_scale = const.mass_conc_scale / const.std_mass
abs_error_scale = 2. * db_scale / (db_error_rate * m0_scale)
# Generate perturbed variable list.
perturbed_variables = []
for mom in moments:
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
    perturbed_variables.append((wv, transform, db_scale))
# Rate of error growth.
perturbation_rate = db_error_rate**2 * np.eye(len(perturbed_variables)) / 3600.
# Rate at which invalid moment combinations are relaxed toward valid values.
correction_time = 5. # seconds
# Form state descriptor from variables.
desc = bm.ModelStateDescriptor(const, grid,
                               dsd_deriv_names=dsd_deriv_names,
                               dsd_deriv_scales=dsd_deriv_scales,
                               perturbed_variables=perturbed_variables,
                               perturbation_rate=perturbation_rate,
                               correction_time=correction_time)

# Define integrator
integrator = bm.RK45Integrator(const, DT)

for d_micron in INITIAL_DS_MICRONS:
    for nu in INITIAL_NUS:
        # Convert to meters.
        d = d_micron * 1.e-6
        # Lambda parameter for initial condition (m^-1)
        lambda_init = nu / d
        # Initial DSD and derivative values.
        dsd = bm.gamma_dist_d(grid, lambda_init, nu)
        dsd_deriv = np.zeros((desc.dsd_deriv_num, grid.num_bins))
        dsd_deriv[0,:] = bm.gamma_dist_d_lam_deriv(grid, lambda_init, nu)
        dsd_deriv[1,:] = bm.gamma_dist_d_nu_deriv(grid, lambda_init, nu)
        dsd_deriv[2,:] = dsd / INITIAL_MASS
        # Scale DSD to have the correct initial mass.
        dsd_scale = INITIAL_MASS / np.dot(dsd, grid.bin_widths)
        dsd *= dsd_scale
        dsd_deriv *= dsd_scale
        # Construct initial state.
        raw = desc.construct_raw(dsd, dsd_deriv=dsd_deriv)
        initial_state = bm.ModelState(desc, raw)
        # Run experiment.
        try:
            exp = integrator.integrate(END_TIME, initial_state, [ktens])
        except ValueError as err:
            print(err)
            traceback.print_tb(err.__traceback__)
            print("Integration failed for d=", d_micron, " microns, nu = ", nu)
            print("Continuing after the above error...", flush=True)
            continue
        except AssertionError as err:
            print(err)
            traceback.print_tb(err.__traceback__)
            print("Integration failed for d=", d_micron, " microns, nu = ", nu)
            print("Continuing after the above error...", flush=True)
            continue
        # Save output.
        exp_file_name = EXP_FILE_NAME_TEMPLATE.format(
            d_micron, nu, moments_str)
        with nc4.Dataset(exp_file_name, "w") as nc:
            netcdf_file = bm.NetcdfFile(nc)
            netcdf_file.write_full_experiment(exp, [KERNEL_FILE_NAME])
