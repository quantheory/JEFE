#!/usr/bin/env python3

import os
import traceback

import numpy as np
from scipy.special import gamma
import netCDF4 as nc4

import bin_model as bm

os.makedirs('sensitivity_experiments', exist_ok=True)

KERNEL_FILE_NAME = \
    os.path.join("kernels", "Hall_ScottChen_kernel_nb336.nc")

EXP_FILE_NAME_TEMPLATE = \
    os.path.join("sensitivity_experiments",
                 "experiment_d{:02d}nu{:02d}_c0-3_r0-3_logt.nc")

# Initial conditions
INITIAL_MASS = 1.e-3 # Initial mass concentration (kg/m^3)
# Initial mean diameters (micron)
INITIAL_DS_MICRONS = [15 + 3*i for i in range(6)]
# Shape parameter for initial condition
INITIAL_NUS = [7]

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
# Define perturbed variables.
# The cloud and rain 0th and 3rd moments...
wvc0 = grid.moment_weight_vector(0, cloud_only=True)
wvc3 = grid.moment_weight_vector(3, cloud_only=True)
wvr0 = grid.moment_weight_vector(0, rain_only=True)
wvr3 = grid.moment_weight_vector(3, rain_only=True)
# ... in decibels.
db_scale = 10. / np.log(10.)
perturbed_variables = [
    (wvc0, bm.LogTransform(), db_scale),
    (wvc3, bm.LogTransform(), db_scale),
    (wvr0, bm.LogTransform(), db_scale),
    (wvr3, bm.LogTransform(), db_scale),
]
# Subject to this rate of error growth.
error_rate = 1. # db after one hour
perturbation_rate = error_rate**2 * np.eye(len(perturbed_variables)) / 3600.
# Form state descriptor from variables.
desc = bm.ModelStateDescriptor(const, grid,
                               dsd_deriv_names=dsd_deriv_names,
                               dsd_deriv_scales=dsd_deriv_scales,
                               perturbed_variables=perturbed_variables,
                               perturbation_rate=perturbation_rate)

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
        exp_file_name = EXP_FILE_NAME_TEMPLATE.format(d_micron, nu)
        with nc4.Dataset(exp_file_name, "w") as nc:
            netcdf_file = bm.NetcdfFile(nc)
            netcdf_file.write_full_experiment(exp, [KERNEL_FILE_NAME])
