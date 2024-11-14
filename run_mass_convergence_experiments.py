#!/usr/bin/env python3

from time import perf_counter
import os

import numpy as np
from scipy.special import gamma
import netCDF4 as nc4

import bin_model as bm

os.makedirs('convergence_experiments', exist_ok=True)

BIN_NUMBERS = [42 * 2**i for i in range(7)]

CTENS_TYPE = "Long"
CTENS_FILE_NAME_TEMPLATE = \
    os.path.join("collision_data", CTENS_TYPE+"_ctens_nb{}.nc")

OUTPUT_FILE_NAME_TEMPLATE = \
    os.path.join("convergence_experiments",
                 "mass_convergence_experiment_nb{}.nc")

# Initial conditions
INITIAL_MASS = 1.e-3 # Initial mass concentration (kg/m^3)
INITIAL_NC = 100. # Initial number concentration (cm^-3)
INITIAL_NU = 6. # Shape parameter for initial condition

# Run length in seconds
END_TIME = 3600.

# Time step in seconds
DT = 80.

ctens_file_name = CTENS_FILE_NAME_TEMPLATE.format(BIN_NUMBERS[0])
with nc4.Dataset(ctens_file_name, "r") as nc:
    netcdf_file = bm.NetcdfFile(nc)
    const = netcdf_file.read_constants()

m3_init = INITIAL_MASS / (const.rho_water * np.pi/6.) # m^3 / kg 3rd moment
m0_init = INITIAL_NC * 1.e6 * const.rho_air # kg^-1 number concentration
lambda_init = ( m0_init * gamma(INITIAL_NU + 3)
    / (m3_init * gamma(INITIAL_NU)) )**(1./3.) # m^-1 scale parameter

compiled = False
for nb in BIN_NUMBERS:
    ctens_file_name = CTENS_FILE_NAME_TEMPLATE.format(nb)
    with nc4.Dataset(ctens_file_name, "r") as nc:
        netcdf_file = bm.NetcdfFile(nc)
        const, ckern, grid, basis, ctens = netcdf_file.read_tensor_and_metadata()
    recon = bm.ConstantReconstruction(grid)
    coal = bm.CollisionCoalescence(recon, ctens)
    desc = bm.ModelStateDescriptor(const, grid)
    dsd = bm.gamma_dist_d(grid.bin_bounds_d, lambda_init, INITIAL_NU)
    dsd *= INITIAL_MASS / np.sum(dsd)
    raw = desc.construct_raw(dsd)
    initial_state = bm.ModelState(desc, raw)
    # Avoid Numba compilation inside performance counter.
    if not compiled:
        initial_state.time_derivative_raw([coal])
        compiled = True
    integrator = bm.RK45Integrator(const, DT)
    start_time = perf_counter()
    exp = integrator.integrate(END_TIME, initial_state, [coal])
    time_taken = perf_counter() - start_time
    print(f"Time taken for nb={nb} is {time_taken}.")
    output_file_name = OUTPUT_FILE_NAME_TEMPLATE.format(nb)
    with nc4.Dataset(output_file_name, "w") as nc:
        netcdf_file = bm.NetcdfFile(nc)
        netcdf_file.write_full_experiment(exp, [ctens_file_name])
        netcdf_file.write_scalar('wall_time_taken', time_taken, 'f8', 's',
                                 "Time taken to run simulation")
