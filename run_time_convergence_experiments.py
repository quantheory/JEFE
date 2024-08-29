#!/usr/bin/env python3

from time import perf_counter
import os

import numpy as np
from scipy.special import gamma
import netCDF4 as nc4

import bin_model as bm

os.makedirs('convergence_experiments', exist_ok=True)

CTENS_FILE_NAME = \
    os.path.join("collision_data", "Hall_ScottChen_ctens_nb168.nc")

OUTPUT_FILE_NAME_TEMPLATE = \
    os.path.join("convergence_experiments",
                 "time_convergence_experiment_dtexp{}_{}.nc")

# Initial conditions
INITIAL_MASS = 1.e-3 # Initial mass concentration (kg/m^3)
INITIAL_NC = 100. # Initial number concentration (cm^-3)
INITIAL_NU = 6. # Shape parameter for initial condition

# Run length in seconds
END_TIME = 3600.

# i-th time step in seconds will be MAX_TIME_STEP * 2**(-DTEXPS[i])
MAX_TIME_STEP = 80.
DTEXPS = list(range(5))

with nc4.Dataset(CTENS_FILE_NAME, "r") as nc:
    netcdf_file = bm.NetcdfFile(nc)
    const, ckern, grid, ctens = netcdf_file.read_ckgt()

m3_init = INITIAL_MASS / (const.rho_water * np.pi/6.) # m^3 / kg 3rd moment
m0_init = INITIAL_NC * 1.e6 * const.rho_air # kg^-1 number concentration
lambda_init = ( m0_init * gamma(INITIAL_NU + 3)
    / (m3_init * gamma(INITIAL_NU)) )**(1./3.) # m^-1 scale parameter

desc = bm.ModelStateDescriptor(const, grid)
dsd = bm.gamma_dist_d(grid.bin_bounds_d, lambda_init, INITIAL_NU)
dsd *= INITIAL_MASS / np.sum(dsd)
raw = desc.construct_raw(dsd)
initial_state = bm.ModelState(desc, raw)

integrator_types = {
    'FE': bm.ForwardEulerIntegrator,
    'RK4': bm.RK4Integrator,
    'RK45': bm.RK45Integrator,
}

for integrator_name, integrator_type in integrator_types.items():
    for x in DTEXPS:
        print(integrator_name, x)
        dt = MAX_TIME_STEP * 2**(-x)
        integrator = integrator_type(const, dt)
        start_time = perf_counter()
        exp = integrator.integrate(END_TIME, initial_state, [ctens])
        time_taken = perf_counter() - start_time
        print(f"Time taken for x={x}, integrator={integrator_name} is"
              f" {time_taken}.")
        output_file_name = OUTPUT_FILE_NAME_TEMPLATE.format(x, integrator_name)
        with nc4.Dataset(output_file_name, "w") as nc:
            netcdf_file = bm.NetcdfFile(nc)
            netcdf_file.write_full_experiment(exp, [CTENS_FILE_NAME])
            netcdf_file.write_scalar('wall_time_taken', time_taken, 'f8', 's',
                                     "Time taken to run simulation")
