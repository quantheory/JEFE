#!/usr/bin/env python3

import os

import numpy as np
import netCDF4 as nc4

import bin_model as bm

os.makedirs('kernels', exist_ok=True)

# Define physical constants.
RHO_WATER = 1000. # Density of water (kg/m^3)
RHO_AIR = 1.2 # Density of air (kg/m^3)

# Model physical parameters.
RAIN_D = 8.e-5 # Cutoff diameter between particle sizes defined as cloud vs. rain (m).

# Grid parameters
D_MIN = 1.e-6 # Minimum particle diameter (m).
D_MAX = 1.6384e-2 # Maximum particle diameter (m).
BIN_NUMBERS = [42 * 2**i for i in range(7)]

# Numerical tuning parameters.
DIAMETER_SCALE = 1.e-4 # Internal scaling for particle size (m)
MASS_CONC_SCALE = 1.e-3 # Internal scaling for mass concentration (kg/m^3)
# Long's kernel magnitude kc (m^3/kg^2/s)
long_kernel_size = 9.44e9
# Internal scaling for time (s)
TIME_SCALE = 1. / (long_kernel_size * ((np.pi*RHO_WATER/6.)*DIAMETER_SCALE**3)
                       * MASS_CONC_SCALE)

FILE_NAME_TEMPLATE = os.path.join("kernels", "Hall_ScottChen_kernel_nb{}.nc")

const = bm.ModelConstants(rho_water=RHO_WATER, rho_air=RHO_AIR, diameter_scale=DIAMETER_SCALE,
                          rain_d=RAIN_D, mass_conc_scale=MASS_CONC_SCALE,
                          time_scale=TIME_SCALE)

kernel = bm.HallKernel(const, 'ScottChen')

for nb in BIN_NUMBERS:
    print(f"Creating kernel with {nb} bins.")
    grid = bm.GeometricMassGrid(const, d_min=D_MIN, d_max=D_MAX, num_bins=nb)
    ctens = bm.CollisionTensor(grid, kernel=kernel)
    file_name = FILE_NAME_TEMPLATE.format(nb)
    with nc4.Dataset(file_name, "w") as nc:
        netcdf_file = bm.NetcdfFile(nc)
        netcdf_file.write_cgk(ctens)
