#!/usr/bin/env python

from time import perf_counter

import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
from scipy.integrate import quad
import scipy.linalg as la
from scipy.special import i1e, gammainc

import bin_model as bm

# Define physical constants.
RHO_WATER = 1000. # Density of water (kg/m^3)
RHO_AIR = 1.2 # Density of air (kg/m^3)

# Model physical parameters.
RAIN_D = 8.e-5 # Cutoff diameter between particle sizes defined as cloud vs. rain (m).
B = 1.5 # Golovin kernel coefficient (m^3/kg/s)

# Grid parameters
D_MIN = 1.e-6 # Minimum particle diameter (m).
D_MAX = 1.6384e-2 # Maximum particle diameter (m).
BIN_NUMBER = 84 # For D_MAX / D_MIN = 16384, mass-doubling is 42 bins.

# Numerical tuning parameters.
DIAMETER_SCALE = 1.e-4 # Internal scaling for particle size (m)
MASS_CONC_SCALE = 1.e-3 # Internal scaling for mass concentration (kg/m^3)
TIME_SCALE = 1. / (B * MASS_CONC_SCALE) # Internal scaling for time (s)

# Initial Conditions
INITIAL_MEAN_MASS = 1.e-3 # Initial liquid water mass (kg/m^3)
INITIAL_MEAN_DIAMETER = 20.e-6 # Initial mean drop diameter (m)

# Run length in seconds
END_TIME = 3600.

# Time step in seconds
DT = 1.

# Define constants object and grid.
const = bm.ModelConstants(rho_water=RHO_WATER, rho_air=RHO_AIR,
                          diameter_scale=DIAMETER_SCALE,
                          rain_d=RAIN_D, mass_conc_scale=MASS_CONC_SCALE,
                          time_scale=TIME_SCALE)
grid = bm.GeometricMassGrid(const, d_min=D_MIN, d_max=D_MAX,
                            num_bins=BIN_NUMBER)

# Define kernel and tensor object.
ckern = bm.make_golovin_kernel(const, B)
ctens = bm.CollisionTensor(grid, ckern=ckern)

# Create initial dsd on the grid.
v0 = (np.pi / 6.) * (INITIAL_MEAN_DIAMETER)**3 # Initial mean drop volume (m^3)
m0 = RHO_WATER * v0 # Initial mean drop mass (kg^-1)
# Convert bin bounds to drop mass for ease of integrating.
bin_bounds_m = (np.pi * RHO_WATER / 6.) * grid.bin_bounds_d**3 # (kg)
# We want to calculate the mass in each bin for an initially exponential
# distribution, i.e. the integral over each bin of
#     (m / m0) * exp(-m / m0)
# scaled to give the correct total initial mass.
gamma_integrals = INITIAL_MEAN_MASS * gammainc(2., bin_bounds_m / m0)
dsd = gamma_integrals[1:] - gamma_integrals[:-1] # (kg/m^3)

# Set up state descriptor and initial value.
desc = bm.ModelStateDescriptor(const, grid)
raw = desc.construct_raw(dsd)
initial_state = bm.ModelState(desc, raw)

# Model integration.
integrator = bm.ForwardEulerIntegrator(const, DT)
start_time = perf_counter()
exp = integrator.integrate(END_TIME, initial_state, [ctens])
time_taken = perf_counter() - start_time
print(f"Time taken for integration was {time_taken} seconds.")

# Extract mass in each bin and time step from experiment object.
nt = exp.num_time_steps
nb = BIN_NUMBER # Shorter name is useful when using this as an index!
masses = np.zeros((nt, nb))
for i in range(nt):
    masses[i,:] = exp.states[i].dsd()

# Define initial condition information for analytic solution.
n0 = INITIAL_MEAN_MASS / m0 # Initial number concentration (m^-3)
b_vol = B * RHO_WATER # Convert to a rate per volume of drops.

def t_to_tau(t, b, n0, v0):
    """Convert actual time to Golovin dimensionless time coordinate."""
    return 1. - np.exp(-n0 * b * v0 * t)

def d_to_x(d, v0):
    """Convert diameter to Golovin dimensionless volume coordinate."""
    return (np.pi / 6.) * d**3 / v0

def golovin_phi(x, tau):
    """Evaluate analytic Golovin solution over dimensionless coordinates."""
    x_root_tau = x * np.sqrt(tau)
    fac = (1. - tau) / x_root_tau
    bessel = i1e(2. * x_root_tau)
    # To avoid overflow/underflow, we use i1e instead of i1,
    # which is off by a factor of exp(2. * x_root_tau) that
    # we have to explicitly add back here.
    expon = np.exp(2. * x_root_tau - (1. + tau) * x)
    return fac * bessel * expon

def mass_weighted_dsd_at_bin(d, t, b, n0, v0):
    """Evaluate analytic Golovin solution for a given diameter and time."""
    tau = t_to_tau(t, b, n0, v0)
    x = d_to_x(d, v0)
    num_dsd_over_v = (n0 / v0) * golovin_phi(x, tau)
    # Convert a number-weighted DSD over volume to a mass-weighted DSD over
    # diameter. This requires multiplying by d**3 * pi * RHO_WATER / 6 to
    # convert to mass-weighting, and another d**2 * pi / 2 to convert from a DSD
    # with respect to volume to one with respect to diameter.
    mass_weighted_dsd = (np.pi**2 * RHO_WATER / 12.) * d**5 * num_dsd_over_v
    # Prevent underflow-related warnings when integrating with quad.
    if mass_weighted_dsd > 1.e-300:
        return mass_weighted_dsd
    else:
        return 0.

# Calculate analytic solution on same grid as numerical solution.
times = exp.times
gol_masses = np.zeros((2, nb))
for i in range(2):
    t = (i+1) * 1800.
    for j in range(nb):
        upper = grid.bin_bounds_d[j]
        lower = grid.bin_bounds_d[j+1]
        # Integrate with quad so that we have a high-precision estimate of the
        # total mass in each bin.
        gol_masses[i,j], _ = quad(
            mass_weighted_dsd_at_bin,
            upper,
            lower,
            args=(t, b_vol, n0, v0),
            epsabs=1.e-8 * (upper-lower)
        )

idx_1800 = int(np.round(1800 / DT))
idx_3600 = int(np.round(3600 / DT))
print("Mass RMS difference at 1800s: ", la.norm(masses[idx_1800,:]
                                                - gol_masses[0,:]))
print("Mass RMS difference at 3600s: ", la.norm(masses[idx_3600,:]
                                                - gol_masses[0,:]))

with nc4.Dataset("bott_golovin_s2.nc") as data:
    bott_masses = data['mass'][:,:].T

print("Bott mass RMS difference at 1800s: ", la.norm(bott_masses[idx_1800,:]
                                                - gol_masses[0,:]))
print("Bott mass RMS difference at 3600s: ", la.norm(bott_masses[idx_3600,:]
                                                - gol_masses[0,:]))

# Middle radius in each bin in microns, for plotting.
bin_centers_r = 1.e6 * np.sqrt(grid.bin_bounds_d[:-1]
                               * grid.bin_bounds_d[1:]) / 2.
# Convert from the mass in kilograms each bin to approximate mass DSD in grams
# over the log of radius; this can be done by multiplying by 100 to convert to
# grams, then dividing by bin_widths / 3 to convert to DSD over log(radius),
# since the bin widths are in log(mass) and mass is proportional to radius^3.
dsd_facs = 3000. / grid.bin_widths
plt.semilogx(bin_centers_r, dsd_facs * gol_masses[0,:], 'k-',
             label="Analytic (30m)")
plt.semilogx(bin_centers_r, dsd_facs * gol_masses[1,:], 'k--',
             label="Analytic (60m)")
plt.semilogx(bin_centers_r, dsd_facs * masses[idx_1800,:], 'b-',
             label="First-order (30m)")
plt.semilogx(bin_centers_r, dsd_facs * masses[idx_3600,:], 'b--',
             label="First-order (60m)")
plt.semilogx(bin_centers_r, dsd_facs * bott_masses[idx_1800,:], '-',
             color='orange', label="Bott (30m)")
plt.semilogx(bin_centers_r, dsd_facs * bott_masses[idx_3600,:], '--',
             color='orange', label="Bott (60m)")
plt.legend()
plt.xlabel("$\log(r)$ (radius in micron)")
plt.ylabel("Mass-weighted DSD (g/kg)")
plt.axis([grid.bin_bounds_d[0]*5.e5, grid.bin_bounds_d[-1]*5.e5,
          0., 0.8])
plt.savefig("golovin_compare.png", dpi=300)
