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

"""Class containing constants for use in JEFE's bin model."""

import numpy as np

# pylint: disable-next=too-many-instance-attributes
class ModelConstants:
    """
    Define relevant constants and scalings for the model.

    Initialization arguments:
    rho_water - Density of water (kg/m^3).
    rho_air - Density of air (kg/m^3).
    diameter_scale - Diameter (m) of a particle of "typical" size, used
                   internally to non-dimensionalize particle sizes.
    rain_d - Diameter (m) of threshold diameter defining the distinction
             between cloud and rain particles.
    mass_conc_scale (optional) - Mass concentration scale used for
                                 nondimensionalization.
    time_scale (optional) - Time scale used for nondimensionalization.

    Other attributes:
    std_mass - Mass in kg corresponding to a scaled mass of 1.
    rain_m - `rain_d` converted to scaled mass.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(self, rho_water, rho_air, diameter_scale, rain_d,
                 mass_conc_scale=None, time_scale=None):
        if mass_conc_scale is None:
            mass_conc_scale = 1.
        if time_scale is None:
            time_scale = 1.
        self.rho_water = rho_water
        self.rho_air = rho_air
        self.diameter_scale = diameter_scale
        self.std_mass = rho_water * np.pi/6. * diameter_scale**3
        self.rain_d = rain_d
        self.rain_m = self.diameter_to_scaled_mass(rain_d)
        self.mass_conc_scale = mass_conc_scale
        self.time_scale = time_scale

    def diameter_to_scaled_mass(self, d):
        """Convert diameter in meters to non-dimensionalized particle size."""
        return (d / self.diameter_scale)**3

    def scaled_mass_to_diameter(self, x):
        """Convert non-dimensionalized particle size to diameter in meters."""
        return self.diameter_scale * x**(1./3.)

    @classmethod
    def from_netcdf(cls, netcdf_file):
        """Retrieve a ModelConstants object from a NetcdfFile."""
        rho_water = netcdf_file.read_scalar('rho_water')
        rho_air = netcdf_file.read_scalar('rho_air')
        diameter_scale = netcdf_file.read_scalar('diameter_scale')
        rain_d = netcdf_file.read_scalar('rain_d')
        mass_conc_scale = netcdf_file.read_scalar('mass_conc_scale')
        time_scale = netcdf_file.read_scalar('time_scale')
        return ModelConstants(rho_water=rho_water,
                              rho_air=rho_air,
                              diameter_scale=diameter_scale,
                              rain_d=rain_d,
                              mass_conc_scale=mass_conc_scale,
                              time_scale=time_scale)

    def to_netcdf(self, netcdf_file):
        """Write data from this object to a netCDF file."""
        netcdf_file.write_scalar('rho_water', self.rho_water,
            'f8', "kg/m^3",
            "Density of water")
        netcdf_file.write_scalar('rho_air', self.rho_air,
            'f8', "kg/m^3",
            "Density of air")
        netcdf_file.write_scalar('diameter_scale', self.diameter_scale,
            'f8', "m",
            "Particle length scale used for nondimensionalization")
        netcdf_file.write_scalar('rain_d', self.rain_d,
            'f8', "m",
            "Threshold diameter defining the boundary between cloud and rain")
        netcdf_file.write_scalar('mass_conc_scale', self.mass_conc_scale,
            'f8', "kg/m^3",
            "Liquid mass concentration scale used for nondimensionalization")
        netcdf_file.write_scalar('time_scale', self.time_scale,
            'f8', "s",
            "Time scale used for nondimensionalization")
