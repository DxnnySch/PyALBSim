import math
from camera import Camera
from laser import Laser
from utils.ffscatter import FFScatter

import numpy as np


class World:
    def __init__(self, laser_settings: Laser, camera_settings: Camera):
        self.light_speed_air = 2.998e8; # speed of light in vacuum. m/s
        self.refractive_index_water = 1.33334; # Refractive index of water
        self.light_speed_water = self.light_speed_air / self.refractive_index_water; # speed of light in water. m/s

        # Optical parameters of water body
        self.absorption_coefficient = 0.114; # total absorption coefficient in water. m-1. Here is clean water.
        self.total_scattering_coefficient = 0.037; # total scattering coefficient
        self.salinity_unit = 37; # salinity unit. The salt content in pure seawater. S is generally 37 ppt (Pure water is 0 ppt).
        self.seawater_molecular_scattering_coefficient = (1 + 0.008027 * self.salinity_unit) * 0.00012 * laser_settings.laser_wavelength**(-4.24); # Seawater Molecular Scattering Coefficient
        self.particle_scattering_coefficient = self.total_scattering_coefficient - self.seawater_molecular_scattering_coefficient; # Particle scattering coefficient
        self.seafloor_albedo = 0.15 # how much energy is reflected of the sea floor (how much is not absorbed)

        # FF scattering phase function parameter settings---CORE
        # User parameters (Mobley,2002)
        self.refractive_index_ratio = 1.10 # Refractive index ratio of particles relative to pure water (real part)
        self.ff_scatter_divisions = 18000 # 1~180 degrees Equally divided into M parts.
        self.ff_shape_parameter, self.back_scatter_proportion, self.p_ct_r, self.ff_delta_pi, self.ct_r = FFScatter(self.refractive_index_ratio, self.ff_scatter_divisions)
        self.ff_phase_pdf = self.p_ct_r / np.sum(self.p_ct_r) # normalize p_ct_r as probability density function, sum is okay as values are uniform
        self.ff_phase_cdf = np.cumsum(self.ff_phase_pdf) # cumulative density function
        self.ff_phase_cdf /= self.ff_phase_cdf[-1]

        laser_spot_radius_surface = camera_settings.flying_height * math.tan(laser_settings.laser_divergence_angle / 2) # laser beam spot radius on the sea surface. m
        # Optical parameters of water body--- kd Diffuse attenuation coefficient
        laser_spot_diameter_surface = 2 * laser_spot_radius_surface # the lidar spot diameter on the surface
        kd = self.absorption_coefficient + 4.18 * self.particle_scattering_coefficient * self.back_scatter_proportion * (1 - 0.52 * math.exp(-10.8 * self.absorption_coefficient)) # Diffuse attenuation coefficient (Churnside, 2014)
        print("diffuse attenuation coefficient kd", kd)
        self.lidar_attenuation_coefficient = kd + (self.particle_scattering_coefficient - 4.18 * self.particle_scattering_coefficient * self.back_scatter_proportion * (1 - 0.52 * math.exp(-10.8 * self.total_scattering_coefficient))) * math.exp(-0.85 * laser_spot_diameter_surface * (self.absorption_coefficient + self.particle_scattering_coefficient)) # lidar attenuation coefficient #
        self.water_single_scattering_albedo = (self.lidar_attenuation_coefficient - self.absorption_coefficient) / self.lidar_attenuation_coefficient # Single Scattering Albedo of Water
        print("water single scattering albedo", self.water_single_scattering_albedo)
        self.attenuation_per_scatter = 0.5#(1 - self.water_single_scattering_albedo) # Remaining energy weight after each collision #
        
        print("lidar_attenuation_coefficient", self.lidar_attenuation_coefficient)
        print("attenuation_per_scatter", self.attenuation_per_scatter)  

    # Functions for:
        # Gewässertrübung (Streuung, Absorption)
        # Reflektion und Beugung and Wasseroberfläche, sowohl Hin- als auch Rückweg
        # Reflektion an Gewässerboden
        # Im Sensor sichtbar?