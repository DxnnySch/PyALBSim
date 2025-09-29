import math
from camera import Camera
from laser import Laser
# from utils.ffscatter import FFScatter

import numpy as np

from utils.gpt_ffscatter import sample_scattering_directions_batch
from tabulate import tabulate



class World:
    def __init__(self, laser_settings: Laser, camera_settings: Camera):
        self.light_speed_air = 2.998e8; # speed of light in vacuum. m/s
        self.refractive_index_water = 1.33334; # Refractive index of water
        self.light_speed_water = self.light_speed_air / self.refractive_index_water; # speed of light in water. m/s
        self.base_reflectance = ((1 - self.refractive_index_water) / (1 + self.refractive_index_water)) ** 2

        # Optical parameters of water body
        self.absorption_coefficient = 0.169; # total absorption coefficient in water. m-1.
        self.total_scattering_coefficient = 2.5#1.21; # total scattering coefficient
        self.salinity_unit = 37; # salinity unit. The salt content in pure seawater. S is generally 37 ppt (Pure water is 0 ppt).
        self.seawater_molecular_scattering_coefficient = (1 + 0.008027 * self.salinity_unit) * 0.00012 * laser_settings.laser_wavelength**(-4.24); # Seawater Molecular Scattering Coefficient
        self.particle_scattering_coefficient = self.total_scattering_coefficient - self.seawater_molecular_scattering_coefficient; # Particle scattering coefficient
        self.seafloor_albedo = 0.05 # how much energy is reflected of the sea floor (how much is not absorbed)
        self.water_surface_roughness = 0.035 # alpha parameter for microfacet brdf
        self.water_surface_albedo = 0.1 # how much energy is reflected of the water surface

        # FF scattering phase function parameter settings---CORE
        # User parameters (Mobley,2002)
        self.epsilon = 1e-6
        self.refractive_index_ratio = 1.1 # Refractive index ratio of particles relative to pure water (real part)
        self.junge_slope = 3.62
        self.ff_scatter_divisions = 18000 # 1~180 degrees Equally divided into M parts.
        self.oowb_ff_scatter_theta, self.oowb_ff_scatter_pf = self.ff_scatter_pf()
        self.oowb_ff_scatter_cdf = np.cumsum(self.oowb_ff_scatter_pf * np.sin(self.oowb_ff_scatter_theta))
        self.oowb_ff_scatter_cdf /= self.oowb_ff_scatter_cdf[-1]
        self.oowb_ff_backscatter_fraction = self.ff_backscatter_fraction()
        
        self.back_scatter_proportion = self.oowb_ff_backscatter_fraction

        laser_spot_radius_surface = camera_settings.flying_height * math.tan(laser_settings.laser_divergence_angle / 2) # laser beam spot radius on the sea surface. m
        # Optical parameters of water body--- kd Diffuse attenuation coefficient
        laser_spot_diameter_surface = 2 * laser_spot_radius_surface # the lidar spot diameter on the surface
        kd = self.absorption_coefficient + 4.18 * self.particle_scattering_coefficient * self.back_scatter_proportion * (1 - 0.52 * math.exp(-10.8 * self.absorption_coefficient)) # Diffuse attenuation coefficient (Churnside, 2014)
        self.lidar_attenuation_coefficient = kd + (self.particle_scattering_coefficient - 4.18 * self.particle_scattering_coefficient * self.back_scatter_proportion * (1 - 0.52 * math.exp(-10.8 * self.total_scattering_coefficient))) * math.exp(-0.85 * laser_spot_diameter_surface * (self.absorption_coefficient + self.particle_scattering_coefficient)) # lidar attenuation coefficient #
        self.water_single_scattering_albedo = (self.lidar_attenuation_coefficient - self.absorption_coefficient) / self.lidar_attenuation_coefficient # Single Scattering Albedo of Water
        self.attenuation_per_scatter = self.water_single_scattering_albedo # Remaining energy weight after each collision #0.5#(1 - 
        
        params = {k: v for k,v in vars(self).items() if k not in {"oowb_ff_scatter_theta", "oowb_ff_scatter_pf", "oowb_ff_scatter_cdf"}}
        table = tabulate(params.items(), headers=["Parameter", "Value"], tablefmt="grid")
        # print(table)


    def ff_scatter_pf(self):
        theta = np.linspace(self.epsilon, np.pi, self.ff_scatter_divisions)

        v = (3-self.junge_slope) / 2
        delta = (4 / (3 * (self.refractive_index_ratio - 1)**2)) * np.sin(theta / 2)**2
        delta_180 = (4 / (3 * (self.refractive_index_ratio - 1)**2)) * np.sin(np.pi / 2)**2

        term1 = 1/(4 * np.pi * (1-delta)**2 * delta**v)
        term2 = v * (1-delta) - (1-delta**v) + (delta * (1-delta**v) - v*(1-delta)) * np.sin(theta/2)**(-2)
        term3 = ((1 - delta_180**v)/(16 * np.pi * (delta_180 - 1) * delta_180**v)) * (3 * np.cos(theta)**2 - 1)
        ff = term1 * term2 + term3

        return theta, ff
    
    def ff_backscatter_fraction(self):
        v = (3-self.junge_slope) / 2
        delta_90 = (4 / (3 * (self.refractive_index_ratio - 1)**2)) * np.sin(np.pi / 4)**2

        return 1 - ((1 - delta_90**(v+1) - 0.5 * (1 - delta_90**v)) / ((1-delta_90) * delta_90**v))

    def sample_scattering_angle_batch(self, num_samples):
        rng = np.random.default_rng(42)
        u = rng.random(num_samples)
        return np.interp(u, self.oowb_ff_scatter_cdf, self.oowb_ff_scatter_theta)

    def scatter_energy(self, a_dir, b_dir):
        cos_theta = np.clip(np.dot(a_dir, b_dir), -1.0, 1.0)
        scatter_angle = np.arccos(cos_theta)
        
        delta = (np.pi - self.epsilon) / self.ff_scatter_divisions
        i = int((scatter_angle - self.epsilon) / delta)
        i = min(i, self.ff_scatter_divisions - 1)  # clamp to avoid overflow at edge

        theta_lower = i * delta
        theta_upper = (i + 1) * delta

        cdf_lower = np.interp(theta_lower, self.oowb_ff_scatter_theta, self.oowb_ff_scatter_cdf)
        cdf_upper = np.interp(theta_upper, self.oowb_ff_scatter_theta, self.oowb_ff_scatter_cdf)

        return cdf_upper - cdf_lower

    def sample_scattering_directions_batch(self, incoming_dirs, rng):
        return sample_scattering_directions_batch(self.oowb_ff_scatter_pf, self.oowb_ff_scatter_theta, incoming_dirs, rng)
    # Functions for:
        # Gewässertrübung (Streuung, Absorption)
        # Reflektion und Beugung and Wasseroberfläche, sowohl Hin- als auch Rückweg
        # Reflektion an Gewässerboden
        # Im Sensor sichtbar?