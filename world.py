from laser import Laser


class World:
    def __init__(self, laser_settings: Laser):
        self.light_speed_air = 2.998e8; # speed of light in vacuum. m/s
        self.refractive_index_water = 1.33334; # Refractive index of water
        self.light_speed_water = self.light_speed_air / self.refractive_index_water; # speed of light in water. m/s

        self.absorption_coefficient = 0.114; # total absorption coefficient in water. m-1. Here is clean water.
        self.total_scattering_coefficient = 0.037; # total scattering coefficient
        self.salinity_unit = 37; # salinity unit. The salt content in pure seawater. S is generally 37 ppt (Pure water is 0 ppt).
        self.seawater_molecular_scattering_coefficient = (1 + 0.008027 * self.salinity_unit) * 0.00012 * laser_settings.laser_wavelength**(-4.24); # Seawater Molecular Scattering Coefficient
        self.particle_scattering_coefficient = self.total_scattering_coefficient - self.seawater_molecular_scattering_coefficient; # Particle scattering coefficient

    # Functions for:
        # Gewässertrübung (Streuung, Absorption)
        # Reflektion und Beugung and Wasseroberfläche, sowohl Hin- als auch Rückweg
        # Reflektion an Gewässerboden
        # Im Sensor sichtbar?