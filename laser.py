import numbers
import math

import numpy as np

from camera import Camera

class Laser:
    def __init__(self, camera_settings: Camera, sample_multiplier: int, aperture = 0.1, laser_divergence_angle = 1 * 1e-3, field_of_view = 10 * 1e-3 / 2, laser_wavelength = 0.532, t_max = 50e-9):
        self.aperture = aperture # Receiver Telescope Radius. Unit is m.
        self.laser_divergence_angle = laser_divergence_angle # unit in rad.
        self.laser_direction = [math.sin(math.radians(30 / 2)), -math.cos(math.radians(30 / 2)), 0]#[0.1, -1, 0]
        self.field_of_view = field_of_view # unit in rad
        self.laser_wavelength = laser_wavelength # Laser wavelength. um
        
        self.dt = 1 / (camera_settings.sample_rate * sample_multiplier)  # time step (s)
        self.FWHM = 8.3e-9  # full width at half maximum (s)
        self.sigma_t = self.FWHM / (2 * np.sqrt(2 * np.log(2)))  # std of pulse (s)
        t_max = t_max  # +/- 50 ns window
        self.t_steps = np.arange(-t_max, t_max + self.dt, self.dt)
        # Gaussian pulse (unnormalized)
        self.pulse = np.exp(-(self.t_steps ** 2) / (2 * self.sigma_t ** 2))
        # normalize to probability mass function (sum = 1)
        self.pulse /= self.pulse.sum()
        
    def get_emission_times(self, num_photons: int, rng: np.random.Generator):
        indices = rng.choice(len(self.t_steps), size=num_photons, p=self.pulse)
        time_step_offsets = indices - len(self.t_steps) // 2  # relative to center (0)
        return np.array(time_step_offsets * self.dt, dtype=np.float32)




