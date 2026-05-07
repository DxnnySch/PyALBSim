import math

import numpy as np

from alb_sim.config.laser import LaserConfig
from alb_sim.utils.types import Vector3


class LaserModel:
    def __init__(self, config: LaserConfig):
        """
        Laser source model derived from a laser configuration.

        Parameters
        ----------
        config : LaserConfig
            Configuration describing wavelength, divergence, and pointing angles.
        """
        self._config = config
        self.direction = self._compute_direction()

    def _compute_direction(self) -> Vector3:
        """Compute the unit direction vector of the laser beam in world coordinates."""
        theta = math.radians(self._config.nadir_angle)
        phi = math.radians(self._config.azimuth_angle)

        return np.array(
            [
                math.sin(theta) * math.cos(phi),  # x
                -math.cos(theta),  # -y (down)
                math.sin(theta) * math.sin(phi),  # z
            ]
        )

    @property
    def divergence_angle(self) -> float:
        """Beam divergence half-angle in radians."""
        return self._config.divergence_angle

    @property
    def pulse_fwhm(self) -> float:
        """Laser pulse full width at half maximum in seconds."""
        return self._config.pulse_fwhm
