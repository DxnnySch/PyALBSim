import math

import numpy as np

from alb_sim.config.laser import LaserConfig
from alb_sim.utils.types import Vector3


class LaserModel:
    def __init__(self, config: LaserConfig):
        self._config = config
        self.direction = self._compute_direction()

    def _compute_direction(self) -> Vector3:
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
        return self._config.divergence_angle

    @property
    def pulse_fwhm(self) -> float:
        return self._config.pulse_fwhm
