import numpy as np

from alb_sim.config.sea_surface import SeaSurfaceConfig
from alb_sim.config.water import WaterConfig
from alb_sim.physics.reflection.microfacet_brdf import (
    microfacet_reflected_energy,
    microfacet_transmitted_energy,
)
from alb_sim.physics.reflection.snell_refraction import calculate_refraction_direction
from alb_sim.utils.types import Array, BoolArray, Vector3, Vector3Array


class SeaSurfaceModel:
    def __init__(self, config: SeaSurfaceConfig, water_config: WaterConfig):
        self._config = config
        self._water_config = water_config

        self.base_reflectance = self._calculate_base_reflectance()

    def _calculate_base_reflectance(self) -> float:
        return (
            (1 - self._water_config.layers[0].refractive_index)
            / (1 + self._water_config.layers[0].refractive_index)
        ) ** 2

    def reflected_energy(
        self, photon_directions: Vector3Array, sensor_photon_direction: Vector3
    ) -> Array:
        normal_direction = np.array([0, 1, 0])
        specular = microfacet_reflected_energy(
            photon_directions,
            sensor_photon_direction,
            normal_direction,
            self._config.roughness,
            self.base_reflectance,
        )

        return specular  # + (self._config.albedo / np.pi) * np.maximum(0.0, dot_batch_single(photon_directions, normal_direction))

    def transmitted_energy(
        self,
        photon_directions: Vector3Array,
        sensor_photon_direction: Vector3,
        normal_direction: Vector3,
    ) -> Array:
        transmission = (
            microfacet_transmitted_energy(
                photon_directions,
                sensor_photon_direction,
                normal_direction,
                self._config.roughness,
                self._water_config.layers[0].refractive_index,
                1,
                self.base_reflectance,
            )
            + self._config.albedo / np.pi
        )

        return transmission

    def calculate_refraction_direction(
        self, incoming_directions: Vector3Array, normal: Vector3
    ) -> tuple[Vector3Array, BoolArray]:
        return calculate_refraction_direction(
            incoming_directions,
            normal,
            1,
            self._water_config.layers[0].refractive_index,
        )
