import numpy as np

from alb_sim.config.sea_surface import SeaSurfaceConfig
from alb_sim.physics.models.water import WaterModel
from alb_sim.physics.reflection.microfacet_brdf import (
    microfacet_reflected_energy,
    microfacet_transmitted_energy,
)
from alb_sim.physics.reflection.snell_refraction import calculate_refraction_direction
from alb_sim.utils.types import Array, BoolArray, Vector3, Vector3Array


class SeaSurfaceModel:
    def __init__(self, config: SeaSurfaceConfig, water_model: WaterModel):
        """
        Microfacet-based model of the sea surface boundary.

        Parameters
        ----------
        config : SeaSurfaceConfig
            Configuration describing albedo and surface roughness.
        water_model : WaterModel
            Water column model providing refractive index information.
        """
        self._config = config
        self._water_model = water_model

        self.base_reflectance = self._calculate_base_reflectance()

    def _calculate_base_reflectance(self) -> float:
        """Compute normal-incidence Fresnel reflectance at the air–water interface."""
        refractive_index = self._water_model.layers[0].refractive_index_at(0)
        return ((1 - refractive_index) / (1 + refractive_index)) ** 2

    def reflected_energy(
        self, photon_directions: Vector3Array, sensor_photon_direction: Vector3
    ) -> Array:
        """
        Compute specularly reflected energy towards the sensor at the sea surface.

        Parameters
        ----------
        photon_directions : Vector3Array
            Directions of incident photons in water.
        sensor_photon_direction : Vector3
            Backward ray direction from the sensor.

        Returns
        -------
        Array
            Specular reflection energy multipliers per photon.
        """
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
        """
        Compute transmitted energy from water into air at the sea surface.

        Parameters
        ----------
        photon_directions : Vector3Array
            Directions of incident photons in water.
        sensor_photon_direction : Vector3
            Backward ray direction from the sensor.
        normal_direction : Vector3
            Surface normal direction (typically up or down).

        Returns
        -------
        Array
            Transmission energy multipliers per photon.
        """
        transmission = (
            microfacet_transmitted_energy(
                photon_directions,
                sensor_photon_direction,
                normal_direction,
                self._config.roughness,
                self._water_model.layers[0].refractive_index_at(0),
                1,
                self.base_reflectance,
            )
            + self._config.albedo / np.pi
        )

        return transmission

    def calculate_refraction_direction(
        self, incoming_directions: Vector3Array, normal: Vector3
    ) -> tuple[Vector3Array, BoolArray]:
        """
        Calculate refraction directions at the sea surface.

        Parameters
        ----------
        incoming_directions : Vector3Array
            Incident photon directions.
        normal : Vector3
            Surface normal direction.

        Returns
        -------
        tuple of (Vector3Array, BoolArray)
            Refracted directions and mask of total internal reflections.
        """
        return calculate_refraction_direction(
            incoming_directions,
            normal,
            1,
            self._water_model.layers[0].refractive_index_at(0),
        )
