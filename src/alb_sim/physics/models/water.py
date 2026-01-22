import math

import numpy as np

from alb_sim.config.laser import LaserConfig
from alb_sim.config.scene import SceneConfig
from alb_sim.config.water import FournierForandConfig, TurbidityLayerConfig, WaterConfig
from alb_sim.physics.constants import EPSILON, LIGHT_SPEED_AIR
from alb_sim.physics.scatter.direction import sample_scattering_directions_batch
from alb_sim.physics.scatter.energy import calculate_energy_batch
from alb_sim.physics.scatter.fournier_forand_phase_function import (
    calculate_backscatter_fraction,
    calculate_phase_function,
)
from alb_sim.utils.types import Array, IntArray, Vector3, Vector3Array


class FournierForandModel:
    def __init__(self, config: FournierForandConfig):
        self._config = config
        self.theta, self.pf, self.cdf = calculate_phase_function(
            self._config.junge_slope, self._config.refractive_index_ratio
        )
        self.backscatter_fraction = calculate_backscatter_fraction(
            self._config.junge_slope, self._config.refractive_index_ratio
        )


class TurbidityLayerModel:
    def __init__(
        self,
        config: TurbidityLayerConfig,
        laser_config: LaserConfig,
        scene_config: SceneConfig,
    ):
        self._config = config
        self._laser_config = laser_config
        self._scene_config = scene_config
        self.fournier_forand_model = FournierForandModel(
            self._config.fournier_forand_parameters
        )

        self.light_speed = self._calculate_light_speed()
        self.lidar_attenuation_coefficient = (
            self._calculate_lidar_attenuation_coefficient()
        )
        self.single_scattering_albedo = self._calculate_single_scattering_albedo()

    def _calculate_light_speed(self) -> float:
        return LIGHT_SPEED_AIR / self._config.refractive_index

    def _calculate_lidar_attenuation_coefficient(self) -> float:
        seawater_molecular_scattering_coefficient: float = (
            (1 + 0.008027 * self._config.salinity)
            * 0.00012
            * self._laser_config.wavelength ** (-4.24)
        )
        particle_scattering_coefficient = (
            self._config.scattering_coefficient
            - seawater_molecular_scattering_coefficient
        )
        print("particle scattering", particle_scattering_coefficient)
        laser_spot_diameter_surface = (
            2
            * self._scene_config.flying_height
            * math.tan(self._laser_config.divergence_angle / 2)
        )
        print("laser spot diameter", laser_spot_diameter_surface)

        # Diffuse attenuation coefficient (Churnside, 2014)
        # See eq. 7, "Review of profiling oceanographic lidar", Churnside 12/2013
        kd = (
            self._config.absorption_coefficient
            + 4.18
            * particle_scattering_coefficient  # TODO: Shouldn't be here?
            * self.fournier_forand_model.backscatter_fraction
            * (
                1 - 0.52 * math.exp(-10.8 * self._config.absorption_coefficient)
            )  # TODO: Should be +0.52?
        )
        print("kd", kd)

        # For ALB systems if IFOV is greater than 10 mrad, appropriate attenuation coefficient is the diffuse attenuation coefficient K_d
        # If the IFOV is narrower than 10 mrad, the appropriate attenuation coefficient will be the beam attenuation coefficient c
        # LiDAR attenuation coefficient alpha is used to include both cases
        # See eq. 5, "Review of profiling oceanographic lidar", Churnside 12/2013
        lidar_attenuation_coefficient = kd + (
            particle_scattering_coefficient
            - 4.18
            * particle_scattering_coefficient
            * self.fournier_forand_model.backscatter_fraction
            * (
                1 - 0.52 * math.exp(-10.8 * self._config.scattering_coefficient)
            )  # TODO: here is scattering coefficient instead of absorption coefficient? I think that is wrong
        ) * math.exp(
            -0.85
            * laser_spot_diameter_surface
            * (self._config.absorption_coefficient + particle_scattering_coefficient)
        )

        return lidar_attenuation_coefficient

    def _calculate_single_scattering_albedo(self) -> float:
        return (
            self.lidar_attenuation_coefficient - self._config.absorption_coefficient
        ) / self.lidar_attenuation_coefficient

    @property
    def height(self) -> float:
        return self._config.height


class WaterModel:
    def __init__(
        self, config: WaterConfig, laser_config: LaserConfig, scene_config: SceneConfig
    ):
        self._config = config
        self._scene_config = scene_config
        self.layers = tuple(
            TurbidityLayerModel(layer, laser_config, self._scene_config)
            for layer in self._config.layers
        )
        self._calculate_heights()

    def _calculate_heights(self):
        self.layer_height_cumsum = np.cumsum([layer.height for layer in self.layers])
        self.depth = self.layer_height_cumsum[-1]

    def layer_index(self, depths: Array | float) -> IntArray | int:
        return np.searchsorted(self.layer_height_cumsum, depths, side="right").astype(
            np.int32
        )

    def velocities_by_depth(self, depths: Array) -> Array:
        velocities = np.full_like(depths, EPSILON)
        idx = self.layer_index(depths)
        for layer in range(len(self.layers)):
            mask = idx == layer
            velocities[mask] = self.layers[layer].light_speed
        return velocities

    def scatter_energy(
        self,
        position: Vector3,
        photon_directions: Vector3Array,
        sensor_photon_direction: Vector3,
    ) -> Array:
        photon_depth = (position[1] + self._scene_config.flying_height) * -1
        layer: TurbidityLayerModel = self.layers[self.layer_index(photon_depth)]
        return calculate_energy_batch(
            photon_directions,
            sensor_photon_direction,
            layer.fournier_forand_model.theta,
            layer.fournier_forand_model.cdf,
        )

    def lidar_attenuation_coefficients_by_depth(self, depths: Array) -> Array:
        lidar_attenuation_coefficients = np.full_like(depths, EPSILON, dtype=np.float32)
        idx = self.layer_index(depths)
        for layer in range(len(self.layers)):
            mask = idx == layer
            lidar_attenuation_coefficients[mask] = self.layers[
                layer
            ].lidar_attenuation_coefficient
        return lidar_attenuation_coefficients

    def single_scattering_albedos_by_depth(self, depths: Array) -> Array:
        single_scattering_albedos = np.full_like(depths, EPSILON, dtype=np.float32)
        idx = self.layer_index(depths)
        for layer in range(len(self.layers)):
            mask = idx == layer
            single_scattering_albedos[mask] = self.layers[
                layer
            ].single_scattering_albedo
        return single_scattering_albedos

    def sample_scattering_directions_by_depth(
        self, depths: Array, incoming_directions: Vector3Array, rng: np.random.Generator
    ) -> Vector3Array:
        directions = np.empty_like(incoming_directions, dtype=np.float32)
        idx = self.layer_index(depths)
        for layer in range(len(self.layers)):
            mask = idx == layer
            directions[mask] = sample_scattering_directions_batch(
                self.layers[layer].fournier_forand_model.theta,
                self.layers[layer].fournier_forand_model.cdf,
                incoming_directions,
                rng,
            )
        return directions
