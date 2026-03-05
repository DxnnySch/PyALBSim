import math
from typing import Union, overload

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
from alb_sim.utils.parameter_profile import NumberOrScalar
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

        self._build_lidar_attenuation_lut()

        # self.light_speed = self._calculate_light_speed()
        # self.lidar_attenuation_coefficient = (
        #     self._calculate_lidar_attenuation_coefficient()
        # )
        # self.single_scattering_albedo = self._calculate_single_scattering_albedo()

    def _is_constant_layer(self) -> bool:
        return all(
            isinstance(v, float)
            for v in (
                self._config.scattering_coefficient,
                self._config.absorption_coefficient,
                self._config.salinity,
                self._config.refractive_index,
            )
        )

    def _build_lidar_attenuation_lut(self):
        if self._is_constant_layer():
            self._constant_alpha = self._calculate_lidar_attenuation_coefficient(0.0)
            self.lidar_attenuation_coefficient_at = lambda z: self._constant_alpha
            return

        self._z_lut = np.linspace(
            0.0,
            self._config.height,
            self._config.num_sublayers,
            dtype=np.float32,
        )

        self._lidar_alpha_lut = self._calculate_lidar_attenuation_coefficient(
            self._z_lut
        ).astype(np.float32)

    def _eval(
        self, value: NumberOrScalar, z_local: Union[Array, float]
    ) -> Union[Array, float]:
        if isinstance(value, float):
            return value
        return value.at(z_local / self._config.height)

    def _calculate_lidar_attenuation_coefficient(
        self, z_local: Union[Array, float]
    ) -> Union[Array, float]:
        seawater_molecular_scattering_coefficient: Union[Array, float] = (
            (1 + 0.008027 * self._eval(self._config.salinity, z_local))
            * 0.00012
            * self._laser_config.wavelength ** (-4.24)
        )
        particle_scattering_coefficient = (
            self._eval(self._config.scattering_coefficient, z_local)
            - seawater_molecular_scattering_coefficient
        )

        laser_spot_diameter_surface = (
            2
            * self._scene_config.flying_height
            * math.tan(self._laser_config.divergence_angle / 2)
        )

        # Diffuse attenuation coefficient (Churnside, 2014)
        # See eq. 7, "Review of profiling oceanographic lidar", Churnside 12/2013
        kd = self._eval(
            self._config.absorption_coefficient, z_local
        ) + 4.18 * particle_scattering_coefficient * self.fournier_forand_model.backscatter_fraction * (  # TODO: Shouldn't be here?
            1
            - 0.52
            * np.exp(-10.8 * self._eval(self._config.absorption_coefficient, z_local))
        )  # TODO: Should be +0.52?

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
                1
                - 0.52
                * np.exp(
                    -10.8 * self._eval(self._config.scattering_coefficient, z_local)
                )
            )  # TODO: here is scattering coefficient instead of absorption coefficient? I think that is wrong
        ) * np.exp(
            -0.85
            * laser_spot_diameter_surface
            * (
                self._eval(self._config.absorption_coefficient, z_local)
                + particle_scattering_coefficient
            )
        )

        return lidar_attenuation_coefficient

    @overload
    def light_speed_at(self, z_local: Array) -> Array: ...

    @overload
    def light_speed_at(self, z_local: float) -> float: ...

    def light_speed_at(self, z_local: Union[Array, float]) -> Union[Array, float]:
        refractive_index = self._eval(self._config.refractive_index, z_local)
        return LIGHT_SPEED_AIR / refractive_index

    @overload
    def lidar_attenuation_coefficient_at(self, z_local: Array) -> Array: ...

    @overload
    def lidar_attenuation_coefficient_at(self, z_local: float) -> float: ...

    def lidar_attenuation_coefficient_at(
        self, z_local: Union[Array, float]
    ) -> Union[Array, float]:
        z = np.asarray(z_local, dtype=np.float32)

        alpha = np.interp(
            z,
            self._z_lut,
            self._lidar_alpha_lut,
            left=self._lidar_alpha_lut[0],
            right=self._lidar_alpha_lut[-1],
        )

        if np.isscalar(z_local):
            return float(alpha)

        return alpha

    @overload
    def single_scattering_albedo_at(self, z_local: Array) -> Array: ...

    @overload
    def single_scattering_albedo_at(self, z_local: float) -> float: ...

    def single_scattering_albedo_at(
        self, z_local: Union[Array, float]
    ) -> Union[Array, float]:
        lidar_attenuation_coefficient = self.lidar_attenuation_coefficient_at(z_local)
        return (
            lidar_attenuation_coefficient
            - self._eval(self._config.absorption_coefficient, z_local)
        ) / lidar_attenuation_coefficient

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

    @overload
    def layer_index(self, depths: Array) -> IntArray: ...

    @overload
    def layer_index(self, depths: float) -> int: ...

    def layer_index(self, depths: Union[Array, float]) -> Union[IntArray, int]:
        idx = np.searchsorted(self.layer_height_cumsum, depths, side="right")
        return np.minimum(idx, len(self.layers) - 1).astype(np.int32)

    @overload
    def layer_local_depth(self, depths: Array) -> Array: ...

    @overload
    def layer_local_depth(self, depths: float) -> float: ...

    def layer_local_depth(self, depths: Union[Array, float]) -> Union[Array, float]:
        idx = self.layer_index(depths)
        z0 = np.concatenate(([0.0], self.layer_height_cumsum[:-1])).astype(np.float32)
        # print(idx, z0)
        return depths - z0[idx]

    #
    #
    #

    def velocities_by_depth(self, depths: Array) -> Array:
        velocities = np.full_like(depths, EPSILON)
        idx = self.layer_index(depths)
        z_local = self.layer_local_depth(depths)

        for layer_idx, layer in enumerate(self.layers):
            mask = idx == layer_idx
            if np.any(mask):
                velocities[mask] = layer.light_speed_at(z_local[mask])
        return velocities

    def lidar_attenuation_coefficients_by_depth(self, depths: Array) -> Array:
        lidar_attenuation_coefficients = np.full_like(depths, EPSILON, dtype=np.float32)
        idx = self.layer_index(depths)
        z_local = self.layer_local_depth(depths)

        for layer_idx, layer in enumerate(self.layers):
            mask = idx == layer_idx
            if np.any(mask):
                lidar_attenuation_coefficients[mask] = (
                    layer.lidar_attenuation_coefficient_at(z_local[mask])
                )
        return lidar_attenuation_coefficients

    def single_scattering_albedos_by_depth(self, depths: Array) -> Array:
        single_scattering_albedos = np.full_like(depths, EPSILON, dtype=np.float32)
        idx = self.layer_index(depths)
        z_local = self.layer_local_depth(depths)

        for layer_idx, layer in enumerate(self.layers):
            mask = idx == layer_idx
            if np.any(mask):
                single_scattering_albedos[mask] = layer.single_scattering_albedo_at(
                    z_local[mask]
                )
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
                incoming_directions[mask],
                rng,
            )
        return directions

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
