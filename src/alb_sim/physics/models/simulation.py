import numpy as np

from alb_sim.config.simulation import SimulationConfig
from alb_sim.math.direction_cone import (
    sample_directions_in_cone_gaussian,
    sample_directions_in_cone_uniform,
)
from alb_sim.math.disk_sampling import sample_disk_points
from alb_sim.physics.constants import EPSILON, LIGHT_SPEED_AIR
from alb_sim.physics.emission.times import (
    sample_gaussian_pulse_batch,
    time_offset_to_steps,
)
from alb_sim.physics.models.laser import LaserModel
from alb_sim.physics.models.sea_surface import SeaSurfaceModel
from alb_sim.physics.models.water import WaterModel
from alb_sim.utils.types import Array, Vector3, Vector3Array


class SimulationModel:
    def __init__(self, config: SimulationConfig):
        self._config = config

        self.laser = LaserModel(self._config.laser)
        self.water = WaterModel(
            self._config.water, self._config.laser, self._config.scene
        )
        self.sea_surface = SeaSurfaceModel(self._config.sea_surface, self.water)

        self.effective_sample_rate = (
            self._config.sample_multiplier * self._config.sensor.sample_rate
        )
        self.time_step = 1 / self.effective_sample_rate

        self.steps = self._calculate_steps()

    def _calculate_steps(self) -> int:
        distance = self.seafloor_y / self.laser.direction[1]
        # print(distance)
        return int(1.5 * (distance * round(self.sample_rate)) / LIGHT_SPEED_AIR)

    @property
    def sample_rate(self) -> int:
        return self._config.sensor.sample_rate

    @property
    def sample_multiplier(self) -> int:
        return self._config.sample_multiplier

    @property
    def photon_mapping_k(self) -> int:
        return self._config.photon_mapping_k

    @property
    def water_surface_y(self) -> float:
        return -self._config.scene.flying_height

    @property
    def seafloor_y(self) -> float:
        return -(self._config.scene.flying_height + self.water.depth)

    @property
    def seafloor_albedo(self) -> float:
        return self._config.sea_floor.albedo

    @property
    def heatmap_config(self):
        """Return the heatmap configuration."""
        return self._config.heatmap

    # ========================================
    # Intersection Points
    # ========================================

    @property
    def water_surface_intersection(self) -> Vector3:
        """
        Compute the intersection position of the laser beam with the water surface.
        Returns the (x, y, z) intersection point on the water surface (y = water_surface_y).
        """
        # Laser origin at (0, 0, 0)
        laser_direction = self.laser.direction

        water_surface_interaction_point = (
            self.water_surface_y / laser_direction[1]
        ) * laser_direction

        return water_surface_interaction_point

    @property
    def sea_floor_intersection(self) -> Vector3:
        """
        Compute the intersection position of the laser beam with the sea floor.
        Returns the (x, y, z) intersection point on the sea floor (y = seafloor_y).
        """
        # Laser origin at (0, 0, 0)
        laser_direction = self.laser.direction
        refraction_direction = self.sea_surface.calculate_refraction_direction(
            np.array([laser_direction]), np.array([0, 1, 0])
        )[0][0]

        water_surface_interaction_point = (
            self.water_surface_y / laser_direction[1]
        ) * laser_direction
        seafloor_interaction_point = (
            water_surface_interaction_point
            + (-self.water.depth / refraction_direction[1]) * refraction_direction
        )

        return seafloor_interaction_point

    # ========================================
    # Emission
    # ========================================

    def sample_starting_direction(
        self, num_samples: int, rng: np.random.Generator, *, forward: bool
    ) -> Vector3Array:
        cone_direction = self.laser.direction
        cone_half_angle = (
            self.laser.divergence_angle
            if forward
            else self._config.sensor.field_of_view
        )

        origin_point = (
            np.array([0, 0, 0], dtype=np.float64)
            if forward
            else sample_disk_points(
                cone_direction,
                self._config.sensor.aperture_radius,
                np.array([0, 0, 0]),
                num_samples,
                rng,
            )
        )

        return origin_point + (
            sample_directions_in_cone_gaussian(
                cone_direction, cone_half_angle, num_samples, rng
            )
            if forward
            else sample_directions_in_cone_uniform(
                cone_direction, cone_half_angle, num_samples, rng
            )
        )

    def get_emission_time_deltas(
        self, num_samples: int, rng: np.random.Generator, *, forward: bool
    ) -> Array:
        if forward:
            return time_offset_to_steps(
                sample_gaussian_pulse_batch(num_samples, self.laser.pulse_fwhm, rng),
                self.effective_sample_rate,
            )
        else:
            return np.zeros(num_samples, dtype=np.float32)

    # ========================================
    # General
    # ========================================

    def velocities_by_position(self, positions: Vector3Array) -> Array:
        y: Array = positions[:, 1]
        velocities = np.full(len(positions), EPSILON)

        mask_air = y > self.water_surface_y  # laser and sensor at (y=0)
        if np.any(mask_air):
            velocities[mask_air] = LIGHT_SPEED_AIR
        if np.any(~mask_air):
            # print("y shape", y.shape)
            # print("mask shape", mask_air.shape)
            # print("velocities shape", velocities.shape)
            velocities[~mask_air] = self.water.velocities_by_depth(
                (y[~mask_air] - self.water_surface_y) * -1
            )

        return velocities.astype(np.float64)

    # ========================================
    # Scattering
    # ========================================

    def lidar_attenuation_coefficients_by_position(
        self, positions: Vector3Array
    ) -> Array:
        y: Array = positions[:, 1]
        return self.water.lidar_attenuation_coefficients_by_depth(
            (y - self.water_surface_y) * -1
        )

    def single_scattering_albedos_by_position(self, positions: Vector3Array) -> Array:
        y: Array = positions[:, 1]
        return self.water.single_scattering_albedos_by_depth(
            (y - self.water_surface_y) * -1
        )

    def sample_scattering_directions_by_position(
        self,
        positions: Vector3Array,
        incoming_directions: Vector3Array,
        rng: np.random.Generator,
    ) -> Array:
        y: Array = positions[:, 1]
        return self.water.sample_scattering_directions_by_depth(
            (y - self.water_surface_y) * -1, incoming_directions, rng
        )
