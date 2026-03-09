import logging
from time import perf_counter
from typing import cast

import numpy as np

from alb_sim.config.simulation import SimulationConfig
from alb_sim.math.accumulate_heatmap import (
    accumulate_scatter_radius_heatmap,
    accumulate_to_heatmap,
)
from alb_sim.math.vector_math import dot_batch_single, length_batch
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.physics.models.simulation import SimulationModel
from alb_sim.physics.reflection.fresnel import fresnel_schlick
from alb_sim.physics.reflection.lambertian import heuristic_sample_batch
from alb_sim.utils.photon_position_state import PhotonPositionState
from alb_sim.utils.photon_wrapper import PhotonWrapper
from alb_sim.utils.types import Array, IntArray, Vector3Array

logger = logging.getLogger(__name__)


class Simulation:
    """Photon Monte Carlo simulation engine for forward and backward passes."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        """
        Initialise simulation state, geometry model, and photon storage.

        Parameters
        ----------
        config : SimulationConfig
            Global configuration for the scene, water column, laser, sensor, and outputs.
        rng : numpy.random.Generator
            Random number generator used for all stochastic sampling.
        """
        self.model = SimulationModel(config)
        self.rng = rng

        self.photon_storage = PhotonStorage()
        self.photon_maps: dict[PhotonType, PhotonMapIndex] = {}

        self.return_waveform: dict[PhotonType, Array] = {
            type: np.zeros(self.model.steps * 2, dtype=np.float32)
            for type in PhotonType
        }

        if self.model.heatmap_config.enabled:
            bins = self.model.heatmap_config.bins
            # Water surface heatmap (first_water_interaction)
            self.sampled_water_heatmap = np.zeros((bins, bins), dtype=np.float32)
            self._water_extent = self.model.heatmap_config.water_extent
            self._water_bin_size = 2 * self._water_extent / bins

            # Seafloor heatmap (seafloor_interaction)
            self.sampled_seafloor_heatmap = np.zeros((bins, bins), dtype=np.float32)
            self._seafloor_extent = self.model.heatmap_config.seafloor_extent
            self._seafloor_bin_size = 2 * self._seafloor_extent / bins

            # Center point for heatmaps
            self._heatmap_water_surface_center = self.model.water_surface_intersection
            self._heatmap_sea_floor_center = self.model.sea_floor_intersection
            self._heatmap_bins = bins

            # Scatter radius correlation heatmap (forward pass only)
            self.scatter_radius_heatmap = np.zeros((bins, bins), dtype=np.float32)

    def simulate_batch(self, num_photons: int, *, forward: bool):
        """Run a full batch of photons through all simulation time steps."""
        self.current_step = 0

        positions = np.zeros((num_photons, 3), dtype=np.float64)
        directions = self.model.sample_starting_direction(
            num_photons, self.rng, forward=forward
        )
        energies = np.full(num_photons, 1, dtype=np.float32)
        optical_depth = np.zeros(num_photons, dtype=np.float32)
        optical_depth_target = np.full(num_photons, np.inf, dtype=np.float32)
        time_deltas = self.model.get_emission_time_deltas(
            num_photons, self.rng, forward=forward
        )
        first_water_interaction = np.full((num_photons, 3), np.nan, dtype=np.float32)
        seafloor_interaction = np.full((num_photons, 3), np.nan, dtype=np.float32)

        wrapper = PhotonWrapper(
            positions,
            directions,
            energies,
            optical_depth,
            optical_depth_target,
            time_deltas,
            first_water_interaction,
            seafloor_interaction,
        )

        batch_start_time = perf_counter()

        for _ in range(self.model.steps * self.model.sample_multiplier):
            self.simulate_photon_step(wrapper, forward=forward)
            self.current_step += 1
            time_since_batch_start = perf_counter() - batch_start_time
            logger.debug(
                f"step {self.current_step} of {self.model.steps * self.model.sample_multiplier} in {time_since_batch_start:.2f} s = {(time_since_batch_start / 60):.2f} min, estimated remaining: {(time_since_batch_start / 60 / (self.current_step+1) * (self.model.steps * self.model.sample_multiplier - (self.current_step + 1))):.2f} min"
            )

    # ========================================
    # Photon Mapping
    # ========================================

    def store_photons(
        self,
        wrapper: PhotonWrapper,
        photon_type: PhotonType,
    ) -> None:
        """Append photon state arrays to storage for later photon mapping."""
        self.photon_storage.positions[photon_type].append(wrapper.positions.copy())
        self.photon_storage.directions[photon_type].append(wrapper.directions.copy())
        self.photon_storage.energies[photon_type].append(wrapper.energies.copy())
        self.photon_storage.times[photon_type].append(wrapper.time_deltas.copy())
        if wrapper.first_water_interaction is not None:
            self.photon_storage.first_water_interaction[photon_type].append(
                wrapper.first_water_interaction.copy()
            )
        if wrapper.seafloor_interaction is not None:
            self.photon_storage.seafloor_interaction[photon_type].append(
                wrapper.seafloor_interaction.copy()
            )

    def sample_photons(self, wrapper: PhotonWrapper, photon_type: PhotonType) -> None:
        """Query the photon map via KDTree and accumulate energy into waveforms and heatmaps."""
        photon_map = self.photon_maps[photon_type]

        for sensor_position, sensor_direction, sensor_energy, sensor_time_step in zip(
            wrapper.positions, wrapper.directions, wrapper.energies, wrapper.time_deltas
        ):
            dist, idx = cast(
                tuple[Array, IntArray],
                photon_map.tree.query(sensor_position, k=self.model.photon_mapping_k),
            )
            photon_direction: Vector3Array = photon_map.data.directions[idx]
            photon_energy: Array = photon_map.data.energies[idx]
            photon_time_step: Array = photon_map.data.times[idx]

            # Get interaction positions from the sampled photons
            photon_first_water_interaction: Vector3Array = (
                photon_map.data.first_water_interaction[idx]
            )
            photon_sea_floor_interaction: Vector3Array = (
                photon_map.data.seafloor_interaction[idx]
            )

            if photon_type == PhotonType.BOTTOM_REFLECTION:
                # lambertian reflection, energy only depends on angle of sensor photon direction to normal
                cos_term = np.maximum(
                    0.0, np.dot(np.array([0, 1, 0]), -sensor_direction)
                )

                energy_multiplier = (self.model.seafloor_albedo / np.pi) * cos_term

                kernel_size = np.pi * dist[-1] ** 2
            elif photon_type == PhotonType.SCATTER:
                # Vector from scatter point to sensor (i.e. reverse of backward ray)
                view_dir = -sensor_direction

                energy_multiplier = self.model.water.scatter_energy(
                    sensor_position, photon_direction, view_dir
                )

                kernel_size = (4.0 / 3.0) * np.pi * dist[-1] ** 3
            elif photon_type == PhotonType.SURFACE_REFLECTION:
                energy_multiplier = self.model.sea_surface.reflected_energy(
                    -photon_direction,
                    -sensor_direction,
                )

                kernel_size = np.pi * dist[-1] ** 2
            elif photon_type == PhotonType.SURFACE_TRANSMISSION_UP:
                energy_multiplier = self.model.sea_surface.transmitted_energy(
                    -photon_direction, -sensor_direction, np.array([0, 1, 0])
                )

                kernel_size = np.pi * dist[-1] ** 2

            kernel_norm = 1.0 / kernel_size

            store_energy = (
                sensor_energy * photon_energy * kernel_norm * energy_multiplier
            )
            store_time = sensor_time_step + photon_time_step
            sample_idx = (store_time / self.model.sample_multiplier).astype(int)

            np.add.at(self.return_waveform[photon_type], sample_idx, store_energy)

            # Accumulate to heatmaps if enabled
            if self.model.heatmap_config.enabled:
                accumulate_to_heatmap(
                    photon_first_water_interaction,
                    store_energy,
                    self.sampled_water_heatmap,
                    self._heatmap_water_surface_center,
                    self._water_extent,
                    self._water_bin_size,
                )
                accumulate_to_heatmap(
                    photon_sea_floor_interaction,
                    store_energy,
                    self.sampled_seafloor_heatmap,
                    self._heatmap_sea_floor_center,
                    self._seafloor_extent,
                    self._seafloor_bin_size,
                )

    # ========================================
    # Step Function
    # ========================================

    def evaluate_state(self, y_current: Array, y_next: Array) -> IntArray:
        """Classify photons by position state (air, entering/exiting water, seafloor)."""
        states = np.full_like(
            y_current, PhotonPositionState.IN_AIR.value, dtype=np.int32
        )

        air_to_water = (y_current > self.model.water_surface_y) & (
            y_next <= self.model.water_surface_y
        )
        in_water = (y_current <= self.model.water_surface_y) & (
            y_next > self.model.seafloor_y
        )
        water_to_floor = (y_current > self.model.seafloor_y) & (
            y_next <= self.model.seafloor_y
        )
        water_to_air = (y_current <= self.model.water_surface_y) & (
            y_next > self.model.water_surface_y
        )

        states[air_to_water] = PhotonPositionState.ENTERING_WATER.value
        states[in_water] = PhotonPositionState.IN_WATER.value
        states[water_to_floor] = PhotonPositionState.HITTING_SEAFLOOR.value
        states[water_to_air] = PhotonPositionState.EXITING_WATER.value

        return states

    def simulate_photon_step(self, wrapper: PhotonWrapper, *, forward: bool):
        """Advance all photons by one time step and dispatch to interaction handlers."""
        next_positions = (
            wrapper.positions
            + wrapper.directions
            * self.model.velocities_by_position(wrapper.positions)[:, np.newaxis]
            * self.model.time_step
        )

        states = self.evaluate_state(wrapper.positions[:, 1], next_positions[:, 1])

        air_idx = np.nonzero(states == PhotonPositionState.IN_AIR.value)[0]
        enter_idx = np.nonzero(states == PhotonPositionState.ENTERING_WATER.value)[0]
        water_idx = np.nonzero(states == PhotonPositionState.IN_WATER.value)[0]
        seafloor_idx = np.nonzero(states == PhotonPositionState.HITTING_SEAFLOOR.value)[
            0
        ]
        exit_idx = np.nonzero(states == PhotonPositionState.EXITING_WATER.value)[0]

        # Air - Air
        wrapper.positions[air_idx] = next_positions[air_idx]

        # Water - Water
        if water_idx.size > 0:
            water_subset = wrapper.subset(water_idx)
            water_subset = self.handle_water(
                water_subset, next_positions[water_idx], forward=forward
            )
            wrapper.positions[water_idx] = water_subset.positions
            wrapper.directions[water_idx] = water_subset.directions
            wrapper.energies[water_idx] = water_subset.energies
            wrapper.optical_depth[water_idx] = water_subset.optical_depth
            wrapper.optical_depth_target[water_idx] = water_subset.optical_depth_target

        # Air - Water
        if enter_idx.size > 0:
            enter_subset = wrapper.subset(enter_idx)
            enter_subset = self.handle_enter(
                enter_subset, next_positions[enter_idx], forward=forward
            )
            wrapper.positions[enter_idx] = enter_subset.positions
            wrapper.directions[enter_idx] = enter_subset.directions
            wrapper.energies[enter_idx] = enter_subset.energies
            wrapper.optical_depth[enter_idx] = enter_subset.optical_depth
            wrapper.optical_depth_target[enter_idx] = enter_subset.optical_depth_target
            wrapper.first_water_interaction[enter_idx] = (
                enter_subset.first_water_interaction
            )

        # Water - Air
        if exit_idx.size > 0:
            exit_subset = wrapper.subset(exit_idx)
            exit_subset = self.handle_exit(
                exit_subset, next_positions[exit_idx], forward=forward
            )
            wrapper.positions[exit_idx] = exit_subset.positions
            wrapper.directions[exit_idx] = exit_subset.directions
            wrapper.energies[exit_idx] = exit_subset.energies
            wrapper.optical_depth[exit_idx] = exit_subset.optical_depth
            wrapper.optical_depth_target[exit_idx] = exit_subset.optical_depth_target

        # Water - Seafloor - Water
        if seafloor_idx.size > 0:
            seafloor_subset = wrapper.subset(seafloor_idx)
            seafloor_subset = self.handle_seafloor(
                seafloor_subset, next_positions[seafloor_idx], forward=forward
            )
            wrapper.positions[seafloor_idx] = seafloor_subset.positions
            wrapper.directions[seafloor_idx] = seafloor_subset.directions
            wrapper.energies[seafloor_idx] = seafloor_subset.energies
            wrapper.optical_depth[seafloor_idx] = seafloor_subset.optical_depth
            wrapper.optical_depth_target[seafloor_idx] = (
                seafloor_subset.optical_depth_target
            )
            wrapper.seafloor_interaction[seafloor_idx] = (
                seafloor_subset.seafloor_interaction
            )

    # ========================================
    # Interaction Handlers
    # ========================================

    def handle_water(
        self, subset: PhotonWrapper, next_positions: Vector3Array, *, forward: bool
    ) -> PhotonWrapper:
        """
        Handle scattering and general traversing in water.

        Returns:
            New PhotonWrapper containing positions, directions, energies, optical depth and optical depth targets
        """

        step_vectors = next_positions - subset.positions
        step_lengths = np.linalg.norm(step_vectors, axis=1)

        # optical depth is always increased, if target is reached it is reset and new target is created
        subset.optical_depth += (
            self.model.lidar_attenuation_coefficients_by_position(subset.positions)
            * step_lengths
        )

        scatter_occurs = subset.optical_depth >= subset.optical_depth_target

        # all photons advance to next position, no need to backtrack, resolution is ~1.1cm
        subset.positions = next_positions

        # --- Scatter case
        if np.any(scatter_occurs):
            idx = np.nonzero(scatter_occurs)[0]
            scatter_subset = subset.subset(idx)
            scatter_subset.time_deltas += self.current_step
            if forward:
                self.store_photons(scatter_subset, PhotonType.SCATTER)
            else:
                self.sample_photons(scatter_subset, PhotonType.SCATTER)

            subset.directions[idx] = (
                self.model.sample_scattering_directions_by_position(
                    subset.positions[idx], subset.directions[idx], self.rng
                )
            )
            subset.energies[idx] *= self.model.single_scattering_albedos_by_position(
                subset.positions[idx]
            )
            subset.optical_depth[idx] -= subset.optical_depth_target[idx]

            # Resample new optical depth targets
            rand_vals = self.rng.random(len(idx)).astype(np.float32)
            subset.optical_depth_target[idx] = -np.log(rand_vals)

        return subset

    def handle_enter(
        self, subset: PhotonWrapper, next_positions: Vector3Array, *, forward: bool
    ) -> PhotonWrapper:
        """Handle photon entry from air to water: Fresnel splitting and Snell refraction."""

        # Calculate intersection with sea surface
        step_vectors = next_positions - subset.positions
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection_fraction = (
                self.model.water_surface_y - subset.positions[:, 1]
            ) / step_vectors[:, 1]
            intersection_fraction = np.clip(intersection_fraction, 0.0, 1.0)

        intersection_points = (
            subset.positions + intersection_fraction[:, np.newaxis] * step_vectors
        )

        # Record first water interaction position
        subset.first_water_interaction = intersection_points.copy()

        # Refraction calculation, reverse direction away from intersection point
        cos_incidence = dot_batch_single(-subset.directions, np.array([0, 1, 0]))

        reflection_weight = fresnel_schlick(
            cos_incidence, self.model.sea_surface.base_reflectance
        )
        transmission_weight = 1.0 - reflection_weight

        reflection_subset = subset.copy()
        reflection_subset.positions = intersection_points
        reflection_subset.time_deltas += self.current_step
        if forward:
            self.store_photons(reflection_subset, PhotonType.SURFACE_REFLECTION)
        else:
            self.sample_photons(reflection_subset, PhotonType.SURFACE_REFLECTION)
            self.sample_photons(reflection_subset, PhotonType.SURFACE_TRANSMISSION_UP)

        # Refraction direction
        subset.directions, _ = self.model.sea_surface.calculate_refraction_direction(
            subset.directions, np.array([0, 1, 0])
        )
        subset.energies *= transmission_weight  # transmitted energy continues
        # (Reflected part is captured via stored photon for backward pass)

        # Continue step into medium
        step_lengths = length_batch(step_vectors)
        remaining_fraction = 1.0 - intersection_fraction
        remaining_step = subset.directions * (
            remaining_fraction[:, np.newaxis] * step_lengths[:, np.newaxis]
        )
        subset.positions = intersection_points + remaining_step

        # Set optical depth targets
        rand_vals = self.rng.random(len(intersection_points)).astype(np.float32)
        subset.optical_depth_target = -np.log(rand_vals)

        return subset

    def handle_exit(
        self, subset: PhotonWrapper, next_positions: Vector3Array, *, forward: bool
    ) -> PhotonWrapper:
        """Handle photon exit from water to air: Fresnel reflection/transmission and refraction."""

        # Calculate intersection with sea surface
        step_vectors = next_positions - subset.positions
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection_fraction = (
                self.model.water_surface_y - subset.positions[:, 1]
            ) / step_vectors[:, 1]
            intersection_fraction = np.clip(intersection_fraction, 0.0, 1.0)

        intersection_points = (
            subset.positions + intersection_fraction[:, np.newaxis] * step_vectors
        )

        normal_direction = np.array([0, -1, 0])

        # Refraction calculation, reverse direction away from intersection point
        cos_incidence = dot_batch_single(-subset.directions, normal_direction)

        reflection_weight = fresnel_schlick(
            cos_incidence, self.model.sea_surface.base_reflectance
        )
        reflection_mask = (
            self.rng.random(size=reflection_weight.shape) < reflection_weight
        )

        # photon_mapping
        transmission_subset = subset.subset(~reflection_mask)
        transmission_subset.positions = intersection_points[~reflection_mask]
        transmission_subset.time_deltas += self.current_step
        if forward:
            self.store_photons(transmission_subset, PhotonType.SURFACE_TRANSMISSION_UP)

        # Refraction and reflection direction
        refracted_directions, total_internal_reflections_mask = (
            self.model.sea_surface.calculate_refraction_direction(
                subset.directions, normal_direction
            )
        )
        reflection_mask |= total_internal_reflections_mask
        reflected_directions = (
            subset.directions
            - 2
            * dot_batch_single(subset.directions, normal_direction)[:, np.newaxis]
            * normal_direction
        )
        subset.directions = np.where(
            reflection_mask[:, np.newaxis], reflected_directions, refracted_directions
        )

        # Continue step
        step_lengths = length_batch(step_vectors)
        remaining_fraction = 1.0 - intersection_fraction
        remaining_step = subset.directions * (
            remaining_fraction[:, np.newaxis] * step_lengths[:, np.newaxis]
        )
        subset.positions = intersection_points + remaining_step

        # continue optical depth on reflection
        subset.optical_depth[reflection_mask] += (
            self.model.lidar_attenuation_coefficients_by_position(
                subset.positions[reflection_mask]
            )
            * step_lengths[reflection_mask]
        )

        return subset

    def handle_seafloor(
        self, subset: PhotonWrapper, next_positions: Vector3Array, *, forward: bool
    ) -> PhotonWrapper:
        """Handle Lambertian reflection at the seafloor boundary."""
        # Calculate intersection with seafloor
        step_vectors = next_positions - subset.positions
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection_fraction = (
                self.model.seafloor_y - subset.positions[:, 1]
            ) / step_vectors[:, 1]
            intersection_fraction = np.clip(intersection_fraction, 0.0, 1.0)

        intersection_points = (
            subset.positions + intersection_fraction[:, np.newaxis] * step_vectors
        )

        # Record seafloor interaction position
        subset.seafloor_interaction = intersection_points.copy()

        # Accumulate radius correlation heatmap (forward pass only)
        if (
            forward
            and self.model.heatmap_config.enabled
            and subset.first_water_interaction is not None
        ):
            accumulate_scatter_radius_heatmap(
                self.scatter_radius_heatmap,
                subset.first_water_interaction,
                intersection_points,
                self._heatmap_water_surface_center,
                self._heatmap_sea_floor_center,
                self._water_extent,
                self._seafloor_extent,
            )

        # Photon Mapping
        reflection_subset = subset.copy()
        reflection_subset.positions = intersection_points
        reflection_subset.time_deltas += self.current_step
        if forward:
            self.store_photons(reflection_subset, PhotonType.BOTTOM_REFLECTION)
        else:
            self.sample_photons(reflection_subset, PhotonType.BOTTOM_REFLECTION)

        # Sample reflection direction
        normal_direction = np.array([0, 1, 0])
        subset.directions = heuristic_sample_batch(
            len(intersection_points), normal_direction, self.rng
        )

        # Continue step
        step_lengths = length_batch(step_vectors)
        remaining_fraction = 1.0 - intersection_fraction
        remaining_step = subset.directions * (
            remaining_fraction[:, np.newaxis] * step_lengths[:, np.newaxis]
        )
        subset.positions = intersection_points + remaining_step

        # continue optical depth on reflection
        subset.optical_depth += (
            self.model.lidar_attenuation_coefficients_by_position(subset.positions)
            * step_lengths
        )

        subset.energies *= self.model.seafloor_albedo

        return subset
