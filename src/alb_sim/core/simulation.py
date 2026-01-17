from time import perf_counter
import numpy as np
import logging

logger = logging.getLogger(__name__)

from alb_sim.config.simulation import SimulationConfig
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


class Simulation:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.model = SimulationModel(config)
        self.rng = rng

        self.photon_storage = PhotonStorage()
        self.photon_maps: dict[PhotonType, PhotonMapIndex] = {}

        self.return_waveform: dict[PhotonType, Array] = {
            type: np.zeros(self.model.steps * 2, dtype=np.float32)
            for type in PhotonType
        }

    def simulate_batch(self, num_photons: int, *, forward: bool):
        self.current_step = 0

        positions = np.zeros((num_photons, 3), dtype=np.float32)
        directions = self.model.sample_vector_direction_in_cone(
            num_photons, self.rng, forward=forward
        )
        energies = np.full(num_photons, 1, dtype=np.float32)
        optical_depth = np.zeros(num_photons, dtype=np.float32)
        optical_depth_target = np.full(num_photons, np.inf, dtype=np.float32)
        time_deltas = self.model.get_emission_time_deltas(
            num_photons, self.rng, forward=forward
        )

        wrapper = PhotonWrapper(
            positions,
            directions,
            energies,
            optical_depth,
            optical_depth_target,
            time_deltas,
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
        self.photon_storage.positions[photon_type].append(wrapper.positions)
        self.photon_storage.directions[photon_type].append(wrapper.directions)
        self.photon_storage.energies[photon_type].append(wrapper.energies)
        self.photon_storage.times[photon_type].append(wrapper.time_deltas)

    def sample_photons(self, wrapper: PhotonWrapper, photon_type: PhotonType) -> None:
        for sensor_position, sensor_direction, sensor_energy, sensor_time_step in zip(
            wrapper.positions,
            wrapper.directions,
            wrapper.energies,
            wrapper.time_deltas,
            strict=True,
        ):
            dist, idx = self.photon_maps[photon_type].tree.query(
                sensor_position, k=self.model.photon_mapping_k
            )
            photon_direction: Vector3Array = self.photon_maps[
                photon_type
            ].data.directions[idx]
            photon_energy: Array = self.photon_maps[photon_type].data.energies[idx]
            photon_time_step: Array = self.photon_maps[photon_type].data.times[idx]

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
                # print()
                # print(photon_direction)
                # print(view_dir)
                # print()

                kernel_size = (4.0 / 3.0) * np.pi * dist[-1] ** 3
            elif photon_type == PhotonType.SURFACE_REFLECTION:
                energy_multiplier = self.model.sea_surface.reflected_energy(
                    -photon_direction,
                    -sensor_direction,
                )

                kernel_size = np.pi * dist[-1] ** 2
            elif photon_type == PhotonType.SURFACE_TRANSMISSION:
                energy_multiplier = self.model.sea_surface.transmitted_energy(
                    -photon_direction,
                    -sensor_direction,
                )

                kernel_size = np.pi * dist[-1] ** 2

            kernel_norm = 1.0 / kernel_size

            store_energy = (
                sensor_energy * photon_energy * kernel_norm * energy_multiplier
            )
            store_time = sensor_time_step + photon_time_step
            sample_idx = (store_time / self.model.sample_multiplier).astype(int)

            # TODO: Test multiple waveforms
            np.add.at(self.return_waveform[photon_type], sample_idx, store_energy)

    # ========================================
    # Step Function
    # ========================================

    def evaluate_state(self, y_current: Array, y_next: Array) -> IntArray:
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
            # print("in water water", len(wrapper.optical_depth_target))
            # print(wrapper.optical_depth_target)

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
            # print("in sim step, len", len(wrapper.optical_depth_target))
            # print(wrapper.optical_depth_target)

        # Water - Air
        if exit_idx.size > 0:
            exit_subset = wrapper.subset(exit_idx)
            exit_subset = self.handle_exit(exit_subset, next_positions[exit_idx], forward=forward)
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
        """
        Handle refraction or reflection at a flat water surface.

        Args:
            subset (PhotonWrapper): _description_
            next_positions (Vector3Array): _description_
            forward (bool): _description_
        """

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

        # Refraction calculation, reverse direction away from intersection point
        cos_incidence = dot_batch_single(-subset.directions, np.array([0, 1, 0]))

        reflection_weight = fresnel_schlick(
            cos_incidence, self.model.sea_surface.base_reflectance
        )
        # print("mean reflection weight", reflection_weight.mean()) # TODO
        transmission_weight = 1.0 - reflection_weight

        # Photon mapping
        # reflection_mask = (
        #     self.rng.random(size=reflection_weight.shape) < reflection_weight
        # )
        # reflection_subset = subset.subset(reflection_mask)
        # reflection_subset.positions = intersection_points[reflection_mask]
        reflection_subset = subset.copy()
        reflection_subset.positions = intersection_points
        reflection_subset.time_deltas += self.current_step
        if forward:
            self.store_photons(reflection_subset, PhotonType.SURFACE_REFLECTION)
        else:
            self.sample_photons(reflection_subset, PhotonType.SURFACE_REFLECTION)
            # self.sample_photons(reflection_subset, PhotonType.SURFACE_TRANSMISSION)

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
        # print("setting optical depth target, len", len(subset.optical_depth_target))
        # print(subset.optical_depth_target)

        return subset

    def handle_exit(
        self,
        subset: PhotonWrapper,
        next_positions: Vector3Array,
        *,
        forward: bool
    ) -> PhotonWrapper:
        """
        Handle refraction or reflection at a flat water surface.

        Args:
            subset (PhotonWrapper): _description_
            next_positions (Vector3Array): _description_
            forward (bool): _description_
        """

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
        # print("transmitted", np.count_nonzero(~reflection_mask))
        transmission_subset = subset.subset(~reflection_mask)
        transmission_subset.positions = intersection_points[~reflection_mask]
        # transmission_subset = subset.copy()
        # transmission_subset.positions = intersection_points
        transmission_subset.time_deltas += self.current_step
        if forward:
            self.store_photons(transmission_subset, PhotonType.SURFACE_REFLECTION)
        else:
            self.sample_photons(transmission_subset, PhotonType.SURFACE_REFLECTION)
            # self.sample_photons(transmission_subset, PhotonType.SURFACE_TRANSMISSION)

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
