import cProfile
import pstats
import numpy as np
import time
import secrets
import math
import sys
from scipy.spatial import KDTree
from typing import List, Tuple
from numpy.typing import NDArray
from line_profiler import profile
from enum import Enum

from camera import Camera
from laser import Laser
from alb_sim.photon_mapping.build_photon_map_data import build_photon_map_data
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.print_photon_map_stats import photon_map_stats
from world import World
from alb_sim.utils.photon_position_state import PhotonPositionState
import utils.numpy_vector as np_vec
from alb_sim.plotting.plot_2d import plot_2d, plot_2d_better
from alb_sim.plotting.plot_scatter_2d import plot_scatter_2d
from alb_sim.plotting.plot_histogram import plot_histogram


PhotonType = Enum("PhotonType", [("BOTTOM_REFLECTION", 0), ("SCATTER", 1), ("SURFACE_REFLECTION", 2)])

photon_dtype = [
    ("position", np.float32, 3),
    ("direction", np.float32, 3),
    ("energy", np.float32),
    ("time", np.float32),
    ("type", PhotonType),
    ("already_reflected", bool),
]

class Simulation:
    def __init__(self, rng: np.random.Generator, num_steps, options = {}):
        self.rng = rng

        self.sample_multiplier = options.get("sample_multiplier", 100)

        self.camera_settings = Camera(options.get("flying_height", 20), options.get("water_depth", 2), options.get("sample_rate", 5_000_000_000))
        self.laser_settings = Laser(self.camera_settings, self.sample_multiplier, 0.1, 1 * 1e-3, 10 * 1e-3 / 2, 0.532, options.get("t_max", 50e-9))
        self.world_settings = World(self.laser_settings, self.camera_settings, options.get("absorption_coefficient", 0.169), options.get("total_scattering_coefficient", 2.5), options.get("salinity_unit", 37), options.get("seafloor_albedo", 0.05), options.get("water_surface_roughness", 0.035), options.get("water_surface_albedo", 0.1))

        self.time_step = 1 / (self.camera_settings.sample_rate * self.sample_multiplier)
        self.water_surface_y = -self.camera_settings.flying_height
        self.seafloor_y = -self.camera_settings.distance_seafloor_flying_height

        self.seafloor_reflection_function_batch = lambda direction, normals: np_vec.cosine_weighted_sample_batch(normals, self.rng)

        # self.photon_batches: List[NDArray] = [] # (position, direction, energy, time_step, reflection)
        # self.photon_np_array: NDArray
        # self.photon_tree: KDTree

        self.photon_storage = PhotonStorage()
        self.photon_maps: dict[PhotonType, PhotonMapIndex] = {}

        self.return_waveform = np.zeros(num_steps * 2)
        self.return_waveform_by_type: dict[PhotonType, NDArray[np.float32]] = {
            type: np.zeros(num_steps * 2, dtype=np.float32)
            for type in PhotonType
        }
        self.photons_found_bottom_reflection = []
        self.photons_found_surface_reflection = []
        self.photons_found_scatter = []
        
        self.photons_in_radius = {x: [] for x in list(PhotonType)}
        self.k_nearest_photons_distance = {x: [] for x in list(PhotonType)}

    @profile
    def simulate_batch(self, num_photons: int, steps: int, forward: bool = True, num_samples_history: int = 0):
        start_batch_time = time.time()
        self.current_step = 0

        history_samples = self.rng.integers(0, num_photons, num_samples_history)

        positions = np.zeros((num_photons, 3), dtype=np.float32)

        directions = np_vec.sample_directions_in_cone(np.array(self.laser_settings.laser_direction), self.laser_settings.laser_divergence_angle if forward else self.laser_settings.field_of_view, num_photons, self.rng)

        velocities = np.full(num_photons, self.world_settings.light_speed_air, dtype=np.float32)
        energies = np.full(num_photons, 1, dtype=np.float32)
        scatter_distances = np.full(num_photons, np.inf, dtype=np.float32)
        time_deltas = self.laser_settings.get_emission_times(num_photons, self.rng) if forward else np.full(num_photons, 0, dtype=np.float32)
        already_reflected = np.full(num_photons, False, dtype=bool)

        full_histories: List[List[NDArray[np.float32]]] = [[] for _ in range(num_samples_history)]

        for i, pos in enumerate(positions[history_samples]):
            full_histories[i].append(pos.copy())


        for _ in range(steps * self.sample_multiplier):
            positions, directions, velocities, energies, scatter_distances, interaction_points, already_reflected = self.simulate_photon_step(positions, directions, velocities, energies, scatter_distances, time_deltas, already_reflected, forward = forward)
            for i, idx in enumerate(history_samples):
                inter = interaction_points[idx]
                if not np.isnan(inter).any():
                    full_histories[i].append(inter.copy())  # intermediate point
                full_histories[i].append(positions[idx].copy())  # final position of this step
            # if not forward and self.current_step % (5 * self.sample_multiplier) == 0:
            #     print(f"step {self.current_step} of {steps * self.sample_multiplier} in {(time.time() - start_batch_time):.2f} s = {((time.time() - start_batch_time) / 60):.2f} min, estimated remaining: {((time.time() - start_batch_time) / 60 / (self.current_step+1) * (steps * self.sample_multiplier - (self.current_step + 1))):.2f} min")
            self.current_step += 1

        return full_histories

    @profile
    def store_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        photon_type: PhotonType,
        already_reflected: NDArray[np.bool_] | None = None,
    ) -> None:
        n = positions.shape[0]

        if already_reflected is None:
            already_reflected = np.full(n, True, dtype=bool)

        self.photon_storage.positions[photon_type].append(positions)
        self.photon_storage.directions[photon_type].append(directions)
        self.photon_storage.energies[photon_type].append(energies)
        self.photon_storage.times[photon_type].append(time_steps)

    @profile
    def sample_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        photon_type: PhotonType
    ) -> None:
        k = 100
        for position, direction, energy, time_step in zip(positions, directions, energies, time_steps):
            dist, idx = self.photon_maps[photon_type].tree.query(position, k=k)
            photon_direction = self.photon_maps[photon_type].data.directions[idx]
            photon_energy = self.photon_maps[photon_type].data.energies[idx]
            photon_time_step = self.photon_maps[photon_type].data.times[idx]

            if photon_type == PhotonType.BOTTOM_REFLECTION: 
                cos_term = np.maximum(
                    0.0,
                    np.dot(np.array([0, 1, 0]), -direction)
                )
                
                energy_multiplier = (self.world_settings.seafloor_albedo / np.pi) * cos_term

                kernel_size = np.pi * dist[-1]**2
            elif photon_type == PhotonType.SCATTER:
                # Vector from scatter point to sensor (i.e. reverse of backward ray)
                view_dir = -direction
                
                energy_multiplier = self.world_settings.scatter_energy_batch(
                    photon_direction, view_dir
                )

                kernel_size = (4.0 / 3.0) * np.pi * dist[-1]**3
            elif photon_type == PhotonType.SURFACE_REFLECTION:
                energy_multiplier = np_vec.microfacet_brdf_batch(
                    -photon_direction,
                    -direction,
                    np.array([0, 1, 0]),
                    self.world_settings.water_surface_roughness,
                    self.world_settings.base_reflectance,
                    self.world_settings.water_surface_albedo,
                )

                kernel_size = np.pi * dist[-1]**2

            kernel_norm = 1.0 / kernel_size

            store_energy = energy * photon_energy * kernel_norm * energy_multiplier
            store_time = time_step + photon_time_step
            sample_idx = (store_time / self.sample_multiplier).astype(int)

            np.add.at(self.return_waveform, sample_idx, store_energy)
            np.add.at(self.return_waveform_by_type[photon_type], sample_idx, store_energy)


    # ========================================
    # Step Function
    # ========================================


    @profile
    def evaluate_state(
        self,
        y_current: NDArray[np.float32],
        y_next: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        states = np.full_like(y_current, PhotonPositionState.IN_AIR.value, dtype=np.int32)

        air_to_water = (y_current > self.water_surface_y) & (y_next <= self.water_surface_y)
        in_water = (y_current <= self.water_surface_y) & (y_next > self.seafloor_y)
        water_to_floor = (y_current > self.seafloor_y) & (y_next <= self.seafloor_y)
        water_to_air = (y_current <= self.water_surface_y) & (y_next > self.water_surface_y)

        states[air_to_water] = PhotonPositionState.ENTERING_WATER.value
        states[in_water] = PhotonPositionState.IN_WATER.value
        states[water_to_floor] = PhotonPositionState.HITTING_SEAFLOOR.value
        states[water_to_air] = PhotonPositionState.EXITING_WATER.value

        return states

    @profile
    def simulate_photon_step(
        self,
        positions: NDArray[np.float32],    # (N, 3)
        directions: NDArray[np.float32],   # (N, 3)
        velocities: NDArray[np.float32],   # (N, 3)
        energies: NDArray[np.float32],     # (N, 1)
        scatter_distances: NDArray[np.float32],     # (N, 1)
        time_deltas: NDArray[np.float32],     # (N, 1)
        already_reflected: NDArray,
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray]:
        """
        Simulates a single step of photon movement and updates direction/position based on medium transitions.
        Returns updated positions, directions, states, and previous positions for logging.
        """

        next_positions = positions + directions * velocities[:,np.newaxis] * self.time_step
        interaction_points = np.full_like(positions, np.nan, dtype=np.float32)

        y_current = positions[:, 1]
        y_next = next_positions[:, 1]
        states = self.evaluate_state(y_current, y_next)

        air_idx = np.nonzero(states == PhotonPositionState.IN_AIR.value)[0]
        enter_idx = np.nonzero(states == PhotonPositionState.ENTERING_WATER.value)[0]
        water_idx = np.nonzero(states == PhotonPositionState.IN_WATER.value)[0]
        hit_floor_idx = np.nonzero(states == PhotonPositionState.HITTING_SEAFLOOR.value)[0]
        exit_idx = np.nonzero(states == PhotonPositionState.EXITING_WATER.value)[0]

        # Process each state
        positions[air_idx] = next_positions[air_idx]
        if water_idx.size > 0:
            water_positions, water_directions, water_energies, new_scatter_distances, scatter_points = self.handle_water_scatter(
                positions[water_idx],
                next_positions[water_idx],
                directions[water_idx],
                energies[water_idx],
                scatter_distances[water_idx],
                time_deltas[water_idx],
                forward = forward
            )
            positions[water_idx] = water_positions
            directions[water_idx] = water_directions
            energies[water_idx] = water_energies
            scatter_distances[water_idx] = new_scatter_distances
            interaction_points[water_idx] = scatter_points

        if enter_idx.size > 0:
            refracted_positions, refracted_directions, refracted_energies, new_scatter_distances, intersection_points = self.handle_enter_exit_water_refraction(
                positions[enter_idx],
                next_positions[enter_idx],
                directions[enter_idx],
                energies[enter_idx],
                scatter_distances[enter_idx],
                time_deltas[enter_idx],
                1,
                self.world_settings.refractive_index_water,
                False,
                forward = forward
            )
            positions[enter_idx] = refracted_positions
            directions[enter_idx] = refracted_directions
            energies[enter_idx] = refracted_energies
            scatter_distances[enter_idx] = new_scatter_distances
            interaction_points[enter_idx] = intersection_points

        if hit_floor_idx.size > 0:
            reflected_positions, reflected_directions, reflected_energies, intersection_points, reflected_already_reflected = self.handle_seafloor_reflection(
                positions[hit_floor_idx],
                next_positions[hit_floor_idx],
                directions[hit_floor_idx],
                energies[hit_floor_idx],
                time_deltas[hit_floor_idx],
                already_reflected[hit_floor_idx],
                forward = forward
            )
            positions[hit_floor_idx] = reflected_positions
            directions[hit_floor_idx] = reflected_directions
            energies[hit_floor_idx] = reflected_energies
            interaction_points[hit_floor_idx] = intersection_points
            already_reflected[hit_floor_idx] = reflected_already_reflected

        if exit_idx.size > 0:
            refracted_positions, refracted_directions, refracted_energies, new_scatter_distances, intersection_points = self.handle_enter_exit_water_refraction(
                positions[exit_idx],
                next_positions[exit_idx],
                directions[exit_idx],
                energies[exit_idx],
                scatter_distances[exit_idx],
                time_deltas[exit_idx],
                self.world_settings.refractive_index_water,
                1,
                True,
                forward = forward
            )
            positions[exit_idx] = refracted_positions
            directions[exit_idx] = refracted_directions
            energies[exit_idx] = refracted_energies
            scatter_distances[exit_idx] = new_scatter_distances
            interaction_points[exit_idx] = intersection_points

        return positions, directions, velocities, energies, scatter_distances, interaction_points, already_reflected


    # ========================================
    # Interaction Handlers
    # ========================================


    @profile
    def handle_seafloor_reflection(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_deltas: NDArray[np.float32],
        already_reflected: NDArray,
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        step = next_positions - positions
        y0 = positions[:, 1]
        dy = step[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            f = (self.seafloor_y - y0) / dy
            f = np.clip(f, 0.0, 1.0)

        intersection = positions + f[:, np.newaxis] * step
        normals = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (len(positions), 1))

        # Sample reflection directions toward the sensor
        reflected_dirs = self.seafloor_reflection_function_batch(directions, normals)

        step_lengths = np.linalg.norm(step, axis=1)
        remaining_fraction = 1.0 - f
        remaining_step = reflected_dirs * (remaining_fraction[:, np.newaxis] * step_lengths[:, np.newaxis])

        final_positions = intersection + remaining_step

        if forward:
            self.store_photons(intersection, directions, energies, f + self.current_step + time_deltas, PhotonType.BOTTOM_REFLECTION, already_reflected)
        else:
            self.sample_photons(intersection, directions, energies, f + self.current_step, PhotonType.BOTTOM_REFLECTION)

        energies *= self.world_settings.seafloor_albedo
        already_reflected = np.full(len(energies), True)

        return final_positions, reflected_dirs, energies, intersection, already_reflected

    @profile
    def handle_enter_exit_water_refraction(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        scatter_distances: NDArray[np.float32],
        time_deltas: NDArray[np.float32],
        n1: float,
        n2: float,
        invert_normals: bool,
        forward: bool,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Handle refraction or reflection at a flat water surface.

        Args:
            positions: (N, 3) current photon positions
            next_positions: (N, 3) next positions
            directions: (N, 3) current directions (normalized)
            n1: refractive index of current medium (e.g. air)
            n2: refractive index of next medium (e.g. water)
            invert_normals: True if photon is exiting the water
            forward: Whether this is forward time stepping
            attenuation_coefficient: Used to sample scattering distances (only when entering water)

        Returns:
            - updated positions (after water interaction)
            - updated directions
            - intersection points with water surface
            - scattering distances (np.inf if photon is reflected or not entering water)
        """

        normals = np.tile(np.array([[0, -1 if invert_normals else 1, 0]], dtype=np.float32), (len(positions), 1))

        step = next_positions - positions
        y0 = positions[:, 1]
        dy = step[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            f = (self.water_surface_y - y0) / dy
            f = np.clip(f, 0.0, 1.0)

        intersection = positions + f[:, np.newaxis] * step

        # Refraction calculation
        eta = n1 / n2
        I = directions
        N = normals
        cos_i = -np.einsum('ij,ij->i', I, N)

        # Fresnel reflectance
        R = np_vec.fresnel_schlick(cos_i, self.world_settings.base_reflectance) # fraction reflected
        T = 1.0 - R # fraction transmitted

        # Store incident photon for backward gathering (before splitting energy)
        if True:
            if forward:
                self.store_photons(intersection, directions, energies, f + self.current_step + time_deltas, PhotonType.SURFACE_REFLECTION)
            else:
                self.sample_photons(intersection, directions, energies, f + self.current_step, PhotonType.SURFACE_REFLECTION)

        # Total internal reflection mask
        k = 1.0 - eta**2 * (1.0 - cos_i**2)
        tir_mask = k < 0.0  # total internal reflection

        # Reflected directions
        reflected_dirs = I - 2 * np.einsum('ij,ij->i', I, N)[:, None] * N

        # Refracted directions
        sqrt_k = np.sqrt(np.maximum(k, 0.0))
        refracted_dirs = eta * I + (eta * cos_i - sqrt_k)[:, None] * N

        # Choose final direction
        new_directions = np.where(tir_mask[:, None], reflected_dirs, refracted_dirs)

        # Split energy
        new_energies = energies * T  # transmitted energy continues
        # (Reflected part is captured via stored photon for backward pass)

        # Continue step into medium
        step_lengths = np.linalg.norm(step, axis=1)
        remaining_fraction = 1.0 - f
        remaining_step = new_directions * (remaining_fraction[:, None] * step_lengths[:, None])
        final_positions = intersection + remaining_step

        # Scatter distances: only for refraction into water
        if not invert_normals:
            refracted_mask = ~tir_mask
            # matlab uses: rand = 1 - np.exp(-self.rng.random(np.count_nonzero(refracted_mask)).astype(np.float32))
            rand = self.rng.random(np.count_nonzero(refracted_mask)).astype(np.float32)
            dpath = -np.log(rand) / self.world_settings.lidar_attenuation_coefficient
            scatter_distances[refracted_mask] = dpath.astype(np.float32)

        return final_positions, np_vec.normalize_batch(new_directions), new_energies, scatter_distances, intersection

    @profile
    def handle_water_scatter(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        scatter_distances: NDArray[np.float32],
        time_deltas: NDArray[np.float32],
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Handle scattering and general traversing in water.

        Returns:
            new_positions, new_directions, new_energies, scatter_points, new_scatter_distances
        """

        step_vectors = next_positions - positions
        step_lengths = np.linalg.norm(step_vectors, axis=1)
        # print(step_lengths.mean())
        
        scatter_occurs = scatter_distances <= step_lengths
        no_scatter = ~scatter_occurs

        # Allocate outputs
        new_positions = np.empty_like(positions)
        new_directions = directions.copy()
        new_energies = energies.copy()
        scatter_points = np.full_like(positions, np.nan)
        new_scatter_distances = scatter_distances.copy()

        # --- No scatter: move fully
        new_positions[no_scatter] = next_positions[no_scatter]
        new_scatter_distances[no_scatter] -= step_lengths[no_scatter]

        # --- Scatter case
        if np.any(scatter_occurs):
            idx = np.nonzero(scatter_occurs)[0]
            t = scatter_distances[idx] / step_lengths[idx]
            scatter_vec = step_vectors[idx] * t[:, None]
            scatter_pos = positions[idx] + scatter_vec
            scatter_points[idx] = scatter_pos

            new_dir = np_vec.normalize_batch(self.world_settings.sample_scattering_directions_batch(directions[idx], self.rng))
            new_directions[idx] = new_dir

            # Energy loss due to scattering
            new_energies[idx] *= (self.world_settings.attenuation_per_scatter)

            # Move remaining distance after scattering
            remaining_fraction = 1.0 - t
            remaining_distance = remaining_fraction * step_lengths[idx]
            final_positions = scatter_pos + new_dir * remaining_distance[:, None]

            new_positions[idx] = final_positions

            # Resample new scatter distances
            # matlab uses: rand_vals = 1 - np.exp(-self.rng.random(len(idx)).astype(np.float32))
            rand_vals = self.rng.random(len(idx)).astype(np.float32)
            new_scatter_distances[idx] = -np.log(rand_vals) / self.world_settings.lidar_attenuation_coefficient
            
            if forward:
                self.store_photons(scatter_points[idx], directions[idx], energies[idx], t + self.current_step + time_deltas[idx], PhotonType.SCATTER)
            else:
                self.sample_photons(scatter_points[idx], directions[idx], energies[idx], t + self.current_step, PhotonType.SCATTER)

        return new_positions, new_directions, new_energies, new_scatter_distances, scatter_points


if __name__ == "__main__":
    rng = np.random.default_rng(secrets.randbits(128))
    options = {
        "flying_height": 135,
        "water_depth": 7,
        "sample_rate": 5_000_000_000,
        "sample_multiplier": 10,
        # "absorption_coefficient": 0.114,
        # "total_scattering_coefficient": 0.037
    }
    distance = ((options["flying_height"] + options["water_depth"]) / math.cos(math.radians(15)))
    steps = int(1.5 * (distance * round(options["sample_rate"])) / 2.998e8)
    
    simulation = Simulation(rng, steps, options)
    # print(f"{steps} steps, this will simulate {steps * simulation.world_settings.light_speed_air / simulation.camera_settings.sample_rate} m")
    # print(f"distance laser - seafloor is {round(np.dot(np.array([0, -simulation.camera_settings.distance_seafloor_flying_height, 0]), np.array([0, 1, 0]))/np.dot(np_vec.normalize_vector(np.array(simulation.laser_settings.laser_direction)), np.array([0, 1, 0])), 2)} m")
    start = time.time()
    photons_per_batch = 10_000
    batches = 5
    visualize_paths = 0
    
    # ------------------------------
    # Forward pass
    # ------------------------------

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(batches):
        history = simulation.simulate_batch(photons_per_batch, steps, True, visualize_paths if i == batches - 1 else 0)
        # if (i+1) % 5 == 0:
        print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((((time.time() - start) / 60) / (i+1)) * (batches - (i + 1))):.2f} min")

    profiler.disable()
    profiler.dump_stats("forward.prof")

    elapsed = time.time() - start
    print(f"time forward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)  # Top 30 functions

    # ------------------------------
    # saving
    # ------------------------------
    start = time.time()
    # simulation.photon_np_array = np.concatenate(simulation.photon_batches)
    photon_maps_data = build_photon_map_data(simulation.photon_storage)
    print(photon_map_stats(photon_maps_data))
    # print(f"{len(simulation.photon_np_array)} entries in photon list, {(sys.getsizeof(simulation.photon_np_array) / 1024):.2f} KiB")
    for photon_type, data in photon_maps_data.items():
        simulation.photon_maps[photon_type] = PhotonMapIndex(data)

    # np.save(f"photon-map_{(photons_per_batch*batches):,}-photons.npy", simulation.photon_np_array)
    # positions = np.array(simulation.photon_np_array["position"])
    # simulation.photon_tree = KDTree(positions)
    elapsed = time.time() - start
    print(f"time saving: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    # ------------------------------
    # Backward pass
    # ------------------------------

    start = time.time()
    photons_per_batch = 10_000
    batches = 1

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(batches):
        simulation.simulate_batch(photons_per_batch, steps, False, 0)
        print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((time.time() - start) / 60 / (i+1) * (batches - (i + 1))):.2f} min")

    profiler.disable()
    profiler.dump_stats("backward.prof")

    elapsed = time.time() - start
    print(f"time backward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)  # Top 30 functions

    plot_2d_better(simulation.return_waveform, title="waveform", ylabel="Intensity", xlabel="Sample", xlim=(4650,4800), params=options, show=True)
    

    data = simulation.return_waveform_by_type
    import matplotlib.pyplot as plt
    x = np.arange(len(next(iter(data.values()))))
    labels = [pt.name for pt in data.keys()]
    values = np.vstack(list(data.values()))  
    plt.figure()
    plt.stackplot(x, values, labels=labels)
    plt.legend(loc="upper left")
    plt.xlabel("Step / Distance")
    plt.ylabel("Photon contribution")
    plt.title("Photon contributions (stacked)")
    plt.show()
    # photons_bottom_reflections = np.array(simulation.photons_found_bottom_reflection)
    # print(f"{np.count_nonzero(photons_bottom_reflections == 0)} sensor photons found no bottom reflection photons, {(np.count_nonzero(photons_bottom_reflections == 0) / len(photons_bottom_reflections) * 100):.3f} %")
    # photons_scatters = np.array(simulation.photons_found_scatter)
    # print(f"{np.count_nonzero(photons_scatters == 0)} sensor photons found no scatter photons, {(np.count_nonzero(photons_scatters == 0) / len(photons_scatters) * 100):.3f} %")
    # photons_surface_reflections = np.array(simulation.photons_found_surface_reflection)
    # print(f"{np.count_nonzero(photons_surface_reflections == 0)} sensor photons found no surface reflection photons, {(np.count_nonzero(photons_surface_reflections == 0) / len(photons_surface_reflections) * 100):.3f} %")
    
    
    # for x in list(PhotonType):
    #     y = np.array(simulation.photons_in_radius[x])
    #     z = np.array(simulation.k_nearest_photons_distance[x])
    #     print(x)
    #     print("min", y.min())
    #     print("max", y.max())
    #     print("mean", y.mean())
    #     print(f"{np.count_nonzero(y == 0)} sensor photons found no {x} photons, {(np.count_nonzero(y == 0) / len(y) * 100):.3f} %")
    #     print("min dist", z.min())
    #     print("max dist", z.max())
    #     print("mean dist", z.mean())
    # photons_surface_in_radius = np.array(simulation.photons_in_radius[PhotonType.SURFACE_REFLECTION])
    # plot_histogram(photons_surface_in_radius[photons_surface_in_radius < 400], bins = 400, title="Number of Photons found at surface Reflections", xlabel="Photons in Radius")
    # plot_histogram(photons_scatters[photons_scatters < 400], bins = 400, title="Number of Photons found at Scatters", xlabel="Photons in Radius") 