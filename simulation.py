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

from camera import Camera
from laser import Laser
from utils.gpt_ffscatter import generate_ff_phase_function, sample_scattering_directions_batch, scatter_energy
from world import World
from utils.photon_state import PhotonState
import utils.numpy_vector as np_vec
from utils.plot_2d import plot_2d
from utils.visualize_paths import visualize_photon_paths
from utils.plot_scatter_2d import plot_scatter_2d
from utils.plot_histogram import plot_histogram

photon_dtype = [
    ("position", np.float32, 3),
    ("direction", np.float32, 3),
    ("energy", np.float32),
    ("time", np.float32),
    ("reflection", bool),
    ("already_reflected", bool),
]


class Simulation:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        self.number_of_photons = 1e5 # Number of photon packets.
        self.photon_survival_threshold_weight = 0.0001 # epsilon

        self.laser_settings = Laser()
        self.camera_settings = Camera()
        self.world_settings = World(self.laser_settings, self.camera_settings)
        self.ff_theta, self.ff_phase = generate_ff_phase_function(n_ff=1.0, M=18000)
        
        # TODO: fix sample multiplier changing breaking store_time in sample_photons
        self.sample_multiplier = 10

        self.time_step = 1 / (self.camera_settings.sample_rate * self.sample_multiplier)
        self.water_surface_y = -self.camera_settings.flying_height
        self.seafloor_y = -self.camera_settings.distance_seafloor_flying_height

        self.seafloor_reflection_function_batch = lambda direction, normals: np_vec.cosine_weighted_sample_batch(normals, self.rng)

        self.photon_batches: List[NDArray] = [] # (position, direction, energy, time_step, reflection)
        self.photon_np_array: NDArray
        self.photon_tree: KDTree

        self.return_waveform = np.zeros(2500)
        self.photons_found_reflection = []
        self.photons_found_scatter = []

    def simulate_batch(self, num_photons: int, steps: int, forward: bool = True, num_samples_history: int = 0):
        self.current_step = 0
        inner_start_time = time.time()

        N = num_photons
        history_samples = self.rng.integers(0, num_photons, num_samples_history)

        positions = np.zeros((N, 3), dtype=np.float32)

        directions = np_vec.sample_directions_in_cone(np.array(self.laser_settings.laser_direction), self.laser_settings.laser_divergence_angle if forward else self.laser_settings.field_of_view, N, self.rng)

        velocities = np.full(N, self.world_settings.light_speed_air, dtype=np.float32)
        energies = np.full(N, 1, dtype=np.float32)
        scatter_distances = np.full(N, np.inf, dtype=np.float32)
        already_reflected = np.full(N, False, dtype=bool)

        full_histories: List[List[NDArray[np.float32]]] = [[] for _ in range(num_samples_history)]

        for i, pos in enumerate(positions[history_samples]):
            full_histories[i].append(pos.copy())


        for c_step in range(steps * self.sample_multiplier):
            positions, directions, velocities, energies, scatter_distances, interaction_points, already_reflected = self.simulate_photon_step(positions, directions, velocities, energies, scatter_distances, already_reflected, forward = forward)
            for i, idx in enumerate(history_samples):
                inter = interaction_points[idx]
                if not np.isnan(inter).any():
                    full_histories[i].append(inter.copy())  # intermediate point
                full_histories[i].append(positions[idx].copy())  # final position of this step
            self.current_step += 1
            # if not forward and ((c_step + 1) / self.sample_multiplier) % 10 == 0:
            #     print(f"current at inner step {(c_step + 1) / self.sample_multiplier} after {(time.time() - inner_start_time):.2f} s = {((time.time() - inner_start_time) / 60):.2f} min")

        return full_histories

    def store_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        reflection: bool,
        already_reflected = None
    ) -> None:
        n = positions.shape[0]
        new_photons = np.empty(n, dtype=photon_dtype)
        if already_reflected is None:
            already_reflected = np.full(n, True)
        new_photons["position"] = positions
        new_photons["direction"] = directions
        new_photons["energy"] = energies
        new_photons["time"] = time_steps
        new_photons["reflection"] = reflection
        new_photons["already_reflected"] = already_reflected

        self.photon_batches.append(new_photons)

    def sample_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        reflection: bool
    ) -> None:
        for position, direction, energy, time_step in zip(positions, directions, energies, time_steps):
            idx = self.photon_tree.query_ball_point(position, r=0.1 if reflection else 0.5)
            if not idx: # nothing found
                (self.photons_found_reflection if reflection else self.photons_found_scatter).append(0)
                continue

            photons = self.photon_np_array[idx]
            match_mask = (photons["reflection"] == reflection)
            matching = photons[match_mask]

            (self.photons_found_reflection if reflection else self.photons_found_scatter).append(len(matching))

            if len(matching) == 0:
                continue

            dists = np.linalg.norm(matching["position"] - position, axis=1)
            
            # k nearest idx
            k = 2
            k_eff = min(k, len(dists))
            nearest_idx = np.argpartition(dists, k_eff - 1)[:k_eff] if k_eff > 0 else []
            for photon_position, photon_direction, photon_energy, photon_time_step, _, _ in matching[nearest_idx]: #found[sort_idx]:
                # if done >= 2:
                #     break
                # if photon_reflection != reflection: continue

                if reflection: 
                    store_energy = energy * photon_energy * (self.world_settings.seafloor_albedo / np.pi) * max(0, np_vec.dot_vector(np.array([0, 1, 0]), -direction))
                    store_time = time_step + photon_time_step

                    self.return_waveform[int(store_time / self.sample_multiplier)] += store_energy
                else:
                    # 1. Vector from scatter point to sensor (i.e. reverse of backward ray)
                    view_dir = -direction

                    # # 2. Photon originally came from photon_direction
                    # cos_theta = np.dot(view_dir, -photon_direction)
                    # cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical safety

                    # theta = np.arccos(cos_theta)

                    # # 3. Interpolate FF phase function probability at this angle
                    # ff_prob = np.interp(theta, self.world_settings.ct_r, self.world_settings.ff_phase_pdf).astype(np.float32)

                    # # 4. Energy contribution (scaled by FF phase prob)
                    # # The 1/pi term may not apply here — FF scattering is anisotropic, not Lambertian
                    # # This is the scattering PDF * incoming energy * current energy
                    # store_energy = energy * photon_energy * ff_prob

                    store_energy = energy * photon_energy * scatter_energy(self.ff_phase, self.ff_theta, photon_direction, view_dir)

                    # 5. Total time is round-trip time
                    store_time = time_step + photon_time_step
                    if 0 <= int(store_time / self.sample_multiplier) < len(self.return_waveform):
                        self.return_waveform[int(store_time / self.sample_multiplier)] += store_energy
                    else:
                        print("ERROR: SHOULD NEVER HAPPEN!!!!")

                # done += 1

    # def sample_photons(
    #     self,
    #     positions: NDArray[np.float32],
    #     directions: NDArray[np.float32],
    #     energies: NDArray[np.float32],
    #     time_steps: NDArray[np.float32],
    #     reflection: bool
    # ) -> None:
    #     r = 0.1 if reflection else 0.5
    #     all_idxs = KDTree(positions).query_ball_tree(self.photon_tree, r=r)
    #     normal = np.array([0, 1, 0], dtype=np.float32)
    #     sample_inv = 1.0 / self.sample_multiplier
    #     # print("in sample_photons")
    #     # print(positions)
    #     # print(len(all_idxs))

    #     for i, idx_list in enumerate(all_idxs):
    #         # print(i, idx_list)
    #         if not idx_list:
    #             (self.photons_found_reflection if reflection else self.photons_found_scatter).append(0)
    #             continue

    #         photons = self.photon_np_array[idx_list]
    #         match_mask = (photons["reflection"] == reflection)
    #         matching = photons[match_mask]

    #         (self.photons_found_reflection if reflection else self.photons_found_scatter).append(len(matching))

    #         if len(matching) == 0:
    #             continue

    #         dists = np.linalg.norm(matching["position"] - positions[i], axis=1)
            
    #         # k nearest idx
    #         k = 2
    #         k_eff = min(k, len(dists))
    #         nearest_idx = np.argpartition(dists, k_eff - 1)[:k_eff] if k_eff > 0 else []


    #         for j in nearest_idx:
    #             photon_pos = matching["position"][j]
    #             photon_dir = matching["direction"][j]
    #             photon_energy = matching["energy"][j]
    #             photon_time = matching["time"][j]

    #             if reflection:
    #                 dot_n = max(0, np.dot(normal, -photon_dir))
    #                 store_energy = energies[i] * photon_energy * (self.world_settings.seafloor_albedo / np.pi) * dot_n
    #             else:
    #                 view_dir = -directions[i]
    #                 cos_theta = np.dot(view_dir, -photon_dir)
    #                 cos_theta = np.clip(cos_theta, -1.0, 1.0)
    #                 theta = np.arccos(cos_theta)
    #                 ff_prob = np.interp(theta, self.world_settings.ct_r, self.world_settings.ff_phase_pdf).astype(np.float32)
    #                 store_energy = energies[i] * photon_energy * ff_prob

    #             store_time = time_steps[i] + photon_time
    #             bin_idx = int(store_time * sample_inv)

    #             if 0 <= bin_idx < len(self.return_waveform):
    #                 self.return_waveform[bin_idx] += store_energy


    # ========================================
    # Step Function
    # ========================================


    def evaluate_state(
        self,
        y_current: NDArray[np.float32],
        y_next: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        states = np.full_like(y_current, PhotonState.IN_AIR.value, dtype=np.int32)

        air_to_water = (y_current > self.water_surface_y) & (y_next <= self.water_surface_y)
        in_water = (y_current <= self.water_surface_y) & (y_next > self.seafloor_y)
        water_to_floor = (y_current > self.seafloor_y) & (y_next <= self.seafloor_y)
        water_to_air = (y_current <= self.water_surface_y) & (y_next > self.water_surface_y)

        states[air_to_water] = PhotonState.ENTERING_WATER.value
        states[in_water] = PhotonState.IN_WATER.value
        states[water_to_floor] = PhotonState.HITTING_SEAFLOOR.value
        states[water_to_air] = PhotonState.EXITING_WATER.value

        return states

    def simulate_photon_step(
        self,
        positions: NDArray[np.float32],    # (N, 3)
        directions: NDArray[np.float32],   # (N, 3)
        velocities: NDArray[np.float32],   # (N, 3)
        energies: NDArray[np.float32],     # (N, 3)
        scatter_distances: NDArray[np.float32],     # (N, 3)
        already_reflected: NDArray,     # (N, 3)
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

        air_idx = np.nonzero(states == PhotonState.IN_AIR.value)[0]
        enter_idx = np.nonzero(states == PhotonState.ENTERING_WATER.value)[0]
        water_idx = np.nonzero(states == PhotonState.IN_WATER.value)[0]
        hit_floor_idx = np.nonzero(states == PhotonState.HITTING_SEAFLOOR.value)[0]
        exit_idx = np.nonzero(states == PhotonState.EXITING_WATER.value)[0]

        # Process each state
        positions[air_idx] = next_positions[air_idx]
        if water_idx.size > 0:
            water_positions, water_directions, water_energies, new_scatter_distances, scatter_points = self.handle_water_scatter(
                positions[water_idx],
                next_positions[water_idx],
                directions[water_idx],
                energies[water_idx],
                scatter_distances[water_idx],
                forward = forward
            )
            positions[water_idx] = water_positions
            directions[water_idx] = water_directions
            energies[water_idx] = water_energies
            scatter_distances[water_idx] = new_scatter_distances
            interaction_points[water_idx] = scatter_points

        if enter_idx.size > 0:
            refracted_positions, refracted_directions, new_scatter_distances, intersection_points = self.handle_enter_exit_water_refraction(
                positions[enter_idx],
                next_positions[enter_idx],
                directions[enter_idx],
                scatter_distances[enter_idx],
                1,
                self.world_settings.refractive_index_water,
                False,
                forward = forward
            )
            positions[enter_idx] = refracted_positions
            directions[enter_idx] = refracted_directions
            scatter_distances[enter_idx] = new_scatter_distances
            interaction_points[enter_idx] = intersection_points

        if hit_floor_idx.size > 0:
            reflected_positions, reflected_directions, reflected_energies, intersection_points, reflected_already_reflected = self.handle_seafloor_reflection(
                positions[hit_floor_idx],
                next_positions[hit_floor_idx],
                directions[hit_floor_idx],
                energies[hit_floor_idx],
                already_reflected[hit_floor_idx],
                forward = forward
            )
            # print(reflected_positions.shape)
            positions[hit_floor_idx] = reflected_positions
            directions[hit_floor_idx] = reflected_directions
            energies[hit_floor_idx] = reflected_energies
            interaction_points[hit_floor_idx] = intersection_points
            already_reflected[hit_floor_idx] = reflected_already_reflected

        if exit_idx.size > 0:
            refracted_positions, refracted_directions, new_scatter_distances, intersection_points = self.handle_enter_exit_water_refraction(
                positions[exit_idx],
                next_positions[exit_idx],
                directions[exit_idx],
                scatter_distances[exit_idx],
                self.world_settings.refractive_index_water,
                1,
                True,
                forward = forward
            )
            positions[exit_idx] = refracted_positions
            directions[exit_idx] = refracted_directions
            scatter_distances[exit_idx] = new_scatter_distances
            interaction_points[exit_idx] = intersection_points

        return positions, directions, velocities, energies, scatter_distances, interaction_points, already_reflected


    # ========================================
    # Interaction Handlers
    # ========================================


    def handle_seafloor_reflection(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
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
            self.store_photons(intersection, directions, energies, f + self.current_step, True, already_reflected)
        else:
            self.sample_photons(intersection, directions, energies, f + self.current_step, True)

        energies *= self.world_settings.seafloor_albedo
        already_reflected = np.full(len(energies), True)

        return final_positions, reflected_dirs, energies, intersection, already_reflected

    def handle_enter_exit_water_refraction(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        scatter_distances: NDArray[np.float32],
        n1: float,
        n2: float,
        invert_normals: bool,
        forward: bool,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
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
        k = 1.0 - eta**2 * (1.0 - cos_i**2)
        tir_mask = k < 0.0  # total internal reflection

        # Reflected directions
        reflected_dirs = I - 2 * np.einsum('ij,ij->i', I, N)[:, None] * N

        # Refracted directions
        sqrt_k = np.sqrt(np.maximum(k, 0.0))
        refracted_dirs = eta * I + (eta * cos_i - sqrt_k)[:, None] * N

        # Choose final direction
        new_directions = np.where(tir_mask[:, None], reflected_dirs, refracted_dirs)

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

        return final_positions, np_vec.normalize_batch(new_directions), scatter_distances, intersection

    def handle_water_scatter(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        scatter_distances: NDArray[np.float32],
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

        # if not forward:
        #     self.sample_photons(positions[no_scatter], directions[no_scatter], energies[no_scatter], np.full(len(positions[no_scatter]), self.current_step), False)

        # --- Scatter case
        if np.any(scatter_occurs):
            idx = np.nonzero(scatter_occurs)[0]
            t = scatter_distances[idx] / step_lengths[idx]
            scatter_vec = step_vectors[idx] * t[:, None]
            scatter_pos = positions[idx] + scatter_vec
            scatter_points[idx] = scatter_pos

            # Random azimuth and longitudinal angle (based on ff phase function)
            # f2 = self.rng.uniform(0, 2 * np.pi, size=len(idx)).astype(np.float32)
            # rand_ct = self.rng.random(len(idx)).astype(np.float32)
            # ct2 = np.interp(rand_ct, self.world_settings.ff_phase_cdf, self.world_settings.ct_r).astype(np.float32)

            # ux2 = directions[idx][:, 0]
            # uy2 = directions[idx][:, 1]
            # uz2 = directions[idx][:, 2]

            # same_z_mask = np.abs(uz2) > 0.99999
            # new_dir = np.empty((len(idx), 3), dtype=np.float32)

            # sin_ct2 = np.sin(ct2)
            # cos_ct2 = np.cos(ct2)

            # # Case 1: nearly aligned with Z
            # new_dir[same_z_mask, 0] = sin_ct2[same_z_mask] * np.cos(f2[same_z_mask])
            # new_dir[same_z_mask, 1] = sin_ct2[same_z_mask] * np.sin(f2[same_z_mask])
            # new_dir[same_z_mask, 2] = np.sign(uz2[same_z_mask]) * cos_ct2[same_z_mask]

            # # Case 2: general case
            # not_same_z = ~same_z_mask
            # denom = np.sqrt(1 - uz2[not_same_z]**2)
            # sin_f2 = np.sin(f2[not_same_z])
            # cos_f2 = np.cos(f2[not_same_z])

            # new_dir[not_same_z, 0] = (
            #     sin_ct2[not_same_z] * (
            #         ux2[not_same_z] * uz2[not_same_z] * cos_f2 -
            #         uy2[not_same_z] * sin_f2
            #     ) / denom +
            #     ux2[not_same_z] * cos_ct2[not_same_z]
            # )
            # new_dir[not_same_z, 1] = (
            #     sin_ct2[not_same_z] * (
            #         uy2[not_same_z] * uz2[not_same_z] * cos_f2 +
            #         ux2[not_same_z] * sin_f2
            #     ) / denom +
            #     uy2[not_same_z] * cos_ct2[not_same_z]
            # )
            # new_dir[not_same_z, 2] = (
            #     -sin_ct2[not_same_z] * denom * cos_f2 +
            #     uz2[not_same_z] * cos_ct2[not_same_z]
            # )

            # # Normalize directions
            # new_dir = np_vec.normalize_batch(new_dir)
            # new_directions[idx] = new_dir
            new_dir = np_vec.normalize_batch(sample_scattering_directions_batch(self.ff_phase, self.ff_theta, directions[idx], self.rng))
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
                self.store_photons(scatter_points[idx], directions[idx], energies[idx], t + self.current_step, False)
            else:
                self.sample_photons(scatter_points[idx], directions[idx], energies[idx], t + self.current_step, False)

        return new_positions, new_directions, new_energies, new_scatter_distances, scatter_points


if __name__ == "__main__":
    rng = np.random.default_rng(secrets.randbits(128))
    simulation = Simulation(rng)
    start = time.time()
    photons_per_batch = 25_000
    batches = 250
    steps = 1250
    visualize_paths = 0

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(batches):
        history = simulation.simulate_batch(photons_per_batch, steps, True, visualize_paths if i == batches - 1 else 0)
        if (i+1) % 5 == 0:
            print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((((time.time() - start) / 60) / (i+1)) * (batches - (i + 1))):.2f} min")

    profiler.disable()

    elapsed = time.time() - start
    print(f"time forward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)  # Top 30 functions


    # visualize_photon_paths(history, simulation.water_surface_y, simulation.seafloor_y)

    simulation.photon_np_array = np.concatenate(simulation.photon_batches)
    np.save(f"photon-map_{(photons_per_batch*batches):,}-photons.npy", simulation.photon_np_array)
    print(f"{len(simulation.photon_np_array)} entries in photon list, {(sys.getsizeof(simulation.photon_np_array) / 1024):.2f} KiB")
    positions = np.array(simulation.photon_np_array["position"])
    simulation.photon_tree = KDTree(positions)

    # visualize_photons = simulation.photon_np_array[rng.integers(0, len(simulation.photon_np_array), 10_000)]
    # plot_scatter_2d(visualize_photons["position"][:, 0], visualize_photons["position"][:, 1], ylabel="Y-Axis")

    start = time.time()
    photons_per_batch = 5_000
    batches = 25
    steps = 1250

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(batches):
        simulation.simulate_batch(photons_per_batch, steps, False, 0)
        print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((time.time() - start) / 60 / (i+1) * (batches - i + 1)):.2f} min")

    profiler.disable()

    elapsed = time.time() - start
    print(f"time backward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)  # Top 30 functions

    plot_2d(simulation.return_waveform, title="waveform", ylabel="Intensity", xlabel="Sample")
    photons_reflections = np.array(simulation.photons_found_reflection)
    print(f"{np.count_nonzero(photons_reflections == 0)} sensor photons found no reflection photons, {(np.count_nonzero(photons_reflections == 0) / len(photons_reflections) * 100):.3f} %")
    photons_scatters = np.array(simulation.photons_found_scatter)
    print(f"{np.count_nonzero(photons_scatters == 0)} sensor photons found no scatter photons, {(np.count_nonzero(photons_scatters == 0) / len(photons_scatters) * 100):.3f} %")
    plot_histogram(photons_reflections[photons_reflections < 400], bins = 400, title="Number of Photons found at Reflections", xlabel="Photons in Radius")
    plot_histogram(photons_scatters[photons_scatters < 400], bins = 400, title="Number of Photons found at Scatters", xlabel="Photons in Radius")