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
from utils.plot_2d import plot_2d
from world import World
from utils.photon_state import PhotonState
import utils.numpy_vector as np_vec
from utils.visualize_paths import visualize_photon_paths
from utils.plot_scatter_2d import plot_scatter_2d
from utils.plot_histogram import plot_histogram


class Simulation:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        self.number_of_photons = 1e5 # Number of photon packets.
        self.photon_survival_threshold_weight = 0.0001 # epsilon

        self.laser_settings = Laser()
        self.camera_settings = Camera()
        self.world_settings = World(self.laser_settings)

        self.time_step = 1 / self.camera_settings.sample_rate
        self.water_surface_y = -self.camera_settings.flying_height
        self.seafloor_y = -self.camera_settings.distance_seafloor_flying_height

        self.seafloor_reflection_function_batch = lambda direction, normals: np_vec.heuristic_sample_batch(normals, self.rng)

        self.photons_list: List[Tuple[NDArray[np.float32], NDArray[np.float32], float, float, bool]] = [] # (position, direction, energy, time_step, reflection)
        self.photon_np_array: NDArray
        self.photon_tree: KDTree

        self.return_waveform = np.zeros(2500)

    def simulate_batch(self, num_photons: int, steps: int, forward: bool = True, num_samples_history: int = 0):
        self.current_step = 0

        N = num_photons
        history_samples = self.rng.integers(0, num_photons, num_samples_history)

        positions = np.zeros((N, 3), dtype=np.float32)

        directions = np_vec.sample_directions_in_cone(np.array(self.laser_settings.laser_direction), self.laser_settings.laser_divergence_angle if forward else self.laser_settings.field_of_view, N, self.rng)

        velocities = np.full(N, self.world_settings.light_speed_air, dtype=np.float32)
        energies = np.full(N, 1, dtype=np.float32)

        full_histories: List[List[NDArray[np.float32]]] = [[] for _ in range(num_samples_history)]

        for i, pos in enumerate(positions[history_samples]):
            full_histories[i].append(pos.copy())


        for _ in range(steps):
            positions, directions, velocities, energies, interaction_points = self.simulate_photon_step(positions, directions, velocities, energies, forward = forward)
            for i, idx in enumerate(history_samples):
                inter = interaction_points[idx]
                if not np.isnan(inter).any():
                    full_histories[i].append(inter.copy())  # intermediate point
                full_histories[i].append(positions[idx].copy())  # final position of this step
            self.current_step += 1

        return full_histories

    def store_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        reflection: bool
    ) -> None:
        for position, direction, energy, time_step in zip(positions, directions, energies, time_steps):
            self.photons_list.append((position, direction, energy, time_step, reflection))

    def sample_photons(
        self,
        positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        time_steps: NDArray[np.float32],
        reflection: bool
    ) -> None:
        for position, direction, energy, time_step in zip(positions, directions, energies, time_steps):
            idx = self.photon_tree.query_ball_point(position, r=0.02)
            done = 0
            for photon_position, photon_direction, photon_energy, photon_time_step, photon_reflection in self.photon_np_array[idx]:
                if done >= 10:
                    break
                if photon_reflection != reflection: continue

                if reflection: 
                    store_energy = energy * photon_energy * (1 / np.pi) * max(0, np_vec.dot_vector(np.array([0, 1, 0]), -photon_direction))
                    store_time = time_step + photon_time_step

                    self.return_waveform[int(store_time)] += store_energy

                done += 1


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
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
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
        positions[water_idx] = next_positions[water_idx]

        if enter_idx.size > 0:
            refracted_positions, refracted_directions, intersection_points = self.handle_enter_exit_water_refraction(
                positions[enter_idx],
                next_positions[enter_idx],
                directions[enter_idx],
                1,
                self.world_settings.refractive_index_water,
                False,
                forward = forward
            )
            positions[enter_idx] = refracted_positions
            directions[enter_idx] = refracted_directions
            interaction_points[enter_idx] = intersection_points

        if hit_floor_idx.size > 0:
            reflected_positions, reflected_directions, intersection_points = self.handle_seafloor_reflection(
                positions[hit_floor_idx],
                next_positions[hit_floor_idx],
                directions[hit_floor_idx],
                energies[hit_floor_idx],
                forward = forward
            )
            # print(reflected_positions.shape)
            positions[hit_floor_idx] = reflected_positions
            directions[hit_floor_idx] = reflected_directions
            interaction_points[hit_floor_idx] = intersection_points

        if exit_idx.size > 0:
            refracted_positions, refracted_directions, intersection_points = self.handle_enter_exit_water_refraction(
                positions[exit_idx],
                next_positions[exit_idx],
                directions[exit_idx],
                self.world_settings.refractive_index_water,
                1,
                True,
                forward = forward
            )
            positions[exit_idx] = refracted_positions
            directions[exit_idx] = refracted_directions
            interaction_points[exit_idx] = intersection_points

        return positions, directions, velocities, energies, interaction_points


    # ========================================
    # Interaction Handlers
    # ========================================


    def handle_seafloor_reflection(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energies: NDArray[np.float32],
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
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
            self.store_photons(intersection, directions, energies, remaining_fraction + self.current_step, True)
        else:
            self.sample_photons(intersection, directions, energies, remaining_fraction + self.current_step, True)

        return final_positions, reflected_dirs, intersection

    def handle_enter_exit_water_refraction(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        n1: float,
        n2: float,
        invert_normals: bool,
        forward: bool
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Handle refraction or reflection at a flat water surface.

        Args:
            positions: (N, 3) current photon positions
            next_positions: (N, 3) next positions
            directions: (N, 3) current directions (normalized)
            n1: refractive index of current medium (e.g. air)
            n2: refractive index of next medium (e.g. water)

        Returns:
            - updated positions (interaction points)
            - updated directions (after refraction/reflection)
            - interaction_points (intersections with water surface)
        """

        normals = np.tile(np.array([[0, -1 if invert_normals else 1, 0]], dtype=np.float32), (len(positions), 1))

        step = next_positions - positions
        y0 = positions[:, 1]
        dy = step[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            f = (self.water_surface_y - y0) / dy
            f = np.clip(f, 0.0, 1.0)

        intersection = positions + f[:, np.newaxis] * step

        # Step 2: Compute refraction or reflection
        eta = n1 / n2
        I = directions
        N = normals

        cos_i = -np.einsum('ij,ij->i', I, N)
        k = 1.0 - eta**2 * (1.0 - cos_i**2)
        tir_mask = k < 0.0

        # Total internal reflection → reflect(I, N)
        reflected_dirs = I - 2 * np.einsum('ij,ij->i', I, N)[:, None] * N

        # Refraction direction
        sqrt_k = np.sqrt(np.maximum(k, 0.0))
        refracted_dirs = eta * I + (eta * cos_i - sqrt_k)[:, None] * N

        # Final direction: use reflection for TIR, else refraction
        new_directions = np.where(tir_mask[:, None], reflected_dirs, refracted_dirs)

        step_lengths = np.linalg.norm(step, axis=1)
        remaining_fraction = 1.0 - f
        # TODO: Fix velocity under water?
        remaining_step = new_directions * (remaining_fraction[:, None] * step_lengths[:, None])

        final_positions = intersection + remaining_step

        return final_positions, new_directions, intersection



if __name__ == "__main__":
    rng = np.random.default_rng(secrets.randbits(128))
    simulation = Simulation(rng)
    start = time.time()
    photons_per_batch = 10_000
    batches = 5
    steps = 1250
    visualize_paths = 250

    for i in range(batches):
        simulation.simulate_batch(photons_per_batch, steps, True, 0)
        if (i+1) % 10 == 0:
            print(f"{i+1} in {(time.time() - start):.2f} s")
    elapsed = time.time() - start
    print(f"time: {elapsed:.6f} seconds")

    print(f"{len(simulation.photons_list)} entries in photon list, {(sys.getsizeof(simulation.photons_list) / 1024):.2f} KiB")

    simulation.photon_np_array = np.zeros(len(simulation.photons_list), dtype=[("position", np.float32, 3), ("distance", np.float32, 3), ("energy", np.float32), ("time", np.float32), ("reflection", bool)])
    simulation.photon_np_array[...] = simulation.photons_list
    positions = np.array(simulation.photon_np_array["position"])
    simulation.photon_tree = KDTree(positions)
    
    # visualize_photons = simulation.photon_np_array[rng.integers(0, len(simulation.photon_np_array), 1_000)]
    # plot_scatter_2d(visualize_photons["position"][:, 0], visualize_photons["position"][:, 2], ylabel="Z-Axis")

    start = time.time()
    photons_per_batch = 10_000
    batches = 1
    steps = 1250

    for i in range(batches):
        simulation.simulate_batch(photons_per_batch, steps, False, 0)
        if (i+1) % 20 == 0:
            print(f"{i+1} in {(time.time() - start):.2f} s")
    elapsed = time.time() - start
    print(f"time: {elapsed:.6f} seconds")
    
    plot_2d(simulation.return_waveform, ylabel="Intensity", xlabel="Sample")