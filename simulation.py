import numpy as np
import time
import secrets
import math
from typing import List, Tuple
from numpy.typing import NDArray

from camera import Camera
from laser import Laser
from world import World
from utils.photon_state import PhotonState
import utils.numpy_vector as np_vec
from utils.visualize_paths import visualize_photon_paths

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
        self.sensor_direction = np.array(self.laser_settings.laser_direction, dtype=np.float32) * -1

        # self.seafloor_reflection_function_batch = lambda direction, normals: np_vec.reflect_batch(direction, normals)
        self.seafloor_reflection_function_batch = lambda direction, normals: np_vec.heuristic_sample_batch(normals, self.rng)
        
        self.total_close = 0

    def simulate_batch(self, num_photons: int, steps: int, num_samples_history: int = 0):
        N = num_photons
        history_samples = self.rng.integers(0, num_photons, num_samples_history)

        positions = np.zeros((N, 3), dtype=np.float32)

        directions = np_vec.sample_directions_in_cone(np.array(self.laser_settings.laser_direction), self.laser_settings.laser_divergence_angle, N, self.rng)

        velocities = np.full(N, self.world_settings.light_speed_air, dtype=np.float32)
        energies = np.full(N, 1, dtype=np.float32)

        full_histories: List[List[NDArray[np.float32]]] = [[] for _ in range(num_samples_history)]

        for i, pos in enumerate(positions[history_samples]):
            full_histories[i].append(pos.copy())


        for _ in range(steps):
            positions, directions, velocities, energies, interaction_points = self.simulate_photon_step(positions, directions, velocities, energies)
            for i, idx in enumerate(history_samples):
                inter = interaction_points[idx]
                if not np.isnan(inter).any():
                    full_histories[i].append(inter.copy())  # intermediate point
                full_histories[i].append(positions[idx].copy())  # final position of this step

        print("number of vectors that are close to laser direction: ", len(np.nonzero(np.einsum('ij,j->i', directions, np_vec.normalize_vector(np.array(self.laser_settings.laser_direction, dtype=np.float32)*-1)) >= math.cos(self.laser_settings.field_of_view))[0]))
        self.total_close += len(np.nonzero(np.einsum('ij,j->i', directions, np_vec.normalize_vector(np.array(self.laser_settings.laser_direction, dtype=np.float32)*-1)) >= math.cos(self.laser_settings.field_of_view))[0]);
        
        # filtered_histories = [full_histories[i] for i in range(len(full_histories)) if (np.einsum('ij,j->i', directions[history_samples], np_vec.normalize_vector(np.array(self.laser_settings.laser_direction, dtype=np.float32))*-1) >= math.cos(self.laser_settings.field_of_view))[i]]

        return full_histories

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
            refracted_positions, refracted_directions, intersection_points = self.handle_enter_exit_water_refraction(positions[enter_idx], next_positions[enter_idx], directions[enter_idx], 1, self.world_settings.refractive_index_water, False)
            positions[enter_idx] = refracted_positions
            directions[enter_idx] = refracted_directions
            interaction_points[enter_idx] = intersection_points

        if hit_floor_idx.size > 0:
            reflected_positions, reflected_directions, reflected_energies, intersection_points = self.handle_seafloor_reflection(positions[hit_floor_idx], next_positions[hit_floor_idx], directions[hit_floor_idx], energies[hit_floor_idx])
            # print(reflected_positions.shape)
            positions[hit_floor_idx] = reflected_positions
            directions[hit_floor_idx] = reflected_directions
            energies[hit_floor_idx] = reflected_energies
            interaction_points[hit_floor_idx] = intersection_points

        if exit_idx.size > 0:
            refracted_positions, refracted_directions, intersection_points = self.handle_enter_exit_water_refraction(positions[exit_idx], next_positions[exit_idx], directions[exit_idx], self.world_settings.refractive_index_water, 1, True)
            positions[exit_idx] = refracted_positions
            directions[exit_idx] = refracted_directions
            interaction_points[exit_idx] = intersection_points

        return positions, directions, velocities, energies, interaction_points

    def handle_seafloor_reflection(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        energy: NDArray[np.float32],
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
        reflected_dirs, pdf = np_vec.sample_directions_in_cone_with_pdf(
            self.sensor_direction.astype(np.float32),
            math.radians(22.5), # np.pi / 4,
            len(positions),
            self.rng
        )

        # Ensure all directions are normalized
        reflected_dirs = reflected_dirs / np.linalg.norm(reflected_dirs, axis=1, keepdims=True)

        # Compute cos(theta) = dot(N, outgoing_dir)
        cos_theta = np_vec.dot_batch(normals, reflected_dirs)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)

        # Lambertian reflection energy with importance sampling correction
        out_energy = energy * ((1.0 / np.pi) * cos_theta / pdf)

        step_lengths = np.linalg.norm(step, axis=1)
        remaining_fraction = 1.0 - f
        remaining_step = reflected_dirs * (remaining_fraction[:, np.newaxis] * step_lengths[:, np.newaxis])

        final_positions = intersection + remaining_step

        return final_positions, reflected_dirs, out_energy, intersection

    def handle_enter_exit_water_refraction(
        self,
        positions: NDArray[np.float32],
        next_positions: NDArray[np.float32],
        directions: NDArray[np.float32],
        n1: float,
        n2: float,
        invert_normals: bool
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
        remaining_step = new_directions * (remaining_fraction[:, None] * step_lengths[:, None])

        final_positions = intersection + remaining_step

        return final_positions, new_directions, intersection



if __name__ == "__main__":
    simulation = Simulation(np.random.default_rng(secrets.randbits(128)))
    start = time.time()
    total_photons = 100_000_000
    batches = 500
    steps = 700
    visualize_paths = 250

    for i in range(batches):
        histories = simulation.simulate_batch(total_photons // batches, steps, visualize_paths if i == batches - 1 else 0)
        if (i+1) % 20 == 0:
            print(f"after {i+1} in {(time.time() - start):.2f} s: {simulation.total_close} are close")
    elapsed_vectorized = time.time() - start
    print(f"Vectorized time: {elapsed_vectorized:.6f} seconds")
    print(f"total close: {simulation.total_close}")
    visualize_photon_paths(histories, simulation.water_surface_y, simulation.seafloor_y)