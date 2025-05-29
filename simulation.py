import numpy as np
import time
import secrets
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
        self.ratio = 3 # Photon packets within ?? spot radii are used for simulation. Here is 3. TODO: Rewrite
        self.photon_survival_threshold_weight = 0.0001 # epsilon

        self.laser_settings = Laser()
        self.camera_settings = Camera()
        self.world_settings = World(self.laser_settings)

        self.time_step = 1 / self.camera_settings.sample_rate
        self.water_surface_y = -self.camera_settings.flying_height
        self.seafloor_y = -self.camera_settings.distance_seafloor_flying_height,

    def simulate(self, photons: int, steps: int, full_history: False, num_samples_history: int = 0):
        # initialise photons, in batches?
        N = photons
        history_samples = self.rng.integers(0, photons, num_samples_history)
        
        positions = np.zeros((N, 3), dtype=np.float32)
        
        # All directions pointing down
        # directions = np.tile(np.array([[0, -1, 0]], dtype=np.float32), (N, 1))
        
        # Directions down with little randomness
        # angles = np.random.uniform(-0.1, 0.1, size=(N, 2))
        # directions = np.stack([
        #     np.sin(angles[:, 0]),
        #     -np.ones(N),
        #     np.sin(angles[:, 1])
        # ], axis=1)
        # directions = np_vec.normalize(directions)
        # print(directions)
        
        directions = np_vec.sample_directions_in_cone(np.array(self.laser_settings.laser_direction), self.laser_settings.laser_divergence_angle, N, self.rng)
        
        velocities = np.full(N, self.world_settings.light_speed_air, dtype=np.float32)
        
        histories: List[List[NDArray[np.float32]]] = [[] for _ in range(num_samples_history)]
        
        if full_history:
            for i in range(N):
                histories[i].append(positions[i].copy())
        else:
            for i, pos in enumerate(positions[history_samples]):
                histories[i].append(pos.copy())


        for _ in range(steps):
            positions, directions, velocities = self.simulate_photon_step(positions, directions, velocities)
            if full_history:
                for i in range(N):
                    histories[i].append(positions[i].copy())
            else:
                for i, pos in enumerate(positions[history_samples]):
                    histories[i].append(pos.copy())

        return histories
        # step through photons
            # for each photon, determine location and get strategy function
            # put photon through strategy function
            # repeat
            # maybe culling?

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
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Simulates a single step of photon movement and updates direction/position based on medium transitions.
        Returns updated positions, directions, states, and previous positions for logging.
        """

        next_positions = positions + directions * velocities[:,np.newaxis] * self.time_step

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
            # normals = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (enter_idx.size, 1))
            # directions[enter_idx] = refract(directions[enter_idx], normals, eta=1.0 / 1.33)
            positions[enter_idx] = next_positions[enter_idx]

        if hit_floor_idx.size > 0:
            normals = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (hit_floor_idx.size, 1))
            directions[hit_floor_idx] = np_vec.reflect_batch(directions[hit_floor_idx], normals)
            # TODO: Calculate intersection with floor and set position accordingly (if dist to floor = n, then set position = newdir * (regular_dist - n))
            positions[hit_floor_idx] = next_positions[hit_floor_idx]

        if exit_idx.size > 0:
            # normals = np.tile(np.array([[0, -1, 0]], dtype=np.float32), (exit_idx.size, 1))
            # directions[exit_idx] = refract(directions[exit_idx], normals, eta=1.33 / 1.0)
            positions[exit_idx] = next_positions[exit_idx]

        return positions, directions, velocities #, states, next_positions

if __name__ == "__main__":
    simulation = Simulation(np.random.default_rng(secrets.randbits(128)))
    start = time.time()
    for _ in range(5):
        histories = simulation.simulate(20_000, 700, False, 10)
    elapsed_vectorized = time.time() - start
    print(f"Vectorized time: {elapsed_vectorized:.6f} seconds")
    visualize_photon_paths(histories, simulation.water_surface_y, simulation.seafloor_y)