from typing import List, Tuple
import numpy as np
import time
import secrets
from numpy.typing import NDArray

from camera import Camera
from laser import Laser
from alb_sim.utils.photon_position_state import PhotonPositionState
from utils.visualize_paths import visualize_photon_paths
from world import World
import utils.numpy_vector as np_vec

laser_settings = Laser()
camera_settings = Camera()
world_settings = World(laser_settings)

time_step = 1 / camera_settings.sample_rate
water_surface_y = -camera_settings.flying_height
seafloor_y = -camera_settings.distance_seafloor_flying_height

def evaluate_state(
    y_current: float,
    y_next: float
) -> PhotonPositionState:
    global water_surface_y, seafloor_y

    if (y_current > water_surface_y) and (y_next <= water_surface_y):
        return PhotonPositionState.ENTERING_WATER
    if (y_current <= water_surface_y) and (y_next > seafloor_y):
        return PhotonPositionState.IN_WATER
    if (y_current > seafloor_y) and (y_next <= seafloor_y):
        return PhotonPositionState.HITTING_SEAFLOOR
    if (y_current <= water_surface_y) and (y_next > water_surface_y):
        return PhotonPositionState.EXITING_WATER

    return PhotonPositionState.IN_AIR

def simulate_photon_step(
    position: NDArray[np.float32],    # (N, 3)
    direction: NDArray[np.float32],   # (N, 3)
    velocity: float,   # (N, 3)
) -> Tuple[float, float, float]:
    global time_step
    """
    Simulates a single step of photon movement and updates direction/position based on medium transitions.
    Returns updated positions, directions, states, and previous positions for logging.
    """

    next_position = position + direction * velocity * time_step

    y_current = position[1]
    y_next = next_position[1]
    state = evaluate_state(y_current, y_next)

    if state == PhotonPositionState.HITTING_SEAFLOOR:
        normal = np.array([0, 1, 0], dtype=np.float32)
        direction = np_vec.reflect_vector(direction, normal)
    position = next_position

    return position, direction, velocity #, states, next_positions

if __name__ == "__main__":
    start = time.time()
    N = 100_000
    directions = np_vec.sample_directions_in_cone(np.array(laser_settings.laser_direction), laser_settings.laser_divergence_angle, N, np.random.default_rng(secrets.randbits(128)))

    for i, direction in enumerate(directions):
        position = np.array([0, 0, 0])
        velocity = world_settings.light_speed_air
        for step in range(700):
            position, direction, velocity = simulate_photon_step(position, direction, velocity)

    elapsed_linear = time.time() - start
    print(f"linear time: {elapsed_linear:.6f} seconds")