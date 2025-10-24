import numpy as np

from camera import Camera
from laser import Laser
from utils.plot_2d import plot_2d
from utils.plot_histogram import plot_histogram
from world import World
rng = np.random.default_rng(42)

camera_settings = Camera()
laser_settings = Laser(camera_settings, 10)
world_settings = World(laser_settings, camera_settings)

rand_vals = rng.random(10000).astype(np.float32)
# matlab uses: rand_vals = 1 - np.exp(-rng.random(10000).astype(np.float32))
dist = -np.log(rand_vals) / world_settings.lidar_attenuation_coefficient

print(f"mean: {dist.mean()}")
print(f"median: {np.median(dist)}")

plot_histogram(dist, bins=500, title="Distribution of scatter distances", xlabel="Distance (m)")