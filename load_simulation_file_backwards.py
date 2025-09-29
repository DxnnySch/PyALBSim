from simulation import Simulation
import secrets
import numpy as np
from scipy.spatial import KDTree
import cProfile
import pstats
import time

from utils.plot_2d import plot_2d
from utils.plot_histogram import plot_histogram

rng = np.random.default_rng(secrets.randbits(128))
simulation = Simulation(rng)

start = time.time()

simulation.photon_np_array = np.load("photon-map_625,000-photons.npy", allow_pickle=False)
positions = simulation.photon_np_array["position"]
simulation.photon_tree = KDTree(positions)

print(f"loaded numpy file after {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min")
# plot_histogram(simulation.photon_np_array[simulation.photon_np_array["reflection"] == False]["time"] / 10, bins=1250, title="Time of stored reflection photons", xlabel="Time")

start = time.time()
photons_per_batch = 10_000
batches = 10
steps = 1000

profiler = cProfile.Profile()
profiler.enable()

for i in range(batches):
    simulation.simulate_batch(photons_per_batch, steps, False, 0)
    print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((time.time() - start) / 60 / (i+1) * (batches - (i + 1))):.2f} min")

profiler.disable()

elapsed = time.time() - start
print(f"time backward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(30)  # Top 30 functions

plot_2d(simulation.return_waveform, ylabel="Intensity", xlabel="Sample")
photons_reflections = np.array(simulation.photons_found_bottom_reflection)
plot_histogram(photons_reflections[photons_reflections < 400], bins = 400, title="Number of Photons found at Reflections", xlabel="Photons in Radius")
photons_scatters = np.array(simulation.photons_found_scatter)
plot_histogram(photons_scatters[photons_scatters < 400], bins = 400, title="Number of Photons found at Scatters", xlabel="Photons in Radius")