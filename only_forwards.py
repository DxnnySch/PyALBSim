import cProfile
import pstats
import secrets
import time
import numpy as np
from line_profiler import profile

from simulation import Simulation

simulation = None
rng = np.random.default_rng(secrets.randbits(128))

@profile
def main():
    global simulation, rng
    simulation = Simulation(rng)
    start = time.time()
    photons_per_batch = 25_000
    batches = 10
    steps = 1250
    visualize_paths = 0

    # profiler = cProfile.Profile()
    # profiler.enable()

    for i in range(batches):
        simulation.simulate_batch(photons_per_batch, steps, True, visualize_paths if i == batches - 1 else 0)
        if (i+1) % 5 == 0:
            print(f"{i+1} in {(time.time() - start):.2f} s = {((time.time() - start) / 60):.2f} min, estimated remaining: {((((time.time() - start) / 60) / (i+1)) * (batches - (i + 1))):.2f} min")

    # profiler.disable()

    elapsed = time.time() - start
    print(f"time forward: {elapsed:.6f} seconds = {(elapsed / 60):.2f} min")

    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats(30)  # Top 30 functions

    simulation.photon_np_array = np.concatenate(simulation.photon_batches)

if __name__ == "__main__":
    main()