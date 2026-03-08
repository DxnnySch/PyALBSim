import math
import multiprocessing as mp
import os
import secrets, time, sys
import numpy as np

from simulation import Simulation
from alb_sim.photon_mapping.build_photon_map_data import build_photon_map_data
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.print_photon_map_stats import print_photon_map_stats
from alb_sim.plotting.plot_waveform import plot_waveform, plot_2d_better

# ==============================
# Globals for backward workers
# ==============================
# photon_array = None
# photon_tree = None

# ==============================
# Forward worker
# ==============================
def forward_worker(args):
    photons_per_batch, steps, seed, options = args
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps, options)

    sim.simulate_batch(photons_per_batch, steps, True, 0)
    return sim.photon_storage


# ==============================
# Backward worker initializer
# ==============================
def backward_worker_init(shared_photon_maps_data):
    """
    Called once per worker process.
    Builds a KDTree for this worker and stores photon array globally.
    """
    global photon_maps
    photon_maps = {} # dict[PhotonType, PhotonMapIndex]
    for photon_type, data in shared_photon_maps_data.items():
        photon_maps[photon_type] = PhotonMapIndex(data)
    # print(f"Worker PID {mp.current_process().pid}: KDTree built")


# ==============================
# Backward worker batch
# ==============================
def backward_worker_batch(args):
    photons_per_batch, steps, seed, options = args
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps, options)

    # Attach global photon map + tree
    sim.photon_maps = photon_maps

    sim.simulate_batch(photons_per_batch, steps, False, 0)
    return sim.return_waveform


# ==============================
# Progress helper
# ==============================
def run_with_progress(pool, worker, args_list, label, total_batches):
    results = []
    start = time.time()
    for i, res in enumerate(pool.imap_unordered(worker, args_list), 1):
        results.append(res)
        elapsed = time.time() - start
        eta = (elapsed / i) * (total_batches - i)
        print(f"{label} batch {i}/{total_batches} complete in {elapsed:.2f} s "
              f"({elapsed/60:.2f} min), ETA {eta/60:.2f} min")
    return results


# ==============================
# Main driver
# ==============================
if __name__ == "__main__":
    total_waveform = np.array([1])
    total_time = time.time()
    num_rounds = 1
    for i in range(num_rounds):
        start = time.time()
        options = {
            "flying_height": 135,
            "water_depth": 3,
            "sample_rate": 2_000_000_000,
            "sample_multiplier": 10,
            # "absorption_coefficient": 0.114,
            # "total_scattering_coefficient": 0.037
        }
        distance = ((options["flying_height"] + options["water_depth"]) / math.cos(math.radians(15)))
        steps = int(1.5 * (distance * round(options["sample_rate"])) / 2.998e8)
        nproc = os.process_cpu_count() - 1

        # ------------------------------
        # Forward pass (parallel + progress)
        # ------------------------------
        photons_per_batch = 10_000
        forward_batches = (os.process_cpu_count() - 1) * 1
        forward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(forward_batches)]

        print("Starting forward pass...")
        with mp.Pool(processes=nproc) as pool:
            forward_results = run_with_progress(pool, forward_worker,
                                                forward_args, "Forward", forward_batches)

        merged_storage = PhotonStorage()
        for forward_result in forward_results:
            if forward_result is not None:
                merged_storage.merge(forward_result)
        photon_maps_data = build_photon_map_data(merged_storage)
        print()
        print_photon_map_stats(photon_maps_data)
        print()

        # ------------------------------
        # Backward pass (parallel + persistent KDTree + progress)
        # ------------------------------
        photons_per_batch = 5_000
        backward_batches = os.process_cpu_count() - 1

        # Build args list for backward batches (seeds only)
        backward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(backward_batches)]

        print("Starting backward pass...")
        start_time = time.time()
        with mp.Pool(processes=nproc,
                    initializer=backward_worker_init,
                    initargs=(photon_maps_data,)) as pool:
            print(f"initialized workers in {(time.time()-start_time):.2f} s ({(time.time()-start_time)/60:.2f} min)")
            backward_results = run_with_progress(pool, backward_worker_batch,
                                                backward_args, "Backward", backward_batches)

        round_waveform = np.sum(backward_results, axis=0)
        if i == 0:
            total_waveform = round_waveform
        else:
            total_waveform += round_waveform
        print(f"Round {i+1} of {num_rounds} finished.")
        print(f"Round time {(time.time()-start):.2f} s ({(time.time()-start)/60:.2f} min)")
    print("Simulation finished.")
    print(f"Total time {(time.time()-total_time):.2f} s ({(time.time()-total_time)/60:.2f} min)")
    import matplotlib
    matplotlib.use("TkAgg")
    plot_2d_better(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample", show=True, params=options, xlim=(1860,1930)) # , save_path="images/latest_waveform_zoomed_v19.png"
