import math
import multiprocessing as mp
import secrets, time, sys
import numpy as np
from scipy.spatial import KDTree

from simulation import Simulation
from utils.plot_2d import plot_2d, plot_2d_better

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
    return np.concatenate(sim.photon_batches) if sim.photon_batches else None


# ==============================
# Backward worker initializer
# ==============================
def backward_worker_init(shared_photon_array):
    """
    Called once per worker process.
    Builds a KDTree for this worker and stores photon array globally.
    """
    global photon_array, photon_tree
    photon_array = shared_photon_array
    photon_tree = KDTree(np.array(photon_array["position"]))
    # print(f"Worker PID {mp.current_process().pid}: KDTree built")


# ==============================
# Backward worker batch
# ==============================
def backward_worker_batch(args):
    photons_per_batch, steps, seed, options = args
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps, options)

    # Attach global photon map + tree
    sim.photon_np_array = photon_array
    sim.photon_tree = photon_tree

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
    num_rounds = 5
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
        nproc = 16

        # ------------------------------
        # Forward pass (parallel + progress)
        # ------------------------------
        photons_per_batch = 15_000
        forward_batches = 16*2
        forward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(forward_batches)]

        print("Starting forward pass...")
        with mp.Pool(processes=nproc) as pool:
            forward_results = run_with_progress(pool, forward_worker,
                                                forward_args, "Forward", forward_batches)

        photon_np_array = np.concatenate([res for res in forward_results if res is not None])
        print(f"Photon map has {len(photon_np_array):,} entries, "
            f"size {(sys.getsizeof(photon_np_array)/1024/1024):.2f} MiB")

        # ------------------------------
        # Backward pass (parallel + persistent KDTree + progress)
        # ------------------------------
        photons_per_batch = 15_000
        backward_batches = 16*50

        # Build args list for backward batches (seeds only)
        backward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(backward_batches)]

        print("Starting backward pass...")
        start_time = time.time()
        with mp.Pool(processes=nproc,
                    initializer=backward_worker_init,
                    initargs=(photon_np_array,)) as pool:
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
    plot_2d_better(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample", save_path="images/latest_waveform_zoomed_v19.png", show=True, params=options, xlim=(270, 330))
