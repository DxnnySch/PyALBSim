import multiprocessing as mp
import secrets, time, sys
import numpy as np
from scipy.spatial import KDTree

from simulation import Simulation
from utils.plot_2d import plot_2d

# ==============================
# Globals for backward workers
# ==============================
# photon_array = None
# photon_tree = None

# ==============================
# Forward worker
# ==============================
def forward_worker(args):
    photons_per_batch, steps, seed = args
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps)

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
    print(f"Worker PID {mp.current_process().pid}: KDTree built")


# ==============================
# Backward worker batch
# ==============================
def backward_worker_batch(args):
    photons_per_batch, steps, seed = args
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps)

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
    steps = 1000
    nproc = 4

    # ------------------------------
    # Forward pass (parallel + progress)
    # ------------------------------
    photons_per_batch = 15_000
    forward_batches = 12
    forward_args = [(photons_per_batch, steps, secrets.randbits(64))
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
    photons_per_batch = 10_000
    backward_batches = 12

    # Build args list for backward batches (seeds only)
    backward_args = [(photons_per_batch, steps, secrets.randbits(64))
                     for _ in range(backward_batches)]

    print("Starting backward pass...")
    start_time = time.time()
    with mp.Pool(processes=nproc,
                 initializer=backward_worker_init,
                 initargs=(photon_np_array.copy(),)) as pool:
        print(f"initialized workers in {(time.time()-start_time):.2f} s ({(time.time()-start_time)/60:.2f} min)")
        backward_results = run_with_progress(pool, backward_worker_batch,
                                             backward_args, "Backward", backward_batches)

    total_waveform = np.sum(backward_results, axis=0)
    print("Simulation finished.")
    plot_2d(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample")
