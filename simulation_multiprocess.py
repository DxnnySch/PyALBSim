import multiprocessing as mp
from multiprocessing import shared_memory
import secrets, time, sys
import numpy as np
from scipy.spatial import KDTree

from simulation import Simulation


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
# Backward worker
# ==============================
def init_backward_worker(shm_name, shape, dtype):
    """
    Each backward worker attaches to shared photon map in shared memory.
    Builds a local KDTree for queries.
    """
    global photon_array, photon_tree


def backward_worker(args):
    try:
        photons_per_batch, steps, seed, shm_name, shape, dtype = args
        rng = np.random.default_rng(seed)
        sim = Simulation(rng, steps)

        existing_shm = shared_memory.SharedMemory(name=shm_name)
        photon_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        photon_tree = KDTree(np.array(photon_array["position"]))

        sim.photon_np_array = photon_array
        sim.photon_tree = photon_tree

        print("Starting simulation")

        sim.simulate_batch(photons_per_batch, steps, False, 0)
        return sim.return_waveform
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def backward_worker_local(args):
    photons_per_batch, steps, seed, photon_array_local = args
    """
    photon_array_local: a full NumPy array of photons for this worker
    """
    rng = np.random.default_rng(seed)
    sim = Simulation(rng, steps)

    # Attach local photon map and build KDTree once
    sim.photon_np_array = photon_array_local
    sim.photon_tree = KDTree(np.array(photon_array_local["position"]))

    # Run the batch
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
    forward_batches = 8
    forward_args = [(photons_per_batch, steps, secrets.randbits(64)) for _ in range(forward_batches)]

    print("Starting forward pass...")
    with mp.Pool(processes=nproc) as pool:
        forward_results = run_with_progress(pool, forward_worker, forward_args,
                                            "Forward", forward_batches)

    photon_np_array = np.concatenate([res for res in forward_results if res is not None])
    print(f"Photon map has {len(photon_np_array):,} entries, size {(sys.getsizeof(photon_np_array)/1024):.2f} KiB")
    # np.save(f"photon-map_{len(photon_np_array):,}-photons.npy", photon_np_array)

    # ------------------------------
    # Backward pass (parallel + shared memory + progress)
    # ------------------------------
    photons_per_batch = 10_000
    backward_batches = 8  # ideally multiple of nproc

    # shm = shared_memory.SharedMemory(create=True, size=photon_np_array.nbytes)
    # shared_photon_array = np.ndarray(photon_np_array.shape, dtype=photon_np_array.dtype, buffer=shm.buf)
    # shared_photon_array[:] = photon_np_array[:]
    # backward_args = [(photons_per_batch, steps, secrets.randbits(64), shm.name, photon_np_array.shape, photon_np_array.dtype) for _ in range(backward_batches)]

    # print("Starting backward pass...")
    # with mp.Pool(processes=nproc) as pool:
    #     backward_results = run_with_progress(pool, backward_worker, backward_args,
    #                                          "Backward", backward_batches)

    # total_waveform = np.sum(backward_results, axis=0)

    # Split photon map for each worker (could just give the same copy to all workers)
    worker_photon_arrays = [photon_np_array.copy() for _ in range(nproc)]

    # Build args list for all batches
    backward_args = []
    for i in range(backward_batches):
        seed = secrets.randbits(64)
        worker_array = worker_photon_arrays[i % nproc]  # reuse local copy per process
        backward_args.append((photons_per_batch, steps, seed, worker_array))

    with mp.Pool(processes=nproc) as pool:
        backward_results = run_with_progress(pool, backward_worker_local, backward_args,
                                             "Backward", backward_batches)
        # results = []
        # for i, res in enumerate(pool.imap_unordered(backward_worker_local, backward_args), 1):
        #     results.append(res)
        #     print(f"Backward batch {i}/{backward_batches} complete")

    total_waveform = np.sum(backward_results, axis=0)

    # shm.close()
    # shm.unlink()

    print("Simulation finished.")