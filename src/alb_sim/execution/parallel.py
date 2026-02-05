from datetime import datetime
import math
import secrets
from collections import defaultdict
from time import perf_counter
from typing import Union

import multiprocess as mp
import numpy as np

from alb_sim.config.run import RunConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.core.simulation import Simulation
from alb_sim.photon_mapping.build_photon_map_data import build_photon_map_data
from alb_sim.photon_mapping.photon_map_data import PhotonMapData
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.photon_mapping.print_photon_map_stats import photon_map_stats
from alb_sim.utils.types import Array


def merge_results(results: list[dict]):
    merged = defaultdict(list)

    for result in results:
        for photon_type, arr in result.items():
            merged[photon_type].append(arr)

    return {photon_type: np.sum(arrs, axis=0) for photon_type, arrs in merged.items()}


# ==============================
# Forward worker
# ==============================
def forward_worker(args: tuple[SimulationConfig, RunConfig, Union[int, None]]):
    start = perf_counter()
    config, run_config, seed = args
    if seed is None:
        seed = secrets.randbits(64)
    rng = np.random.default_rng(seed)
    sim = Simulation(config, rng)

    sim.simulate_batch(run_config.photons_per_batch_forward, forward=True)
    duration = perf_counter() - start
    return sim.photon_storage, duration


# ==============================
# Backward worker initializer
# ==============================
def backward_worker_init(shared_photon_maps_data: dict[PhotonType, PhotonMapData]):
    """
    Called once per worker process.
    Builds a KDTree for this worker and stores photon array globally.
    """
    global photon_maps
    photon_maps = {}
    for photon_type, data in shared_photon_maps_data.items():
        photon_maps[photon_type] = PhotonMapIndex(data)


# ==============================
# Backward worker batch
# ==============================
def backward_worker_batch(args: tuple[SimulationConfig, RunConfig, Union[int, None]]):
    start = perf_counter()
    config, run_config, seed = args
    if seed is None:
        seed = secrets.randbits(64)
    rng = np.random.default_rng(seed)
    sim = Simulation(config, rng)

    # Attach global photon map + tree
    sim.photon_maps = photon_maps

    sim.simulate_batch(run_config.photons_per_batch_forward, forward=False)
    duration = perf_counter() - start
    return sim.return_waveform, duration


# ==============================
# Progress helper
# ==============================
def run_with_progress(pool, worker, args_list, label, num_batches, num_processes):
    results = []
    times = []
    start_time = perf_counter()
    for i, (res, task_time) in enumerate(pool.imap_unordered(worker, args_list), 1):
        results.append(res)
        times.append(task_time)
        since_start = perf_counter() - start_time
        eta = np.mean(times) * (
            math.ceil(num_batches / num_processes) - math.ceil(i / num_processes)
        )
        print(
            f"{label} batch {i}/{num_batches} in {task_time:.2f} s = {(task_time / 60):.2f} min ({datetime.now().strftime('%H:%M:%S')})",
            end=", ",
        )
        print(f"{(since_start / 60):.2f} min since start, ETA: {(eta / 60):.2f} min")
    return results


# ==============================
# Main driver
# ==============================


def run_parallel(
    simulation_config: SimulationConfig,
    run_config: RunConfig,
    seed: Union[int, None] = None,
) -> dict[PhotonType, Array]:

    # ------------------------------
    # Forward pass
    # ------------------------------

    print("Starting forwards pass")
    forward_args = [
        (simulation_config, run_config, seed) for _ in range(run_config.batches_forward)
    ]
    with mp.Pool(processes=run_config.processes) as pool:
        forward_results = run_with_progress(
            pool,
            forward_worker,
            forward_args,
            "forward",
            run_config.batches_forward,
            run_config.processes,
        )

    # ------------------------------
    # Merge result
    # ------------------------------

    merged_storage = PhotonStorage()
    for forward_result in forward_results:
        if forward_result is not None:
            merged_storage.merge(forward_result)
    photon_maps_data = build_photon_map_data(merged_storage)
    print("\n" + photon_map_stats(photon_maps_data) + "\n")

    # ------------------------------
    # Backward pass
    # ------------------------------

    print("Starting backward pass...")
    backward_args = [
        (simulation_config, run_config, seed)
        for _ in range(run_config.batches_backward)
    ]
    start_time = perf_counter()
    with mp.Pool(
        processes=run_config.processes,
        initializer=backward_worker_init,
        initargs=(photon_maps_data,),
    ) as pool:
        print(
            f"initialized workers in {(perf_counter()-start_time):.2f} s ({(perf_counter()-start_time)/60:.2f} min)"
        )
        backward_results = run_with_progress(
            pool,
            backward_worker_batch,
            backward_args,
            "backward",
            run_config.batches_backward,
            run_config.processes,
        )

    waveform = merge_results(backward_results)

    return waveform
