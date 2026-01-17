from time import perf_counter
import numpy as np
import secrets

from alb_sim.photon_mapping.build_photon_map_data import build_photon_map_data
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.photon_mapping.print_photon_map_stats import photon_map_stats
from alb_sim.utils.types import Array

from alb_sim.config.run import RunConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.core.simulation import Simulation


def linear_forward(simulation: Simulation, photons_per_batch: int, batches: int):
    print("Starting forwards pass")
    start_time = perf_counter()

    for i in range(batches):
        round_start = perf_counter()
        simulation.simulate_batch(photons_per_batch, forward=True)
        since_start = perf_counter() - start_time
        since_round = perf_counter() - round_start
        print(
            f"forward batch {i+1}/{batches} in {since_round:.2f} s = {(since_round / 60):.2f} min",
            end=", ",
        )
        print(
            f"{(since_start / 60):.2f} min since start, ETA: {(((since_start / 60) / (i+1)) * (batches - (i + 1))):.2f} min"
        )
    print("Finished forwards pass")


def linear_backward(simulation: Simulation, photons_per_batch: int, batches: int):
    print("Starting backwards pass")
    start_time = perf_counter()

    for i in range(batches):
        round_start = perf_counter()
        simulation.simulate_batch(photons_per_batch, forward=False)
        since_start = perf_counter() - start_time
        since_round = perf_counter() - round_start
        print(
            f"backward batch {i+1}/{batches} in {since_round:.2f} s = {(since_round / 60):.2f} min",
            end=", ",
        )
        print(
            f"{(since_start / 60):.2f} min since start, ETA: {(((since_start / 60) / (i+1)) * (batches - (i + 1))):.2f} min"
        )
    print("Finished backwards pass")


def run_linear(
    simulation_config: SimulationConfig,
    run_config: RunConfig,
    rng: np.random.Generator = None,
) -> dict[PhotonType, Array]:
    if rng is None:
        rng = np.random.default_rng(secrets.randbits(128))

    simulation = Simulation(simulation_config, rng)

    linear_forward(
        simulation, run_config.photons_per_batch_forward, run_config.batches_forward
    )

    photon_maps_data = build_photon_map_data(simulation.photon_storage)
    print("\n" + photon_map_stats(photon_maps_data))
    for photon_type, data in photon_maps_data.items():
        simulation.photon_maps[photon_type] = PhotonMapIndex(data)

    linear_backward(
        simulation, run_config.photons_per_batch_backward, run_config.batches_backward
    )

    return simulation.return_waveform
