from enum import Enum
from typing import Mapping
from tabulate import tabulate

from utils.photon_mapping.photon_map_data import PhotonMapData

# TODO: temp, add import later in correct place
PhotonType = Enum("PhotonType", [("BOTTOM_REFLECTION", 0), ("SCATTER", 1), ("SURFACE_REFLECTION", 2)])
def print_photon_map_stats(
    photon_maps: Mapping[PhotonType, PhotonMapData],
    tablefmt: str = "github"
) -> None:
    rows = []

    total_photons = 0
    total_mem = 0.0

    for photon_type, data in photon_maps.items():
        n = data.positions.shape[0]

        mem_pos = data.positions.nbytes / 1024
        mem_dir = data.directions.nbytes / 1024
        mem_eng = data.energies.nbytes / 1024
        mem_time = data.times.nbytes / 1024
        mem_ref = data.already_reflected.nbytes / 1024

        mem_sum = mem_pos + mem_dir + mem_eng + mem_time + mem_ref

        rows.append([
            photon_type.name,
            n,
            f"{mem_sum:,.1f}",
            f"{mem_pos:,.1f}",
            f"{mem_dir:,.1f}",
            f"{mem_eng:,.1f}",
            f"{mem_time:,.1f}",
            f"{mem_ref:,.1f}",
        ])

        total_photons += n
        total_mem += mem_sum

    # totals row
    rows.append([
        "TOTAL",
        total_photons,
        f"{total_mem:,.1f}",
        "", "", "", "", ""
    ])

    headers = [
        "Type",
        "Photons",
        "Total KiB",
        "Pos KiB",
        "Dir KiB",
        "Energy KiB",
        "Time KiB",
        "Ref KiB",
    ]

    print(tabulate(rows, headers=headers, tablefmt=tablefmt))


def print_photon_summary(photon_maps):
    n = sum(pm.positions.shape[0] for pm in photon_maps.values())
    mem = sum(
        pm.positions.nbytes +
        pm.directions.nbytes +
        pm.energies.nbytes +
        pm.times.nbytes +
        pm.already_reflected.nbytes
        for pm in photon_maps.values()
    ) / 1024

    print(f"{n} total photons across {len(photon_maps)} maps, {mem:.2f} KiB")