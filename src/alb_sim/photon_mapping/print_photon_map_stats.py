from collections.abc import Mapping

from tabulate import tabulate

from alb_sim.photon_mapping.photon_map_data import PhotonMapData
from alb_sim.photon_mapping.photon_type import PhotonType


def photon_map_stats(
    photon_maps: Mapping[PhotonType, PhotonMapData], tablefmt: str = "github"
) -> str:
    """
    Summarise photon map memory usage per photon type as a formatted table.

    Parameters
    ----------
    photon_maps : Mapping[PhotonType, PhotonMapData]
        Photon maps to summarise.
    tablefmt : str, optional
        Tabulate table format (e.g. ``"github"``, ``"plain"``).

    Returns
    -------
    str
        Formatted table of photon counts and memory usage.
    """
    rows = []

    total_photons = 0
    total_mem = 0.0

    for photon_type, data in photon_maps.items():
        n = data.positions.shape[0]

        mem_pos = data.positions.nbytes / 1024
        mem_dir = data.directions.nbytes / 1024
        mem_eng = data.energies.nbytes / 1024
        mem_time = data.times.nbytes / 1024

        mem_sum = mem_pos + mem_dir + mem_eng + mem_time

        rows.append(
            [
                photon_type.name,
                n,
                f"{mem_sum:,.1f}",
                f"{mem_pos:,.1f}",
                f"{mem_dir:,.1f}",
                f"{mem_eng:,.1f}",
                f"{mem_time:,.1f}",
            ]
        )

        total_photons += n
        total_mem += mem_sum

    # totals row
    rows.append(["TOTAL", total_photons, f"{total_mem:,.1f}", "", "", "", ""])

    headers = [
        "Type",
        "Photons",
        "Total KiB",
        "Pos KiB",
        "Dir KiB",
        "Energy KiB",
        "Time KiB",
    ]

    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def photon_summary(photon_maps) -> str:
    """
    Return a one-line summary of total photons and memory usage.

    Parameters
    ----------
    photon_maps : Mapping[PhotonType, PhotonMapData]
        Photon maps to summarise.

    Returns
    -------
    str
        Human-readable summary string.
    """
    n = sum(pm.positions.shape[0] for pm in photon_maps.values())
    mem = (
        sum(
            pm.positions.nbytes
            + pm.directions.nbytes
            + pm.energies.nbytes
            + pm.times.nbytes
            for pm in photon_maps.values()
        )
        / 1024
    )

    return f"{n} total photons across {len(photon_maps)} maps, {mem:.2f} KiB"
