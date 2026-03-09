from dataclasses import dataclass

from alb_sim.utils.types import Array, Vector3Array


@dataclass
class PhotonMapData:
    """Photon interaction arrays used to build a photon map index."""
    positions: Vector3Array  # (N, 3)
    directions: Vector3Array  # (N, 3)
    energies: Array  # (N,)
    times: Array  # (N,)
    first_water_interaction: Vector3Array  # (N, 3)
    seafloor_interaction: Vector3Array  # (N, 3)
