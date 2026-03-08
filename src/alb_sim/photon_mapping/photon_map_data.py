from dataclasses import dataclass

from alb_sim.utils.types import Array, Vector3Array


@dataclass
class PhotonMapData:
    positions: Vector3Array  # (N, 3)
    directions: Vector3Array  # (N, 3)
    energies: Array  # (N,)
    times: Array  # (N,)
    first_water_interaction: Vector3Array  # (N, 3)
    seafloor_interaction: Vector3Array  # (N, 3)
