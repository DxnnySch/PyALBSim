from __future__ import annotations  # will be unneeded by python 3.14

from collections import defaultdict

from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.utils.types import Array, Vector3Array


class PhotonStorage:
    def __init__(self):
        self.positions: defaultdict[PhotonType, list[Vector3Array]] = defaultdict(list)
        self.directions: defaultdict[PhotonType, list[Vector3Array]] = defaultdict(list)
        self.energies: defaultdict[PhotonType, list[Array]] = defaultdict(list)
        self.times: defaultdict[PhotonType, list[Array]] = defaultdict(list)
        self.first_water_interaction: defaultdict[PhotonType, list[Vector3Array]] = (
            defaultdict(list)
        )
        self.seafloor_interaction: defaultdict[PhotonType, list[Vector3Array]] = (
            defaultdict(list)
        )

    def add(
        self,
        photon_type,
        positions,
        directions,
        energies,
        times,
        first_water_interaction,
        seafloor_interaction,
    ):
        self.positions[photon_type].append(positions)
        self.directions[photon_type].append(directions)
        self.energies[photon_type].append(energies)
        self.times[photon_type].append(times)
        self.first_water_interaction[photon_type].append(first_water_interaction)
        self.seafloor_interaction[photon_type].append(seafloor_interaction)

    def merge(self, other: PhotonStorage):
        """Merge another PhotonStorage into this one."""
        for photon_type in other.positions:
            self.positions[photon_type].extend(other.positions[photon_type])
            self.directions[photon_type].extend(other.directions[photon_type])
            self.energies[photon_type].extend(other.energies[photon_type])
            self.times[photon_type].extend(other.times[photon_type])
            self.first_water_interaction[photon_type].extend(
                other.first_water_interaction[photon_type]
            )
            self.seafloor_interaction[photon_type].extend(
                other.seafloor_interaction[photon_type]
            )
