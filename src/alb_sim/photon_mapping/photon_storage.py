from __future__ import annotations  # will be unneeded by python 3.14

from collections import defaultdict

from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.utils.types import Array, Vector3Array


class PhotonStorage:
    def __init__(self):
        """In-memory container for photon interaction arrays grouped by type."""
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
        """
        Append a batch of photon interactions to storage.

        Parameters
        ----------
        photon_type : PhotonType
            Category of photon interaction being stored.
        positions : Vector3Array
            Photon positions for this batch.
        directions : Vector3Array
            Photon directions for this batch.
        energies : Array
            Photon energies for this batch.
        times : Array
            Photon time indices for this batch.
        first_water_interaction : Vector3Array
            First water interaction points for this batch.
        seafloor_interaction : Vector3Array
            Seafloor interaction points for this batch.
        """
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
