from __future__ import annotations
from collections import defaultdict

from enum import Enum
from typing import DefaultDict, List
import numpy as np
from numpy.typing import NDArray

# TODO: temp, add import later in correct place
PhotonType = Enum("PhotonType", [("BOTTOM_REFLECTION", 0), ("SCATTER", 1), ("SURFACE_REFLECTION", 2)])
class PhotonStorage:
    def __init__(self):
        self.positions: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.directions: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.energies: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.times: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.already_reflected: DefaultDict[PhotonType, List[NDArray[np.bool_]]]  = defaultdict(list)

    def add(
        self,
        photon_type,
        positions,
        directions,
        energies,
        times,
        already_reflected,
    ):
        self.positions[photon_type].append(positions)
        self.directions[photon_type].append(directions)
        self.energies[photon_type].append(energies)
        self.times[photon_type].append(times)
        self.already_reflected[photon_type].append(already_reflected)

    def merge(self, other: PhotonStorage):
        """Merge another PhotonStorage into this one."""
        for photon_type in other.positions.keys():
            self.positions[photon_type].extend(other.positions[photon_type])
            self.directions[photon_type].extend(other.directions[photon_type])
            self.energies[photon_type].extend(other.energies[photon_type])
            self.times[photon_type].extend(other.times[photon_type])
            self.already_reflected[photon_type].extend(
                other.already_reflected[photon_type]
            )
