
from enum import Enum

import numpy as np

from utils.photon_mapping.photon_map_data import PhotonMapData
from utils.photon_mapping.photon_storage import PhotonStorage

# TODO: temp, add import later in correct place
PhotonType = Enum("PhotonType", [("BOTTOM_REFLECTION", 0), ("SCATTER", 1), ("SURFACE_REFLECTION", 2)])
def build_photon_map_data(photon_storage: PhotonStorage) -> dict[PhotonType, PhotonMapData]:
    maps = {}

    for photon_type in photon_storage.positions.keys():
        maps[photon_type] = PhotonMapData(
            positions=np.concatenate(photon_storage.positions[photon_type], axis=0),
            directions=np.concatenate(photon_storage.directions[photon_type], axis=0),
            energies=np.concatenate(photon_storage.energies[photon_type], axis=0),
            times=np.concatenate(photon_storage.times[photon_type], axis=0),
            already_reflected=np.concatenate(photon_storage.already_reflected[photon_type], axis=0),
        )

    return maps