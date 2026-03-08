import numpy as np

from alb_sim.photon_mapping.photon_map_data import PhotonMapData
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.photon_type import PhotonType


def build_photon_map_data(
    photon_storage: PhotonStorage,
) -> dict[PhotonType, PhotonMapData]:
    maps = {}

    for photon_type in photon_storage.positions:
        maps[photon_type] = PhotonMapData(
            positions=np.concatenate(
                photon_storage.positions[photon_type], axis=0
            ).astype(np.float32),
            directions=np.concatenate(
                photon_storage.directions[photon_type], axis=0
            ).astype(np.float32),
            energies=np.concatenate(
                photon_storage.energies[photon_type], axis=0
            ).astype(np.float32),
            times=np.concatenate(photon_storage.times[photon_type], axis=0).astype(
                np.float32
            ),
        )

    return maps
