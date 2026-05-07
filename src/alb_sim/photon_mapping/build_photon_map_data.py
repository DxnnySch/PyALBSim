import numpy as np

from alb_sim.photon_mapping.photon_map_data import PhotonMapData
from alb_sim.photon_mapping.photon_storage import PhotonStorage
from alb_sim.photon_mapping.photon_type import PhotonType


def build_photon_map_data(
    photon_storage: PhotonStorage,
) -> dict[PhotonType, PhotonMapData]:
    """
    Convert a PhotonStorage into dense PhotonMapData arrays per photon type.

    Parameters
    ----------
    photon_storage : PhotonStorage
        In-memory photon interaction storage.

    Returns
    -------
    dict[PhotonType, PhotonMapData]
        Mapping from photon type to concatenated photon map data.
    """
    maps = {}

    for photon_type in photon_storage.positions:
        num_photons = sum(len(p) for p in photon_storage.positions[photon_type])

        # Handle first_water_interaction
        fwi_list = photon_storage.first_water_interaction.get(photon_type, [])
        if fwi_list:
            first_water_interaction = np.concatenate(fwi_list, axis=0)
        else:
            first_water_interaction = np.full(
                (num_photons, 3), np.nan, dtype=np.float32
            )

        # Handle seafloor_interaction
        sfi_list = photon_storage.seafloor_interaction.get(photon_type, [])
        if sfi_list:
            seafloor_interaction = np.concatenate(sfi_list, axis=0)
        else:
            seafloor_interaction = np.full((num_photons, 3), np.nan, dtype=np.float32)

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
            first_water_interaction=first_water_interaction.astype(np.float32),
            seafloor_interaction=seafloor_interaction.astype(np.float32),
        )

    return maps
