from scipy.spatial import KDTree

from alb_sim.photon_mapping.photon_map_data import PhotonMapData


class PhotonMapIndex:
    def __init__(self, data: PhotonMapData):
        """
        Spatial index over stored photon interactions.

        Parameters
        ----------
        data : PhotonMapData
            Photon interaction data used to build the KDTree.
        """
        self.data = data
        self.tree = KDTree(data.positions)
