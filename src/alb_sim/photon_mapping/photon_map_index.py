from scipy.spatial import KDTree

from alb_sim.photon_mapping.photon_map_data import PhotonMapData


class PhotonMapIndex:
    def __init__(self, data: PhotonMapData):
        self.data = data
        self.tree = KDTree(data.positions)
