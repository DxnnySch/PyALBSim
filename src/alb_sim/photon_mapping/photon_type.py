from enum import Enum, auto


class PhotonType(Enum):
    BOTTOM_REFLECTION = auto()
    SCATTER = auto()
    SURFACE_REFLECTION = auto()
    SURFACE_TRANSMISSION = auto()
