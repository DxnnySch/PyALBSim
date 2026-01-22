from enum import Enum, auto


class PhotonType(Enum):
    BOTTOM_REFLECTION = auto()
    SCATTER = auto()
    SURFACE_REFLECTION = auto()
    SURFACE_TRANSMISSION_UP = auto()
    # SURFACE_TRANSMISSION_DOWN = auto()
