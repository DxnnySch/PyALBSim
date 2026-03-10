from enum import Enum, auto


class PhotonType(Enum):
    """Categories of photon interactions stored and reconstructed in the simulation."""

    BOTTOM_REFLECTION = auto()
    SCATTER = auto()
    SURFACE_REFLECTION = auto()
    SURFACE_TRANSMISSION_UP = auto()
