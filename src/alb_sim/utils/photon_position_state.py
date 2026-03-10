from enum import Enum, auto


class PhotonPositionState(Enum):
    """Discrete states describing where a photon is relative to the water column."""

    IN_AIR = auto()
    ENTERING_WATER = auto()
    IN_WATER = auto()
    HITTING_SEAFLOOR = auto()
    EXITING_WATER = auto()
    DONE = auto()
