from enum import Enum, auto

class PhotonState(Enum):
    IN_AIR = auto()
    ENTERING_WATER = auto()
    IN_WATER = auto()
    HITTING_SEAFLOOR = auto()
    EXITING_WATER = auto()
    DONE = auto()