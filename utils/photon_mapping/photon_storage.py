from collections import defaultdict

from enum import Enum
from typing import DefaultDict, List
import numpy as np
from numpy.typing import NDArray

# TODO: temp, add import later in correct place
PhotonType = Enum("PhotonType", [("BOTTOM_REFLECTION", 0), ("SCATTER", 1), ("SURFACE_REFLECTION", 2)])
class PhotonStorage:
    def __init__(self):
        self.positions: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.directions: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.energies: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.times: DefaultDict[PhotonType, List[NDArray[np.float32]]] = defaultdict(list)
        self.already_reflected: DefaultDict[PhotonType, List[NDArray[np.bool_]]]  = defaultdict(list)