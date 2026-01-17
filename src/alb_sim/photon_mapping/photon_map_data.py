from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PhotonMapData:
    positions: NDArray[np.float32]  # (N, 3)
    directions: NDArray[np.float32]  # (N, 3)
    energies: NDArray[np.float32]  # (N,)
    times: NDArray[np.float32]  # (N,)
