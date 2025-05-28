import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List
from numpy.typing import NDArray

def visualize_photon_paths(
    histories: List[List[NDArray[np.float32]]],
    n: int = 10,
    water_surface_y: float = 5.0,
    seafloor_y: float = 0.0,
    figsize=(10, 6),
):
    """
    Visualize 2D (x, y) paths of a sample of photons.
    """
    plt.figure(figsize=figsize)

    # Plot water surface and seafloor
    plt.axhline(y=water_surface_y, color='blue', linestyle='--', label='Water Surface')
    plt.axhline(y=seafloor_y, color='brown', linestyle='--', label='Seafloor')

    # Draw laser origin
    plt.plot(0, 0, 'ro', label='Laser Origin')

    # Sample photon histories
    sample_histories = random.sample(histories, min(n, len(histories)))

    for i, path in enumerate(sample_histories):
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        plt.plot(x, y, '-o', label=f'Photon {i + 1}', alpha=0.6, markersize=2)

    plt.xlabel('X position')
    plt.ylabel('Y position (depth)')
    plt.title(f'Sample of {n} Photon Paths (2D)')
    # plt.gca().invert_yaxis()  # So depth increases downward
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

