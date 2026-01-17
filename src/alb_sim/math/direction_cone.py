import numpy as np

from alb_sim.math.rotation_matrix import rotation_from_z_batch
from alb_sim.utils.types import Vector3, Vector3Array


def sample_directions_in_cone(
    center_dir: Vector3, cone_angle: float, num_samples: int, rng: np.random.Generator
) -> Vector3Array:
    """
    Generate num_samples unit direction vectors within a cone defined by center_dir and cone_angle (in radians).
    """
    # center_dir must be normalized

    # https://math.stackexchange.com/a/205589
    # Create random directions in local cone coordinates
    z = rng.uniform(np.cos(cone_angle), 1, size=num_samples)
    sin_theta = np.sqrt(1 - z**2)
    phi = rng.uniform(0, 2 * np.pi, size=num_samples)

    # Local direction vectors (cone along +Z)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    local_dirs = np.stack([x, y, z], axis=1)  # shape (N, 3)

    # Build rotation matrix from [0, 0, 1] to laser_dir
    rotation_matrix = rotation_from_z_batch(
        center_dir.astype(np.float32)[np.newaxis, :]
    )[0]
    directions = (
        local_dirs @ rotation_matrix.T
    )  # Rotate each local direction to align with laser_dir

    return directions.astype(np.float32)
