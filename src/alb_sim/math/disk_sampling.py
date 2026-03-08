import numpy as np

from alb_sim.math.rotation_matrix import rotation_from_z_batch
from alb_sim.utils.types import Vector3, Vector3Array


def sample_disk_points(
    normal_direction: Vector3,
    radius: float,
    origin: Vector3,
    num_samples: int,
    rng: np.random.Generator,
) -> Vector3Array:
    """
    Generate num_samples positions on a disk, defined by center point at origin, radius and normal_direction
    """
    xi1 = rng.random(num_samples)
    xi2 = rng.random(num_samples)

    r = radius * np.sqrt(xi1)
    theta = 2 * np.pi * xi2

    local_points = np.stack(
        [r * np.cos(theta), r * np.sin(theta), np.zeros_like(r)], axis=1
    )

    # Build rotation matrix from [0, 0, 1] to laser_dir
    rotation_matrix = rotation_from_z_batch(
        normal_direction.astype(np.float64)[np.newaxis, :]
    )[0]

    disk_points = local_points @ rotation_matrix.T
    disk_points += origin

    return disk_points.astype(np.float64)
