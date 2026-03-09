import numpy as np

from alb_sim.math.rotation_matrix import rotation_from_z_batch
from alb_sim.utils.types import Vector3, Vector3Array


def sample_directions_in_cone_uniform(
    center_dir: Vector3, cone_angle: float, num_samples: int, rng: np.random.Generator
) -> Vector3Array:
    """
    Sample unit direction vectors uniformly within a cone.

    Parameters
    ----------
    center_dir : array_like, shape (3,)
        Normalised cone axis direction.
    cone_angle : float
        Cone half-angle in radians.
    num_samples : int
        Number of direction samples to generate.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    Vector3Array
        Array of shape (num_samples, 3) with unit direction vectors.
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
        center_dir.astype(np.float64)[np.newaxis, :]
    )[0]
    directions = (
        local_dirs @ rotation_matrix.T
    )  # Rotate each local direction to align with laser_dir

    return directions.astype(np.float32)


def sample_directions_in_cone_gaussian(
    center_dir: Vector3, cone_angle: float, num_samples: int, rng: np.random.Generator
) -> Vector3Array:
    """
    Sample unit direction vectors in a cone with Gaussian-weighted angles.

    The angular distribution follows a truncated Rayleigh profile (2D Gaussian in
    transverse space) and is hard-clipped at ``cone_angle``.

    Parameters
    ----------
    center_dir : array_like, shape (3,)
        Normalised cone axis direction.
    cone_angle : float
        Maximum divergence half-angle in radians.
    num_samples : int
        Number of direction samples to generate.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    Vector3Array
        Array of shape (num_samples, 3) with unit direction vectors.
    """
    # Use sigma such that cone_angle ≈ 3σ (99.7% would naturally fall within)
    # This gives a nice Gaussian shape while keeping the truncation minimal
    sigma = cone_angle / 3.0

    # Sample polar angle (theta) from truncated Rayleigh distribution
    # Use rejection sampling to ensure all samples are within cone_angle
    theta = np.empty(num_samples)
    remaining = num_samples
    idx = 0

    while remaining > 0:
        # Oversample to reduce iterations
        samples = rng.rayleigh(sigma, size=int(remaining * 1.5) + 100)
        valid = samples[samples <= cone_angle]
        take = min(len(valid), remaining)
        theta[idx : idx + take] = valid[:take]
        idx += take
        remaining = num_samples - idx

    # Sample azimuthal angle uniformly
    phi = rng.uniform(0, 2 * np.pi, size=num_samples)

    # Convert to Cartesian coordinates (local cone frame with +Z as axis)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    local_dirs = np.stack([x, y, z], axis=1)  # shape (N, 3)

    # Rotate from local frame (+Z axis) to align with center_dir
    rotation_matrix = rotation_from_z_batch(
        center_dir.astype(np.float64)[np.newaxis, :]
    )[0]
    directions = (
        local_dirs @ rotation_matrix.T
    )  # Rotate each local direction to align with laser_dir

    return directions.astype(np.float32)
