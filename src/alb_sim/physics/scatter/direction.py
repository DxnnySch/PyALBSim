import numpy as np

from alb_sim.math.rotation_matrix import rotation_from_z_batch
from alb_sim.utils.types import Array, Vector3Array


def sample_scattering_directions_batch(
    vsf_theta: Array,
    vsf_cdf: Array,
    incoming_dirs: Vector3Array,
    rng: np.random.Generator,
) -> Vector3Array:
    """
    Vectorized batch sampling of scattering directions for arbitrary incoming directions.

    Parameters
    ----------
    vsf_theta : Array
        Scattering angle grid (same length as ``vsf_cdf``).
    vsf_cdf : Array
        Cumulative distribution function of the volume scattering function.
    incoming_dirs : Vector3Array
        Unit vectors representing incoming photon directions.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    Vector3Array
        Scattered unit direction vectors of shape (N, 3).
    """
    num_samples = incoming_dirs.shape[0]

    # Sample scattering angles
    random_vals = rng.random(num_samples)
    scatter_theta = np.interp(random_vals, vsf_cdf, vsf_theta)
    scatter_phi = rng.uniform(0, 2 * np.pi, size=num_samples)

    # Local frame scattering vectors (z-aligned frame)
    x_local = np.sin(scatter_theta) * np.cos(scatter_phi)
    y_local = np.sin(scatter_theta) * np.sin(scatter_phi)
    z_local = np.cos(scatter_theta)
    local_dirs = np.stack((x_local, y_local, z_local), axis=1)

    # Build rotation matrices per photon
    # rotation_matrices = orthonormal_from_z_batch(incoming_dirs)
    rotation_matrices = rotation_from_z_batch(incoming_dirs)

    # Apply rotations: batched matrix-vector multiplication
    directions = np.einsum("nij,nj->ni", rotation_matrices, local_dirs)
    return directions
