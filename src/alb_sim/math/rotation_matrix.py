import numpy as np

from alb_sim.math.vector_math import normalize_batch
from alb_sim.utils.types import Matrix3Array, Vector3Array


def orthonormal_from_z_batch(v):
    """
    Vectorized construction of rotation matrices to align local +z with incoming_dirs.

    Parameters
    ----------
    v : ndarray of shape (N, 3)
        Target direction vectors.

    Returns
    -------
    Matrix3Array
        Rotation matrices that align +z with each vector in ``v``.
    """
    w = v / np.linalg.norm(v, axis=1, keepdims=True)  # (N, 3)

    # Choose "up" vector adaptively to avoid numerical instability
    use_z_up = np.abs(w[:, 2]) < 0.99
    up = np.zeros_like(w)
    up[use_z_up] = np.array([0, 0, 1])
    up[~use_z_up] = np.array([1, 0, 0])

    u = np.cross(up, w)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    v = np.cross(w, u)

    # Stack u, v, w as rotation matrix columns
    rotation_matrices = np.stack((u, v, w), axis=2)  # shape (N, 3, 3)
    return rotation_matrices


def rotation_from_z_batch(v: Vector3Array) -> Matrix3Array:
    """
    Build rotation matrices that rotate +z to each direction in v.

    Parameters
    ----------
    v : ndarray, shape (N, 3)
        Target direction vectors (need not be normalized)

    Returns
    -------
    Matrix3Array
        Rotation matrices of shape (N, 3, 3).
    """
    v = np.asarray(v, dtype=float)
    num_samples = v.shape[0]

    # Normalize directions
    v = normalize_batch(v)

    # Dot and cross products
    c = v[:, 2]  # dot(z, v)
    vx = -v[:, 1]  # cross(z, v)
    vy = v[:, 0]
    vz = np.zeros_like(vx)

    s2 = vx * vx + vy * vy  # |cross|^2

    rotation_matrices = np.zeros((num_samples, 3, 3))

    # --- Case 1: already aligned (+z)
    aligned = np.isclose(c, 1.0)
    rotation_matrices[aligned] = np.eye(3)

    # --- Case 2: opposite direction (-z)
    opposite = np.isclose(c, -1.0)
    rotation_matrices[opposite] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # --- Case 3: general Rodrigues rotation
    general = ~(aligned | opposite)
    if np.any(general):
        vxg, vyg = vx[general], vy[general]
        cg = c[general]
        s2g = s2[general]

        # Skew matrices [v]_x
        skew_matrices = np.zeros((np.count_nonzero(general), 3, 3))
        skew_matrices[:, 0, 1] = -vz[general]
        skew_matrices[:, 0, 2] = vyg
        skew_matrices[:, 1, 0] = vz[general]
        skew_matrices[:, 1, 2] = -vxg
        skew_matrices[:, 2, 0] = -vyg
        skew_matrices[:, 2, 1] = vxg

        # Rodrigues
        identity_matrix = np.eye(3)
        rotation_matrices[general] = (
            identity_matrix
            + skew_matrices
            + np.einsum("nij,njk->nik", skew_matrices, skew_matrices)
            * ((1 - cg) / s2g)[:, None, None]
        )

    return rotation_matrices.astype(np.float32)
