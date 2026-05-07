import numpy as np

from alb_sim.utils.types import Array, Vector3, Vector3Array


def normalize_vector(v: Vector3) -> Vector3:
    """Normalize a single 3D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize_batch(v_batch: Vector3Array) -> Vector3Array:
    """Normalize a batch of 3D vectors (shape: (N, 3))."""
    # TODO print(v_batch)
    norms = np.linalg.norm(v_batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v_batch / norms


def length_batch(v_batch: Vector3Array) -> Array:
    """Calculate length of a batch of 3D vectors (shape: (N, 3))"""
    return np.linalg.norm(v_batch, axis=1)


def dot_batch_single(v_batch: Vector3Array, v: Vector3) -> Array:
    """
    Compute dot products between a batch of vectors and a single vector.

    Parameters
    ----------
    v_batch : ndarray of shape (N, 3)
        Batch of input vectors.
    v : ndarray of shape (3,)
        Single vector to dot with each row of ``v_batch``.

    Returns
    -------
    Array
        Dot products for each row in ``v_batch``.
    """
    return np.einsum("ij,j->i", v_batch, v)


def dot_batch_batch(a_batch: Vector3Array, b_batch: Vector3Array) -> Array:
    """
    Compute elementwise dot products for two batches of vectors.

    Parameters
    ----------
    a_batch : ndarray of shape (N, 3)
        First batch of vectors.
    b_batch : ndarray of shape (N, 3)
        Second batch of vectors.

    Returns
    -------
    Array
        Dot products for each pair of rows.
    """
    return np.einsum("ij,ij->i", a_batch, b_batch)


def random_unit_vector_batch(
    num_samples: int, rng: np.random.Generator
) -> Vector3Array:
    """
    Sample random unit vectors with an isotropic distribution.

    Parameters
    ----------
    num_samples : int
        Number of unit vectors to generate.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    Vector3Array
        Array of shape (num_samples, 3) containing unit vectors.
    """
    v = rng.normal(size=(num_samples, 3)).astype(np.float32)
    return normalize_batch(v).astype(np.float32)
