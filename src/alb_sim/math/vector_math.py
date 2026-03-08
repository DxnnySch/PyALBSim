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
    return np.einsum("ij,j->i", v_batch, v)


def dot_batch_batch(a_batch: Vector3Array, b_batch: Vector3Array) -> Array:
    return np.einsum("ij,ij->i", a_batch, b_batch)


def random_unit_vector_batch(
    num_samples: int, rng: np.random.Generator
) -> Vector3Array:
    v = rng.normal(size=(num_samples, 3)).astype(np.float32)
    return normalize_batch(v).astype(np.float32)
