import numpy as np

from alb_sim.math.vector_math import normalize_batch, random_unit_vector_batch
from alb_sim.utils.types import Vector3, Vector3Array


def heuristic_sample_batch(
    num_samples: int, normal: Vector3, rng: np.random.Generator
) -> Vector3Array:
    return normalize_batch(normal + random_unit_vector_batch(num_samples, rng))
