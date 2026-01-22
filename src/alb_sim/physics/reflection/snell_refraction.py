import numpy as np

from alb_sim.math.vector_math import dot_batch_single, normalize_batch
from alb_sim.utils.types import BoolArray, Vector3, Vector3Array


def calculate_refraction_direction(
    incoming_directions: Vector3Array,
    normal: Vector3,
    refractive_index_1: float,
    refractive_index_2: float,
) -> tuple[Vector3Array, BoolArray]:
    eta = refractive_index_1 / refractive_index_2

    cos_i = np.clip(dot_batch_single(-incoming_directions, normal), -1.0, 1.0)

    sin2_t = eta**2 * (1.0 - cos_i**2)

    # total internal reflection mask
    tir = sin2_t > 1.0

    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))

    t_dirs = eta * incoming_directions + (eta * cos_i - cos_t)[:, np.newaxis] * normal

    return normalize_batch(t_dirs), tir
