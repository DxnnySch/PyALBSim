import numpy as np

from alb_sim.math.vector_math import normalize_batch, normalize_vector
from alb_sim.physics.constants import EPSILON
from alb_sim.utils.types import Array, Vector3, Vector3Array


def calculate_energy_batch(
    photon_directions: Vector3Array,
    sensor_photon_direction: Vector3,
    vsf_theta: Array,
    vsf_cdf: Array,
) -> Array:
    """
    photon_direction: (N, 3) photon directions
    sensor_photon_direction: (3,) sensor photon direction
    vsf_theta: (N,) angles of scattering function
    vsf_cdf: (N,) cumulative distribution function of scattering function
    returns: (N,) energies
    """
    # Normalize - needed?
    photon_directions = normalize_batch(photon_directions)
    sensor_photon_direction = normalize_vector(sensor_photon_direction)

    # dot product
    cos_theta = np.einsum("ij,j->i", photon_directions, sensor_photon_direction)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    scatter_angle = np.arccos(cos_theta)

    delta = (np.pi - EPSILON) / len(vsf_theta)
    i = ((scatter_angle - EPSILON) / delta).astype(int)
    i = np.clip(i, 0, len(vsf_theta) - 1)

    theta_lower = i * delta
    theta_upper = (i + 1) * delta

    cdf_lower = np.interp(theta_lower, vsf_theta, vsf_cdf)
    cdf_upper = np.interp(theta_upper, vsf_theta, vsf_cdf)

    return cdf_upper - cdf_lower
