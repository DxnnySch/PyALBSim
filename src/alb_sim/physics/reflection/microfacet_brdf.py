import numpy as np

from alb_sim.math.vector_math import (
    dot_batch_batch,
    dot_batch_single,
    normalize_batch,
    normalize_vector,
)
from alb_sim.physics.constants import EPSILON
from alb_sim.physics.reflection.fresnel import fresnel_schlick
from alb_sim.utils.types import Array, Vector3, Vector3Array


# https://boksajak.github.io/files/CrashCourseBRDF.pdf - Chapter 4
# TODO: Numba
def ggx_distribution(n_dot_h: Vector3Array, alpha: float) -> Array:
    """
    GGX / Trowbridge-Reitz normal distribution function.

    Parameters
    ----------
    n_dot_h : Cosine of half-vector between wi and wo (unit vector)
    alpha : Surface roughness parameter (alpha = roughness^2 typically)

    Returns
    -------
    D : Microfacet distribution value
    """

    alpha2 = alpha * alpha
    denom = (alpha2 - 1.0) * n_dot_h * n_dot_h + 1.0
    denom = np.pi * denom * denom

    return alpha2 / (denom + EPSILON)


def smith_ggx_G1(n_dot_v: Array | float, alpha: float):
    """
    Smith G1 term for GGX distribution.

    Parameters
    ----------
    n_dot_v : Cosine between normal and direction (wi or wo)
    alpha : Surface roughness parameter

    Returns
    -------
    G1 : Masking or shadowing term
    """
    # substitute lambda_ggx function into G1, substitute a into G1
    # alpha2 = alpha * alpha
    # n_dot_v2 = n_dot_v * n_dot_v
    # nom = alpha2 * (1 - n_dot_v2) + (n_dot_v2)
    # denom = n_dot_v2
    # return 2 / (1 + np.sqrt(nom / denom))

    a2 = alpha * alpha
    denom = n_dot_v + np.sqrt(a2 + (1 - a2) * n_dot_v**2)
    return 2.0 * n_dot_v / denom


def smith_ggx_G(n_dot_wi: Array, n_dot_wo: float, alpha: float):
    """
    Height-correlated Smith geometry term.

    Parameters
    ----------
    n_dot_wi : Cosines of incident directions (pointing away from surface) and normal
    n_dot_wo : Cosine of outgoing direction (pointing away from surface) and normal
    alpha : Surface roughness parameter

    Returns
    -------
    G : Geometry attenuation term
    """

    G1_i = smith_ggx_G1(n_dot_wi, alpha)
    G1_o = smith_ggx_G1(n_dot_wo, alpha)

    # return 1 / (1 + G1_i + G1_o)
    return G1_i * G1_o


def microfacet_reflected_energy(
    incident_direction: Vector3Array,
    outgoing_direction: Vector3,
    normal: Vector3,
    roughness: float,
    base_reflectance: float,
) -> Array:
    """
    Compute reflected specular energy using Cook-Torrance microfacet BRDF.

    Parameters
    ----------
    incident_direction : Incident direction (pointing away from reflection point)
    outgoing_direction : Outgoing direction (pointing away from reflection point)
    normal : Surface normal
    roughness : Surface roughness in [0, 1]
    base_reflectance : Reflectance at normal incidence

    Returns
    -------
    energy : Reflected specular energy / BRDF value
    """
    incident_direction = normalize_batch(incident_direction)
    outgoing_direction = normalize_vector(outgoing_direction)
    normal = normalize_vector(normal)

    # Half vector
    half_vector = normalize_batch(incident_direction + outgoing_direction)

    alpha = roughness * roughness

    # Cosines
    n_dot_wi: Array = np.clip(dot_batch_single(incident_direction, normal), 0.0, 1.0)
    n_dot_wo: float = np.clip(np.dot(normal, outgoing_direction), 0.0, 1.0)
    wi_dot_h: Array = np.clip(
        dot_batch_batch(incident_direction, half_vector), 0.0, 1.0
    )
    n_dot_h = np.clip(dot_batch_single(half_vector, normal), 0.0, 1.0)

    valid = (n_dot_wi > 0) & (n_dot_wo > 0)

    # Terms
    D = ggx_distribution(n_dot_h, alpha)
    G = smith_ggx_G(n_dot_wi, n_dot_wo, alpha)
    F = fresnel_schlick(wi_dot_h, base_reflectance)

    # Cook-Torrance BRDF
    denom = 4.0 * n_dot_wi * n_dot_wo + EPSILON
    spec = (D * G * F) / denom
    out = np.zeros(len(incident_direction))
    out[valid] = spec[valid]
    return out


def microfacet_transmitted_energy(
    incident_direction: Vector3Array,
    outgoing_direction: Vector3Array,
    normal: Vector3,
    roughness: float,
    eta_i: float,
    eta_o: float,
    base_reflectance: float,
) -> Array:
    """
    Microfacet BTDF (GGX), Walter et al. style
    """

    wi = normalize_batch(-incident_direction)
    wo = normalize_vector(outgoing_direction)
    n = normalize_vector(normal)

    eta = eta_i / eta_o
    alpha = roughness * roughness

    # --- Half vector (DIFFERENT) ---
    h = normalize_batch(wi + eta * wo)

    # --- Cosines ---
    n_dot_wi = np.clip(dot_batch_single(wi, n), -1.0, 1.0)
    n_dot_wo = np.clip(np.dot(wo, n), -1.0, 1.0)
    n_dot_h = np.clip(dot_batch_single(h, n), 0.0, 1.0)

    wi_dot_h = np.clip(dot_batch_batch(wi, h), 0.0, 1.0)
    wo_dot_h = np.clip(dot_batch_single(h, wo), 0.0, 1.0)

    # Must be on opposite sides
    valid = (n_dot_wi * n_dot_wo) < 0.0

    # --- Microfacet terms (REUSED) ---
    D = ggx_distribution(n_dot_h, alpha)
    G = smith_ggx_G(abs(n_dot_wi), abs(n_dot_wo), alpha)

    # --- Fresnel ---
    F = fresnel_schlick(wi_dot_h, base_reflectance)
    Ft = 1.0 - F

    # --- Jacobian term ---
    denom = eta_i * wi_dot_h + eta_o * wo_dot_h
    factor = (eta_o * eta_o * wo_dot_h) / (denom * denom + EPSILON)

    btdf = Ft * D * G * factor / (abs(n_dot_wi) * abs(n_dot_wo) + EPSILON)

    return np.where(valid, btdf, 0.0)
