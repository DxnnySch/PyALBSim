from typing import overload

from alb_sim.utils.types import Array


@overload
def fresnel_schlick(cos_theta: Array, base_reflectance: float) -> Array: ...


@overload
def fresnel_schlick(cos_theta: float, base_reflectance: float) -> float: ...


def fresnel_schlick(cos_theta: Array | float, base_reflectance: float) -> Array | float:
    """
    Schlick Fresnel approximation.

    Parameters
    ----------
    cos_theta : Cosines between half-vector and incident direction
    base_reflectance : Reflectance at normal incidence

    Returns
    -------
    F : Fresnel reflectance
    """
    return base_reflectance + (1.0 - base_reflectance) * (1.0 - cos_theta) ** 5
