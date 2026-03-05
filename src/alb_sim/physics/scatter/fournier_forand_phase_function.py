import numpy as np

from alb_sim.utils.types import Array

SCATTER_DIVISIONS = 18000


# https://www.oceanopticsbook.info/view/scattering/the-fournier-forand-phase-function
def calculate_phase_function(
    junge_slope: float, refractive_index_ratio: float
) -> tuple[Array, Array, Array]:
    theta = np.linspace(1e-6, np.pi, SCATTER_DIVISIONS, dtype=np.float64)

    v = (3 - junge_slope) / 2
    delta = (4 / (3 * (refractive_index_ratio - 1) ** 2)) * np.sin(theta / 2) ** 2
    delta_180 = (4 / (3 * (refractive_index_ratio - 1) ** 2)) * np.sin(np.pi / 2) ** 2

    term1 = 1 / (4 * np.pi * (1 - delta) ** 2 * delta**v)
    term2 = (
        v * (1 - delta)
        - (1 - delta**v)
        + (delta * (1 - delta**v) - v * (1 - delta)) * np.sin(theta / 2) ** (-2)
    )
    term3 = ((1 - delta_180**v) / (16 * np.pi * (delta_180 - 1) * delta_180**v)) * (
        3 * np.cos(theta) ** 2 - 1
    )
    ff = term1 * term2 + term3

    # calculate normalized cumulative distribution function
    cdf = np.cumsum(ff * np.sin(theta))
    cdf /= cdf[-1]

    return theta, ff, cdf


def calculate_phase_function_matlab(
    junge_slope: float, refractive_index_ratio: float
) -> tuple[Array, Array, Array]:
    theta = np.linspace(1e-6, np.pi, SCATTER_DIVISIONS, dtype=np.float64)

    v = (3 - junge_slope) / 2
    delta = (4 / (3 * (refractive_index_ratio - 1) ** 2)) * np.sin(theta / 2) ** 2
    delta_180 = (4 / (3 * (refractive_index_ratio - 1) ** 2)) * np.sin(np.pi / 2) ** 2

    term1 = 1 / ((1 - delta) * delta**v)
    term2 = (1 - delta ** (v + 1)) - (1 - delta**v) * np.sin(theta / 2) ** (2)
    term3 = ((1 / 8) * (1 - delta_180**v)) / ((delta_180 - 1) * delta_180**v)
    term4 = np.cos(theta) * np.sin(theta) ** 2
    ff = term1 * term2 + term3 * term4

    # calculate normalized cumulative distribution function
    cdf = np.cumsum(ff * np.sin(theta))
    cdf /= cdf[-1]

    return theta, ff, cdf


# https://www.oceanopticsbook.info/view/scattering/the-fournier-forand-phase-function
def calculate_backscatter_fraction(junge_slope: float, refractive_index_ratio: float):
    v = (3 - junge_slope) / 2
    delta_90 = (4 / (3 * (refractive_index_ratio - 1) ** 2)) * np.sin(np.pi / 4) ** 2

    num = 1 - delta_90 ** (v + 1) - 0.5 * (1 - delta_90**v)
    denom = (1 - delta_90) * delta_90**v

    return 1 - (num / denom)
