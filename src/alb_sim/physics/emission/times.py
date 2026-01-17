import numpy as np

from alb_sim.utils.types import Array


def sample_gaussian_pulse_batch(
    num_samples: int, pulse_width_fwhm_s: float, rng=np.random.Generator
) -> Array:
    sigma = pulse_width_fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    return rng.normal(loc=0.0, scale=sigma, size=num_samples).astype(np.float32)


def time_offset_to_steps(delta_t_s: Array, effective_sample_rate: int) -> Array:
    return (delta_t_s / effective_sample_rate).astype(np.float32)
