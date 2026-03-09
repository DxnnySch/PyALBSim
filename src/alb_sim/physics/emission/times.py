import numpy as np

from alb_sim.utils.types import Array


def sample_gaussian_pulse_batch(
    num_samples: int, pulse_width_fwhm_s: float, rng=np.random.Generator
) -> Array:
    """
    Sample emission time offsets from a Gaussian laser pulse.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    pulse_width_fwhm_s : float
        Pulse full width at half maximum in seconds.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    Array
        Time offsets (seconds) drawn from the pulse temporal profile.
    """
    sigma = pulse_width_fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    return rng.normal(loc=0.0, scale=sigma, size=num_samples).astype(np.float32)


def time_offset_to_steps(delta_t_s: Array, effective_sample_rate: int) -> Array:
    """
    Convert time offsets in seconds to discrete simulation steps.

    Parameters
    ----------
    delta_t_s : Array
        Time offsets in seconds.
    effective_sample_rate : int
        Effective sample rate in Hz (after oversampling).

    Returns
    -------
    Array
        Time offsets expressed in simulation time steps.
    """
    return (delta_t_s / effective_sample_rate).astype(np.float32)
