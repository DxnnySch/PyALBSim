from dataclasses import dataclass
from typing import Protocol, Union, overload

import numpy as np

from alb_sim.utils.types import Array


class ScalarParameter(Protocol):
    @overload
    def at(self, t: float) -> float: ...

    @overload
    def at(self, t: Array) -> Array: ...

    def at(self, t: Union[Array, float]) -> Union[Array, float]: ...


NumberOrScalar = Union[float, ScalarParameter]


def normalize_number_or_scalar(value: NumberOrScalar) -> NumberOrScalar:
    """
    Normalise numeric configuration values to floats while preserving scalars.

    Parameters
    ----------
    value : NumberOrScalar
        Either a plain number or a scalar profile object.

    Returns
    -------
    NumberOrScalar
        Float-converted numeric value or the original scalar profile.
    """
    if isinstance(value, bool):
        # defensive: bool is a subclass of int
        raise TypeError("Boolean is not a valid numeric parameter")

    if isinstance(value, int):
        return float(value)

    return value


@dataclass(frozen=True)
class LinearParameter:
    start: float
    end: float

    def at(self, t: Union[Array, float]) -> Union[Array, float]:
        """
        Evaluate the linear profile at fractional depth ``t``.

        Parameters
        ----------
        t : float or Array
            Fraction in ``[0, 1]`` along the profile.

        Returns
        -------
        float or Array
            Interpolated value(s) at the requested location(s).
        """
        scalar = np.isscalar(t)
        t = np.asarray(t, dtype=np.float32)
        out = self.start + t * (self.end - self.start)
        return out.item() if scalar else out


@dataclass(frozen=True)
class ExponentialParameter:
    start: float
    end: float

    def __post_init__(self):
        if self.start <= 0 or self.end <= 0:
            raise ValueError("ExponentialProfile requires positive values")

    def at(self, t: Union[Array, float]) -> Union[Array, float]:
        """
        Evaluate the exponential profile at fractional depth ``t``.

        The value varies exponentially between ``start`` and ``end``.

        Parameters
        ----------
        t : float or Array
            Fraction in ``[0, 1]`` along the profile.

        Returns
        -------
        float or Array
            Interpolated value(s) at the requested location(s).
        """
        log_start = np.log(self.start)
        log_end = np.log(self.end)

        return np.exp(log_start + t * (log_end - log_start))
