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

    def at(self, z: Union[Array, float]) -> Union[Array, float]:
        scalar = np.isscalar(z)
        z = np.asarray(z, dtype=np.float32)
        t = np.clip(z / self.height, 0.0, 1.0)
        out = self.start + t * (self.end - self.start)
        return out.item() if scalar else out
