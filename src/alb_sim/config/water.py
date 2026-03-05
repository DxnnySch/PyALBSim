from dataclasses import dataclass, field
from typing import Union

import numpy as np

from alb_sim.utils.parameter_profile import NumberOrScalar, normalize_number_or_scalar


@dataclass(frozen=True)
class FournierForandConfig:
    refractive_index_ratio: float = field(
        default=1.1,
        metadata={
            "unit": "",
            "description": "Refractive index ratio of particles relative to pure water (real part)",
        },
    )
    junge_slope: float = field(
        default=3.62,  # 3.5835,
        metadata={
            "unit": "",
            "description": "Slope of the hyperbolic (Junge) particle size distribution",
        },
    )


@dataclass(frozen=True)
class TurbidityLayerConfig:
    height: float = field(
        default=3,
        metadata={"unit": "m", "description": "Height of the turbidity layer"},
    )
    scattering_coefficient: NumberOrScalar = field(
        default=2.5,  # 1.21,
        metadata={
            "unit": "1/m",
            "description": "Total scattering coefficient",
        },
    )
    absorption_coefficient: NumberOrScalar = field(
        default=0.169,
        metadata={
            "unit": "1/m",
            "description": "Total absorption coefficient",
        },
    )
    salinity: NumberOrScalar = field(
        default=37,
        metadata={
            "unit": "‰ (ppt)",
            "description": "Salt content of water (seawater is ~35, pure water is 0)",
        },
    )
    refractive_index: NumberOrScalar = field(
        default=1.33334,
        metadata={"unit": "", "description": "Refractive index of water"},
    )
    fournier_forand_parameters: FournierForandConfig = field(
        default_factory=FournierForandConfig,
        metadata={
            "unit": "FournierForandConfig",
            "description": "Fournier-Forand phase function parameters",
        },
    )
    sublayer_dz: Union[float, None] = field(
        default=0.05, metadata={"unit": "", "description": "Height of the sublayers"}
    )
    num_sublayers: Union[int, None] = field(
        default=None, metadata={"unit": "", "description": "Number of sublayers"}
    )

    def __post_init__(self):
        if self.height <= 0:
            raise ValueError("Layer height must be positive")

        object.__setattr__(
            self,
            "scattering_coefficient",
            normalize_number_or_scalar(self.scattering_coefficient),
        )

        object.__setattr__(
            self,
            "absorption_coefficient",
            normalize_number_or_scalar(self.absorption_coefficient),
        )

        object.__setattr__(
            self,
            "salinity",
            normalize_number_or_scalar(self.salinity),
        )

        object.__setattr__(
            self,
            "refractive_index",
            normalize_number_or_scalar(self.refractive_index),
        )

        if self.num_sublayers is None:
            if self.sublayer_dz is None:
                raise ValueError("Either sublayer_dz or num_sublayers must be provided")
            object.__setattr__(
                self,
                "num_sublayers",
                max(2, int(np.ceil(self.height / self.sublayer_dz))),
            )


@dataclass(frozen=True)
class WaterConfig:
    layers: tuple[TurbidityLayerConfig, ...] = field(
        default_factory=lambda: (TurbidityLayerConfig(),),
        metadata={
            "unit": "TurbidityLayerConfig",
            "description": "Tuple of turbidity layer configs",
        },
    )
