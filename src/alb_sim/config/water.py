from dataclasses import dataclass, field


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
    scattering_coefficient: float = field(
        default=2.5,#1.21,
        metadata={
            "unit": "1/m",
            "description": "Total scattering coefficient",
        },
    )
    absorption_coefficient: float = field(
        default=0.169,
        metadata={
            "unit": "1/m",
            "description": "Total absorption coefficient",
        },
    )
    salinity: float = field(
        default=37,  # 1,
        metadata={
            "unit": "‰ (ppt)",
            "description": "Salt content of water (seawater is ~35, pure water is 0)",
        },
    )
    refractive_index: float = field(
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

    def __post_init__(self):
        if self.height <= 0:
            raise ValueError("Layer height must be positive")


@dataclass(frozen=True)
class WaterConfig:
    layers: tuple[TurbidityLayerConfig, ...] = field(
        default_factory=lambda: (TurbidityLayerConfig(),),
        metadata={
            "unit": "TurbidityLayerConfig",
            "description": "Tuple of turbidity layer configs",
        },
    )
