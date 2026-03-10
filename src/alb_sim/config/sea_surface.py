from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeaSurfaceConfig:
    """Configuration for sea surface reflectance and roughness."""

    albedo: float = field(
        default=0.1,
        metadata={"unit": "", "description": "Proportion of light reflected"},
    )
    roughness: float = field(
        default=0.05, metadata={"unit": "", "description": "Roughness of the surface"}
    )

    def __post_init__(self):
        """Validate that sea surface parameters are physically plausible."""
        if not 0 <= self.albedo <= 1:
            raise ValueError("Albedo must be between 0 and 1")
