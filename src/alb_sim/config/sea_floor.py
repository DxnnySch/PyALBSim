from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeaFloorConfig:
    """Configuration for seafloor reflectance properties."""

    albedo: float = field(
        default=0.05,
        metadata={"unit": "", "description": "Proportion of light reflected"},
    )

    def __post_init__(self):
        """Validate that the seafloor albedo lies in [0, 1]."""
        if not 0 <= self.albedo <= 1:
            raise ValueError("Albedo must be between 0 and 1")
