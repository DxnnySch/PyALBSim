from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeaSurfaceConfig:
    albedo: float = field(
        default=0.1,
        metadata={"unit": "", "description": "Proportion of light reflected"},
    )
    roughness: float = field(
        default=0.035, metadata={"unit": "", "description": "Roughness of the surface"}
    )

    def __post_init__(self):
        if not 0 <= self.albedo <= 1:
            raise ValueError("Albedo must be between 0 and 1")
