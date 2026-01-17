from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeaFloorConfig:
    albedo: float = field(
        default=0.05,
        metadata={"unit": "", "description": "Proportion of light reflected"},
    )

    def __post_init__(self):
        if not 0 <= self.albedo <= 1:
            raise ValueError("Albedo must be between 0 and 1")
