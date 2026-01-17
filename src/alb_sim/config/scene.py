from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneConfig:
    flying_height: float = field(
        default=135,
        metadata={"unit": "m", "description": "Height above water of sensor and laser"},
    )

    def __post_init__(self):
        if self.flying_height <= 0:
            raise ValueError("Flying height must be positive")
