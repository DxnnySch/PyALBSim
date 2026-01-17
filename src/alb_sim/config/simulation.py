from dataclasses import dataclass, field

from alb_sim.config.laser import LaserConfig
from alb_sim.config.scene import SceneConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.sea_surface import SeaSurfaceConfig
from alb_sim.config.sensor import SensorConfig
from alb_sim.config.water import WaterConfig


@dataclass(frozen=True)
class SimulationConfig:
    sample_multiplier: int = field(
        default=10,
        metadata={
            "unit": "",
            "description": "Number of sub-sampling steps during sensor step",
        },
    )
    photon_mapping_k: int = field(
        default=100,
        metadata={
            "unit": "",
            "description": "Number of nearest neighbors to query during photon mapping",
        },
    )
    scene: SceneConfig = field(
        default_factory=SceneConfig,
        metadata={"unit": "SceneConfig", "description": "Scene config"},
    )
    water: WaterConfig = field(
        default_factory=WaterConfig,
        metadata={"unit": "WaterConfig", "description": "Water config"},
    )
    laser: LaserConfig = field(
        default_factory=LaserConfig,
        metadata={"unit": "LaserConfig", "description": "Laser config"},
    )
    sensor: SensorConfig = field(
        default_factory=SensorConfig,
        metadata={"unit": "SensorConfig", "description": "Sensor config"},
    )
    sea_floor: SeaFloorConfig = field(
        default_factory=SeaFloorConfig,
        metadata={"unit": "SeaFloorConfig", "description": "Sea floor config"},
    )
    sea_surface: SeaSurfaceConfig = field(
        default_factory=SeaSurfaceConfig,
        metadata={"unit": "SeaSurfaceConfig", "description": "Sea surface config"},
    )
