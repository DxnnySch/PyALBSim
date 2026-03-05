import os
from alb_sim.config.run import RunConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.sea_surface import SeaSurfaceConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig, FournierForandConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.sensor import SensorConfig
from alb_sim.execution.parallel import run_parallel, merge_results
from alb_sim.utils.parameter_profile import LinearParameter, ExponentialParameter

multiplier = 5
simulation_config = SimulationConfig(
    water=WaterConfig(
        layers=(
            TurbidityLayerConfig(
                height=5,
                absorption_coefficient=0.114,
                scattering_coefficient=1.21,
            ),
        )
    ),
    sensor=SensorConfig(sample_rate=2e9),
    sea_floor=SeaFloorConfig(albedo=0.01),
    sample_multiplier=10,
)
run_config = RunConfig(
    processes=8,
    batches_forward=(16) * multiplier,
    batches_backward=(16) * multiplier,
)

waveform = run_parallel(simulation_config, run_config)

data = waveform

import numpy as np
import matplotlib.pyplot as plt
from alb_sim.utils.water_layer_steps import get_water_layer_steps

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.sum(list(data.values()), axis=0))
ax.set_xlim(1850, 1980)
ax.set_xlabel("Step / Distance")
ax.set_ylabel("Photon contribution")
ax.set_title("Photon contributions")
plt.tight_layout()
plt.show()