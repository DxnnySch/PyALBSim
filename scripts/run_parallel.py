import os
from alb_sim.config.run import RunConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.execution.parallel import run_parallel

simulation_config = SimulationConfig(
    water=WaterConfig(
        layers=(
            TurbidityLayerConfig(
                height=7, absorption_coefficient=0.01, scattering_coefficient=3.5
            ),
        )
    ),
    sea_floor=SeaFloorConfig(albedo=0.1),
)
run_config = RunConfig(
    processes=8,
    batches_forward=(8) * 3,
    batches_backward=(8) * 5,
)

waveform = run_parallel(simulation_config, run_config)

data = waveform

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(next(iter(data.values()))))
labels = [pt.name for pt in data.keys()]
values = np.vstack(list(data.values()))
plt.figure()
plt.stackplot(x, values, labels=labels)
plt.legend(loc="upper left")
plt.xlabel("Step / Distance")
plt.ylabel("Photon contribution")
plt.title("Photon contributions (stacked)")
plt.show()
