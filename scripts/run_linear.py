import numpy as np

from alb_sim.config.run import RunConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.execution.linear import run_linear
from alb_sim.plotting.plot_stacked_waveform import plot_stacked_waveform
from alb_sim.plotting.plot_waveform import plot_waveform

# Configuration
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
run_config = RunConfig()

waveform = run_linear(simulation_config, run_config)

plot_stacked_waveform(waveform)

plot_waveform(np.sum(list(waveform.values()), axis=0))
