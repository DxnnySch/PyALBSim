import numpy as np

from alb_sim.config.run import RunConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.execution.parallel import run_parallel
from alb_sim.plotting.plot_stacked_waveform import plot_stacked_waveform
from alb_sim.plotting.plot_waveform import plot_waveform

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

waveform, _, _ = run_parallel(simulation_config, run_config)

plot_stacked_waveform(waveform)

plot_waveform(np.sum(list(waveform.values()), axis=0))
