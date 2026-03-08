from alb_sim.config.run import RunConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.sensor import SensorConfig
from alb_sim.execution.parallel import run_parallel
from alb_sim.plotting.plot_waveform import plot_waveform

multiplier = 1
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

plot_waveform(waveform, xlim=(1850, 1980))