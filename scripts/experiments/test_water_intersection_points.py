import matplotlib.pyplot as plt
import numpy as np

from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.physics.models.simulation import SimulationModel

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

simulation_model = SimulationModel(simulation_config)

center_intersection = simulation_model.laser.direction * (
    simulation_model.water_surface_y / simulation_model.laser.direction[1]
)

num_photons = 10_000
rng = np.random.default_rng(42)
directions = simulation_model.sample_starting_direction(num_photons, rng, forward=True)

y_distances = directions[:, 1]

steps_to_surface = simulation_model.water_surface_y / y_distances

intersections = directions * steps_to_surface[:, np.newaxis]

plt.scatter(
    intersections[:, 0] - center_intersection[0],
    intersections[:, 2] - center_intersection[2],
)
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.show()
