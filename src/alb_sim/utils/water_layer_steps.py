import numpy as np
from alb_sim.config.simulation import SimulationConfig
from alb_sim.physics.constants import LIGHT_SPEED_AIR
from alb_sim.physics.models.simulation import SimulationModel


def get_water_layer_steps(simulation_config: SimulationConfig):
    simulation_model = SimulationModel(simulation_config)

    laser_direction = simulation_model.laser.direction

    water_surface_steps = (
        (simulation_model.water_surface_y / laser_direction[1])
        / LIGHT_SPEED_AIR
        * simulation_model.sample_rate
    )
    return_steps = [water_surface_steps]

    refracted_direction = simulation_model.sea_surface.calculate_refraction_direction(
        np.array([laser_direction]), np.array([0, 1, 0])
    )[0][0]

    for layer in simulation_model.water.layers:
        height = layer.height
        return_steps.append(
            return_steps[-1]
            + (height / (-1 * refracted_direction[1]))
            / layer.light_speed_at(0)
            * simulation_model.sample_rate
        )

    return return_steps
