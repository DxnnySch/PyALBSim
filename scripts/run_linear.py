import secrets
import numpy as np
from alb_sim.core.simulation import Simulation
from alb_sim.execution.linear import linear_backward, linear_forward, run_linear
from alb_sim.config.simulation import SimulationConfig
from alb_sim.config.run import RunConfig
from alb_sim.photon_mapping.build_photon_map_data import build_photon_map_data
from alb_sim.photon_mapping.photon_map_index import PhotonMapIndex
from alb_sim.photon_mapping.print_photon_map_stats import photon_map_stats

# Setup random number generator
rng = np.random.default_rng(secrets.randbits(128))

# Configuration
simulation_config = SimulationConfig()
run_config = RunConfig()

simulation = Simulation(simulation_config, rng)

# forwards
linear_forward(simulation, run_config.photons_per_batch_forward, run_config.batches_forward)

# combine batches
photon_maps_data = build_photon_map_data(simulation.photon_storage)
print("\n" + photon_map_stats(photon_maps_data))
for photon_type, data in photon_maps_data.items():
    simulation.photon_maps[photon_type] = PhotonMapIndex(data)

linear_backward(simulation, run_config.photons_per_batch_backward, run_config.batches_backward)

data = simulation.return_waveform

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