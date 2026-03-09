from dataclasses import dataclass

from alb_sim.utils.types import Array, Vector3Array


@dataclass
class PhotonWrapper:
    positions: Vector3Array
    directions: Vector3Array
    energies: Array
    optical_depth: Array
    optical_depth_target: Array
    time_deltas: Array
    first_water_interaction: Vector3Array
    seafloor_interaction: Vector3Array

    def subset(self, idx):
        """View-based slicing (no copies)."""
        return PhotonWrapper(
            positions=self.positions[idx],
            directions=self.directions[idx],
            energies=self.energies[idx],
            optical_depth=self.optical_depth[idx],
            optical_depth_target=self.optical_depth_target[idx],
            time_deltas=self.time_deltas[idx],
            first_water_interaction=self.first_water_interaction[idx],
            seafloor_interaction=self.seafloor_interaction[idx],
        )

    def copy(self):
        return PhotonWrapper(
            positions=self.positions.copy(),
            directions=self.directions.copy(),
            energies=self.energies.copy(),
            optical_depth=self.optical_depth.copy(),
            optical_depth_target=self.optical_depth_target.copy(),
            time_deltas=self.time_deltas.copy(),
            first_water_interaction=self.first_water_interaction.copy(),
            seafloor_interaction=self.seafloor_interaction.copy(),
        )
