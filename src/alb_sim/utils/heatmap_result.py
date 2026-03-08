from dataclasses import dataclass

from alb_sim.utils.types import MatrixN


@dataclass
class SampledHeatmapsResult:
    """Container for sampled photon heatmaps."""

    water_heatmap: MatrixN  # (bins, bins) - first_water_interaction energy
    seafloor_heatmap: MatrixN  # (bins, bins) - seafloor_interaction energy
    scatter_radius_heatmap: (
        MatrixN  # (bins, bins) - surface radius vs seafloor radius counts
    )
    # water_extent: float  # Half-width of water heatmap extent
    # seafloor_extent: float  # Half-width of seafloor heatmap extent
    # center: Vector3  # (3,) - Center point (water intersection)
