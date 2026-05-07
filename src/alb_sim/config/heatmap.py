from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for sampled photon heatmaps."""

    enabled: bool = field(
        default=True,
        metadata={
            "description": "Whether to compute heatmaps during backward pass",
        },
    )
    bins: int = field(
        default=200,
        metadata={
            "description": "Number of bins in each dimension for the heatmaps",
        },
    )
    water_extent: float = field(
        default=1.0,
        metadata={
            "unit": "m",
            "description": "Half-width of water surface heatmap extent (centered on laser intersection)",
        },
    )
    seafloor_extent: float = field(
        default=10.0,
        metadata={
            "unit": "m",
            "description": "Half-width of seafloor heatmap extent (centered on laser intersection)",
        },
    )
