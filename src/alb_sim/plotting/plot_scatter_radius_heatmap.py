import numpy as np
from matplotlib import pyplot as plt

from alb_sim.config.heatmap import HeatmapConfig
from alb_sim.utils.types import MatrixN


def plot_scatter_radius_heatmap(
    scatter_radius_heatmap: MatrixN, heatmap_config: HeatmapConfig
) -> None:
    """
    Visualise the scatter-radius correlation heatmap.

    Parameters
    ----------
    scatter_radius_heatmap : MatrixN
        2D histogram of surface radius vs seafloor radius.
    heatmap_config : HeatmapConfig
        Configuration providing water and seafloor extents.
    """
    water_ext = heatmap_config.water_extent
    seafloor_ext = heatmap_config.seafloor_extent

    col_counts = scatter_radius_heatmap.sum(
        axis=1
    )  # total photons per surface-radius bin
    min_count = max(10, col_counts.max() * 0.1)  # 1 % of peak, at least 10

    col_sums = col_counts.copy()
    col_sums[col_sums == 0] = 1  # avoid division by zero
    normalized = scatter_radius_heatmap / col_sums[:, np.newaxis]
    normalized[col_counts < min_count, :] = np.nan  # mask sparse columns

    fig, ax = plt.subplots()
    im = ax.imshow(
        normalized.T,
        origin="lower",
        aspect="auto",
        extent=(0, water_ext, 0, seafloor_ext),
    )
    fig.colorbar(im, ax=ax, label="Normalized photon count (per surface-radius bin)")
    ax.set_xlabel("Surface radius from water entry center (m)")
    ax.set_ylabel("Seafloor radius from seafloor center (m)")
    ax.set_title("Forward photons: water surface radius vs seafloor radius")
    plt.tight_layout()
    plt.show()
