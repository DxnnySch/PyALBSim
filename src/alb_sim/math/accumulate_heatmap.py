import numpy as np

from alb_sim.utils.types import Array, MatrixN, Vector3, Vector3Array


def accumulate_to_heatmap(
    positions: Vector3Array,
    energies: Array,
    heatmap: MatrixN,
    center: Vector3,
    extent: float,
    bin_size: float,
) -> None:
    """
    Accumulate energy values into a 2D histogram based on x-z positions.

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
        Photon interaction positions.
    energies : ndarray of shape (N,)
        Energy associated with each interaction.
    heatmap : ndarray of shape (B, B)
        2D heatmap array to accumulate into.
    center : ndarray of shape (3,)
        Heatmap centre position in world coordinates.
    extent : float
        Half-width of the heatmap extent around the centre.
    bin_size : float
        Size of each heatmap bin in world units.
    """
    # Filter out NaN positions
    valid_mask = ~np.isnan(positions[:, 0])
    if not np.any(valid_mask):
        return

    valid_pos = positions[valid_mask]
    valid_energy = energies[valid_mask]

    # Center positions relative to heatmap center
    x_centered = valid_pos[:, 0] - center[0]
    z_centered = valid_pos[:, 2] - center[2]

    # Compute bin indices
    bins = heatmap.shape[0]
    x_idx = ((x_centered + extent) / bin_size).astype(np.int64)
    z_idx = ((z_centered + extent) / bin_size).astype(np.int64)

    # Filter to valid bin range
    valid_bins = (x_idx >= 0) & (x_idx < bins) & (z_idx >= 0) & (z_idx < bins)

    if not np.any(valid_bins):
        return

    x_idx = x_idx[valid_bins]
    z_idx = z_idx[valid_bins]
    valid_energy = valid_energy[valid_bins]

    # Accumulate using linear indexing for np.add.at
    linear_idx = x_idx * bins + z_idx
    np.add.at(heatmap.ravel(), linear_idx, valid_energy)


def accumulate_scatter_radius_heatmap(
    scatter_radius_heatmap: MatrixN,
    surface_positions: Vector3Array,
    seafloor_positions: Vector3Array,
    water_surface_center: Vector3,
    sea_floor_center: Vector3,
    water_surface_extent: float,
    sea_floor_extent: float,
) -> None:
    """
    Accumulate photon counts into a 2D histogram of (surface_radius, seafloor_radius).

    Each valid photon contributes a count of 1; this is not energy-weighted.

    Parameters
    ----------
    scatter_radius_heatmap : ndarray of shape (B, B)
        2D histogram to accumulate photon counts into.
    surface_positions : ndarray of shape (N, 3)
        First water interaction positions at the surface.
    seafloor_positions : ndarray of shape (N, 3)
        Corresponding photon interaction positions at the seafloor.
    water_surface_center : ndarray of shape (3,)
        Centre point used for computing surface radii.
    sea_floor_center : ndarray of shape (3,)
        Centre point used for computing seafloor radii.
    water_surface_extent : float
        Maximum radius on the surface represented in the heatmap.
    sea_floor_extent : float
        Maximum radius on the seafloor represented in the heatmap.
    """
    valid_mask = ~np.isnan(surface_positions[:, 0])
    if not np.any(valid_mask):
        return

    surf = surface_positions[valid_mask]
    floor = seafloor_positions[valid_mask]

    surface_radius = np.sqrt(
        (surf[:, 0] - water_surface_center[0]) ** 2
        + (surf[:, 2] - water_surface_center[2]) ** 2
    )
    seafloor_radius = np.sqrt(
        (floor[:, 0] - sea_floor_center[0]) ** 2
        + (floor[:, 2] - sea_floor_center[2]) ** 2
    )

    num_bins = scatter_radius_heatmap.shape[0]
    x_idx = ((surface_radius / water_surface_extent) * num_bins).astype(np.int64)
    y_idx = ((seafloor_radius / sea_floor_extent) * num_bins).astype(np.int64)

    valid_bins = (x_idx >= 0) & (x_idx < num_bins) & (y_idx >= 0) & (y_idx < num_bins)

    if not np.any(valid_bins):
        return

    np.add.at(
        scatter_radius_heatmap.ravel(),
        x_idx[valid_bins] * num_bins + y_idx[valid_bins],
        1,
    )
