from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def collect_interaction_data_from_photon_map(
    photon_maps_data: dict,
    interaction_type: str,
    photon_types: list | None = None,
) -> tuple:
    """
    Collect positions and energies for a given interaction type, filtering NaN values.
    Args:
        photon_maps_data: Dict mapping PhotonType -> PhotonMapData
        interaction_type: "water" for first_water_interaction, "seafloor" for seafloor_interaction
        photon_types: Optional list of PhotonType values to include. If None, all types are used.
    Returns:
        (positions, energies) as concatenated arrays, or empty arrays if no valid data.
    """
    all_positions = []
    all_energies = []
    for ptype, map_data in photon_maps_data.items():
        if photon_types is not None and ptype not in photon_types:
            continue
        positions = (
            map_data.first_water_interaction
            if interaction_type == "water"
            else map_data.seafloor_interaction
        )
        valid_mask = ~np.isnan(positions[:, 0])
        if np.any(valid_mask):
            all_positions.append(positions[valid_mask])
            all_energies.append(map_data.energies[valid_mask])
    if not all_positions:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(all_positions, axis=0), np.concatenate(all_energies, axis=0)


def interactive_photon_map_interaction_heatmap(
    photon_maps_data: dict,
    center_xz: tuple,
    interaction_type: str = "water",
    initial_bin_size: float = 0.01,
    initial_extent_radius: float = 0.2,
    energy_weighted: bool = False,
    photon_types: list | None = None,
):
    """
    Interactive heatmap of photon interaction positions with bin-size and extent sliders.
    Args:
        photon_maps_data: Dict mapping PhotonType -> PhotonMapData
        center_xz: (x, z) world coordinate to measure positions from (laser footprint center)
        interaction_type: "water" or "seafloor"
        initial_bin_size: Starting bin size in metres
        initial_extent_radius: Starting half-width of the displayed region in metres
        energy_weighted: If True, weight bins by photon energy
        photon_types: Optional list of PhotonType values to include. If None, all types.
    Returns:
        Matplotlib widget references (slider_bin, slider_extent, button_reset) to
        prevent garbage collection.
    """
    positions, energies = collect_interaction_data_from_photon_map(
        photon_maps_data, interaction_type, photon_types
    )
    if len(positions) == 0:
        print(f"No valid {interaction_type} interaction data found")
        return

    cx, cz = center_xz
    x_centered = positions[:, 0] - cx
    z_centered = positions[:, 2] - cz

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.25)

    def compute_heatmap(bin_size, extent_radius):
        bins = max(10, int(2 * extent_radius / bin_size))
        weights = energies if energy_weighted else None
        heatmap, _, _ = np.histogram2d(
            x_centered,
            z_centered,
            bins=bins,
            range=[[-extent_radius, extent_radius], [-extent_radius, extent_radius]],
            weights=weights,
        )
        return heatmap, extent_radius

    heatmap, extent = compute_heatmap(initial_bin_size, initial_extent_radius)
    extent_vals = (-extent, extent, -extent, extent)

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=extent_vals,
        aspect="equal",
        cmap="viridis" if energy_weighted else "hot",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Energy" if energy_weighted else "Photon count")

    title_prefix = "Energy-Weighted " if energy_weighted else ""
    location = "Water Surface" if interaction_type == "water" else "Seafloor"
    ax.set_title(
        f"{title_prefix}{location} Interaction Heatmap\n(Use sliders to adjust)"
    )
    ax.set_xlabel("X offset from laser center (m)")
    ax.set_ylabel("Z offset from laser center (m)")

    crosshair_color = "white" if energy_weighted else "cyan"
    ax.axhline(y=0, color=crosshair_color, linestyle="--", alpha=0.5)
    ax.axvline(x=0, color=crosshair_color, linestyle="--", alpha=0.5)

    ax_bin = plt.axes((0.25, 0.12, 0.5, 0.03))
    ax_extent = plt.axes((0.25, 0.06, 0.5, 0.03))

    slider_bin = Slider(
        ax_bin,
        "Bin Size (m)",
        0.001,
        initial_extent_radius / 5,
        valinit=initial_bin_size,
        valstep=0.001,
    )
    slider_extent = Slider(
        ax_extent,
        "Extent (m)",
        0.05,
        initial_extent_radius * 5,
        valinit=initial_extent_radius,
        valstep=0.01,
    )

    def update(val):
        bin_size = slider_bin.val
        extent_radius = slider_extent.val
        heatmap, _ = compute_heatmap(bin_size, extent_radius)
        new_extent = (-extent_radius, extent_radius, -extent_radius, extent_radius)
        im.set_data(heatmap.T)
        im.set_extent(new_extent)
        im.set_clim(vmin=heatmap.min(), vmax=heatmap.max())
        ax.set_xlim(-extent_radius, extent_radius)
        ax.set_ylim(-extent_radius, extent_radius)
        fig.canvas.draw_idle()

    slider_bin.on_changed(update)
    slider_extent.on_changed(update)

    ax_reset = plt.axes((0.8, 0.01, 0.1, 0.03))
    button_reset = Button(ax_reset, "Reset")

    def reset_all(_: Any) -> None:
        slider_bin.reset()
        slider_extent.reset()

    button_reset.on_clicked(reset_all)

    plt.show()

    return slider_bin, slider_extent, button_reset
