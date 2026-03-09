from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from alb_sim.utils.types import MatrixN


def interactive_precomputed_heatmap(
    heatmap: MatrixN,
    extent: float,
    title: str,
    cmap: str = "viridis",
    crosshair_xz: tuple = (0.0, 0.0),
):
    """
    Display a pre-computed 2D heatmap with bin-size and extent sliders.
    The original high-resolution heatmap is re-binned on the fly by summing
    groups of native pixels, so the displayed bin size is always a multiple of
    the native resolution.
    Args:
        heatmap: 2D array of energy values, shape (bins, bins)
        extent: Half-width of the square heatmap region in metres
        title: Plot title
        cmap: Matplotlib colormap name
        crosshair_xz: (x, z) position of the crosshair within the heatmap (metres,
            relative to the heatmap origin). Use (0, 0) for the water surface and
            the seafloor footprint offset for the seafloor heatmap.
    Returns:
        Matplotlib widget references (slider_bin, slider_extent, button_reset)
        to prevent garbage collection.
    """
    print(f"\n=== {title} ===")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Total energy: {heatmap.sum():.6f}")
    print(f"Max energy per bin: {heatmap.max():.2e}")

    native_bins = heatmap.shape[0]
    native_bin_size = 2 * extent / native_bins

    initial_extent = extent
    initial_bin_size = native_bin_size

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.25)

    def rebin_and_crop(bin_size, view_extent):
        """Re-bin the heatmap and crop to the requested extent."""
        # Determine grouping factor (how many native pixels per displayed bin)
        group = max(1, int(round(bin_size / native_bin_size)))
        # Trim the array so dimensions are divisible by group
        usable = (native_bins // group) * group
        trimmed = heatmap[:usable, :usable]
        rebinned = trimmed.reshape(usable // group, group, usable // group, group).sum(
            axis=(1, 3)
        )

        rebinned_extent = extent * usable / native_bins
        rebinned_bin_size = 2 * rebinned_extent / rebinned.shape[0]

        # Crop to view_extent
        if view_extent < rebinned_extent:
            half = rebinned.shape[0] // 2
            keep = max(1, int(view_extent / rebinned_bin_size))
            lo = max(0, half - keep)
            hi = min(rebinned.shape[0], half + keep)
            cropped = rebinned[lo:hi, lo:hi]
            crop_extent = (hi - lo) * rebinned_bin_size / 2
        else:
            cropped = rebinned
            crop_extent = rebinned_extent

        return cropped, crop_extent

    display, disp_ext = rebin_and_crop(initial_bin_size, initial_extent)
    extent_vals = (-disp_ext, disp_ext, -disp_ext, disp_ext)

    im = ax.imshow(
        display.T,
        origin="lower",
        extent=extent_vals,
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Detected energy contribution")

    ax.set_xlabel("X offset from laser center (m)")
    ax.set_ylabel("Z offset from laser center (m)")
    ax.set_title(f"{title}\n(photons that actually reached the sensor)")
    ax.axhline(y=crosshair_xz[1], color="white", linestyle="--", alpha=0.5)
    ax.axvline(x=crosshair_xz[0], color="white", linestyle="--", alpha=0.5)

    ax_bin = plt.axes((0.25, 0.12, 0.5, 0.03))
    ax_extent_slider = plt.axes((0.25, 0.06, 0.5, 0.03))

    slider_bin = Slider(
        ax_bin,
        "Bin Size (m)",
        native_bin_size,
        extent / 5,
        valinit=initial_bin_size,
        valstep=native_bin_size,
    )
    slider_extent = Slider(
        ax_extent_slider,
        "Extent (m)",
        0.05,
        extent,
        valinit=initial_extent,
        valstep=0.01,
    )

    def update(val):
        display, disp_ext = rebin_and_crop(slider_bin.val, slider_extent.val)
        new_extent = (-disp_ext, disp_ext, -disp_ext, disp_ext)
        im.set_data(display.T)
        im.set_extent(new_extent)
        im.set_clim(vmin=display.min(), vmax=display.max())
        ax.set_xlim(-disp_ext, disp_ext)
        ax.set_ylim(-disp_ext, disp_ext)
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
