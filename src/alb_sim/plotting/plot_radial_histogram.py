import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from alb_sim.config.heatmap import HeatmapConfig
from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.plotting.photon_map_interaction_interactive_heatmap import (
    collect_interaction_data_from_photon_map,
)


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian(centers: np.ndarray, hist: np.ndarray):
    """
    Fit a Gaussian to a 1-D histogram. Returns (popt, smooth_x, smooth_y) or
    None if the fit fails.
    """
    total = hist.sum()
    if total <= 0:
        return None
    mu0 = float(np.average(centers, weights=np.maximum(hist, 0)))
    sigma0 = float(
        np.sqrt(np.average((centers - mu0) ** 2, weights=np.maximum(hist, 0)))
    )
    sigma0 = max(sigma0, float(centers[1] - centers[0]))
    amp0 = float(hist.max())
    try:
        popt, _ = curve_fit(
            _gaussian,
            centers,
            hist,
            p0=[amp0, mu0, sigma0],
            maxfev=5000,
        )
        smooth_x = np.linspace(centers[0], centers[-1], 500)
        smooth_y = _gaussian(smooth_x, *popt)
        return popt, smooth_x, smooth_y
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _radial_histogram_from_heatmap(
    heatmap: np.ndarray,
    extent: float,
    center_offset: tuple,
    n_bins: int,
) -> tuple:
    """
    Compute a 1D radial energy histogram by binning a 2D heatmap into rings.

    Parameters
    ----------
    heatmap : ndarray, shape (N, N)
        2D energy array with coordinates spanning ``[-extent, extent]``.
    extent : float
        Half-width of the heatmap in metres.
    center_offset : tuple
        Offset of the laser-axis footprint within the heatmap (x, z).
    n_bins : int
        Number of radial bins.

    Returns
    -------
    tuple
        ``(r_centers, r_edges, r_hist, max_radius)``.
    """
    num_bins = heatmap.shape[0]
    edges = np.linspace(-extent, extent, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    xx, zz = np.meshgrid(centers, centers, indexing="ij")
    cx, cz = center_offset
    radii = np.sqrt((xx - cx) ** 2 + (zz - cz) ** 2).ravel()
    energy = heatmap.ravel()

    max_radius = extent * math.sqrt(2)
    r_edges = np.linspace(0, max_radius, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    r_hist, _ = np.histogram(radii, bins=r_edges, weights=energy)
    return r_centers, r_edges, r_hist, max_radius


def _radial_histogram_from_photons(
    photon_maps_data: dict,
    interaction_type: str,
    center_xz: tuple,
    max_radius: float,
    n_bins: int,
    photon_types: Optional[list] = None,
) -> tuple:
    """
    Compute a 1D radial energy histogram from individual photon positions.

    Parameters
    ----------
    photon_maps_data : dict
        Mapping from PhotonType to PhotonMapData.
    interaction_type : str
        ``"water"`` for first-water interactions or ``"seafloor"`` for bottom hits.
    center_xz : tuple
        Laser-axis footprint position (x, z) in world coordinates.
    max_radius : float
        Upper edge of the last radial bin.
    n_bins : int
        Number of radial bins.
    photon_types : list, optional
        Optional subset of PhotonType values to include.

    Returns
    -------
    tuple
        ``(r_centers, r_edges, r_hist, max_radius)``.
    """
    positions, energies = collect_interaction_data_from_photon_map(
        photon_maps_data, interaction_type, photon_types
    )

    r_edges = np.linspace(0, max_radius, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    if len(positions) == 0:
        return r_centers, r_edges, np.zeros(n_bins), max_radius

    cx, cz = center_xz
    radii = np.sqrt((positions[:, 0] - cx) ** 2 + (positions[:, 2] - cz) ** 2)
    r_hist, _ = np.histogram(radii, bins=r_edges, weights=energies)
    return r_centers, r_edges, r_hist, max_radius


def _cumulative_pct(r_hist: np.ndarray) -> np.ndarray:
    total = r_hist.sum()
    if total == 0:
        return np.zeros_like(r_hist, dtype=float)
    return 100.0 * np.cumsum(r_hist) / total


def _add_percentile_markers(
    ax: plt.Axes,
    r_centers: np.ndarray,
    cumulative_pct: np.ndarray,
    percentiles: tuple = (50, 90, 95, 99, 100),
):
    """Annotate cumulative curves with percentile crosshairs and radius labels."""
    for pct in percentiles:
        r_at_pct = float(np.interp(pct, cumulative_pct, r_centers))
        ax.axhline(pct, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(r_at_pct, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.annotate(
            f"{pct}% @ {r_at_pct * 1000:.1f} mm",
            xy=(r_at_pct, pct),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="red",
        )


# ---------------------------------------------------------------------------
# Internal figure builders
# ---------------------------------------------------------------------------


def _plot_single_radial_figure(panels: list, title: str):
    """
    Draw a 2×2 figure of radial energy histograms and cumulative curves.

    Parameters
    ----------
    panels : list
        List of ``(r_centers, r_edges, r_hist, label)`` tuples, one per column.
    title : str
        Figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (r_centers, r_edges, r_hist, label) in enumerate(panels):
        cum_pct = _cumulative_pct(r_hist)
        max_radius = float(r_edges[-1])
        bar_width = float(r_edges[1] - r_edges[0])

        ax_hist = axes[0, col]
        ax_hist.bar(
            r_centers,
            r_hist,
            width=bar_width,
            align="center",
            color="steelblue",
            edgecolor="none",
        )
        ax_hist.set_xlabel("Radius from laser center (m)")
        ax_hist.set_ylabel("Energy")
        ax_hist.set_title(f"{label}: Energy by Radius")
        ax_hist.set_xlim(0, max_radius)

        ax_cum = axes[1, col]
        ax_cum.plot(r_centers, cum_pct, color="steelblue")
        ax_cum.set_xlabel("Radius from laser center (m)")
        ax_cum.set_ylabel("Cumulative energy (%)")
        ax_cum.set_title(f"{label}: Cumulative Energy")
        ax_cum.set_xlim(0, max_radius)
        ax_cum.set_ylim(0, 100)
        _add_percentile_markers(ax_cum, r_centers, cum_pct)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _plot_combined_radial_figure(
    water: tuple,
    seafloor: tuple,
):
    """
    Draw a 2×2 figure comparing outgoing vs detected radial energy distributions.

    Parameters
    ----------
    water, seafloor : tuple
        Each a tuple ``(r_centers, r_edges, det_hist, out_hist)`` for the surface.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (water, "Water Surface", axes[0, 0], axes[1, 0]),
        (seafloor, "Seafloor", axes[0, 1], axes[1, 1]),
    ]

    for (r_centers, r_edges, det_hist, out_hist), label, ax_hist, ax_cum in panels:
        max_radius = float(r_edges[-1])
        bar_width = float(r_edges[1] - r_edges[0])

        out_cum = _cumulative_pct(out_hist)
        det_cum = _cumulative_pct(det_hist)

        # Absolute energy histogram
        ax_hist.bar(
            r_centers,
            out_hist,
            width=bar_width,
            align="center",
            color="steelblue",
            alpha=0.6,
            label="Outgoing",
        )
        ax_hist.bar(
            r_centers,
            det_hist,
            width=bar_width,
            align="center",
            color="orange",
            alpha=0.7,
            label="Detected",
        )
        ax_hist.set_xlabel("Radius from laser center (m)")
        ax_hist.set_ylabel("Energy")
        ax_hist.set_title(f"{label}: Outgoing vs Detected")
        ax_hist.set_xlim(0, max_radius)
        ax_hist.legend()

        # Cumulative (each 0→100%)
        ax_cum.plot(r_centers, out_cum, color="steelblue", label="Outgoing")
        ax_cum.plot(r_centers, det_cum, color="orange", label="Detected")
        ax_cum.set_xlabel("Radius from laser center (m)")
        ax_cum.set_ylabel("Cumulative energy (%)")
        ax_cum.set_title(f"{label}: Cumulative Energy")
        ax_cum.set_xlim(0, max_radius)
        ax_cum.set_ylim(0, 100)
        ax_cum.legend(loc="lower right")

    fig.suptitle(
        "Outgoing vs Detected — Absolute Energy Comparison\n"
        "Outgoing = forward-pass photon energy at interaction point  |  "
        "Detected = backward-pass sampled contribution (normalized per sensor ray)"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Directional slice helpers
# ---------------------------------------------------------------------------


def _slice_from_heatmap(
    heatmap: np.ndarray,
    extent: float,
    center_offset: tuple,
    n_bins: int,
    axis: str,
) -> tuple:
    """
    Project a 2D heatmap onto one spatial axis.

    Parameters
    ----------
    heatmap : ndarray
        2D energy array with x along axis-0 and z along axis-1.
    extent : float
        Half-width of the heatmap in metres.
    center_offset : tuple
        Offset of the laser footprint within the heatmap (x, z).
    n_bins : int
        Number of output bins.
    axis : {"x", "z"}
        Axis along which to project.

    Returns
    -------
    tuple
        ``(bin_centers, bin_edges, hist_1d)``.
    """
    num_bins = heatmap.shape[0]
    pixel_edges = np.linspace(-extent, extent, num_bins + 1)
    pixel_centers = (pixel_edges[:-1] + pixel_edges[1:]) / 2

    cx, cz = center_offset
    if axis == "x":
        # Sum along z (axis-1), bin by x
        marginal = heatmap.sum(axis=1)  # shape (N,)
        coords = pixel_centers - cx
    else:
        # Sum along x (axis-0), bin by z
        marginal = heatmap.sum(axis=0)  # shape (N,)
        coords = pixel_centers - cz

    bin_edges = np.linspace(-extent, extent, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(coords, bins=bin_edges, weights=marginal)
    return bin_centers, bin_edges, hist


def _slice_from_photons(
    photon_maps_data: dict,
    interaction_type: str,
    center_xz: tuple,
    extent: float,
    n_bins: int,
    axis: str,
    photon_types: Optional[list] = None,
) -> tuple:
    """
    Project photon interaction positions onto one spatial axis.

    Parameters
    ----------
    photon_maps_data : dict
        Mapping from PhotonType to PhotonMapData.
    interaction_type : str
        ``"water"`` or ``"seafloor"``.
    center_xz : tuple
        Laser-footprint position (x, z) in world coordinates.
    extent : float
        Half-width of the binning range.
    n_bins : int
        Number of output bins.
    axis : {"x", "z"}
        Axis along which to project.
    photon_types : list, optional
        Optional subset of PhotonType values to include.

    Returns
    -------
    tuple
        ``(bin_centers, bin_edges, hist_1d)``.
    """
    positions, energies = collect_interaction_data_from_photon_map(
        photon_maps_data, interaction_type, photon_types
    )

    bin_edges = np.linspace(-extent, extent, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if len(positions) == 0:
        return bin_centers, bin_edges, np.zeros(n_bins)

    cx, cz = center_xz
    coords = positions[:, 0] - cx if axis == "x" else positions[:, 2] - cz

    hist, _ = np.histogram(coords, bins=bin_edges, weights=energies)
    return bin_centers, bin_edges, hist


def _plot_directional_figure(
    water_x: tuple,
    water_z: tuple,
    seafloor_x: tuple,
    seafloor_z: tuple,
    title: str,
):
    """
    Draw a 2×2 figure comparing X vs Z directional slices for water and seafloor.

    Parameters
    ----------
    water_x, water_z, seafloor_x, seafloor_z : tuple
        Each a tuple ``(bin_centers, bin_edges, hist_1d)``.
    title : str
        Figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (water_x, water_z, "Water Surface", axes[0, 0], axes[1, 0]),
        (seafloor_x, seafloor_z, "Seafloor", axes[0, 1], axes[1, 1]),
    ]

    for (xc, xe, xh), (zc, _, zh), label, ax_hist, ax_cum in panels:
        bar_width = float(xe[1] - xe[0])
        half_width = bar_width / 2

        ax_hist.bar(
            xc - half_width / 2,
            xh,
            width=half_width,
            align="center",
            color="steelblue",
            alpha=0.5,
            label="X-direction",
        )
        ax_hist.bar(
            zc + half_width / 2,
            zh,
            width=half_width,
            align="center",
            color="darkorange",
            alpha=0.5,
            label="Z-direction",
        )

        # Fit and overlay Gaussians
        x_fit = _fit_gaussian(xc, xh)
        z_fit = _fit_gaussian(zc, zh)
        if x_fit is not None:
            (_, mu, sig), sx, sy = x_fit
            ax_hist.plot(
                sx,
                sy,
                color="steelblue",
                linewidth=2,
                label=f"X fit: \u03bc={mu*1e3:.1f} mm, \u03c3={sig*1e3:.1f} mm",
            )
        if z_fit is not None:
            (_, mu, sig), sz_x, sz_y = z_fit
            ax_hist.plot(
                sz_x,
                sz_y,
                color="darkorange",
                linewidth=2,
                label=f"Z fit: \u03bc={mu*1e3:.1f} mm, \u03c3={sig*1e3:.1f} mm",
            )

        ax_hist.set_xlabel("Offset from laser center (m)")
        ax_hist.set_ylabel("Energy")
        ax_hist.set_title(f"{label}: X vs Z slice")
        ax_hist.set_xlim(float(xe[0]), float(xe[-1]))
        ax_hist.axvline(0, color="grey", linestyle="--", alpha=0.4)
        ax_hist.legend(fontsize=7)

        x_cum = _cumulative_pct(xh)
        z_cum = _cumulative_pct(zh)
        ax_cum.plot(xc, x_cum, color="steelblue", label="X-direction")
        ax_cum.plot(zc, z_cum, color="darkorange", label="Z-direction")
        ax_cum.set_xlabel("Offset from laser center (m)")
        ax_cum.set_ylabel("Cumulative energy (%)")
        ax_cum.set_title(f"{label}: Cumulative X vs Z")
        ax_cum.set_xlim(float(xe[0]), float(xe[-1]))
        ax_cum.set_ylim(0, 100)
        ax_cum.axvline(0, color="grey", linestyle="--", alpha=0.4)
        ax_cum.legend(loc="lower right")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_radial_energy_histograms(
    sampled_heatmaps,
    photon_maps_data: dict,
    heatmap_config: HeatmapConfig,
    water_center_xz: tuple,
    seafloor_center_xz: tuple,
    n_backward_total: int,
    n_bins: int = 100,
):
    """
    Plot multiple figures summarising radial energy distributions from photon maps.

    Figures include detected vs outgoing energy by radius and directional slices
    for both water surface and seafloor interactions.

    Parameters
    ----------
    sampled_heatmaps : SampledHeatmapsResult
        Pre-computed detected-energy heatmaps.
    photon_maps_data : dict
        All stored photon data from the forward pass.
    heatmap_config : HeatmapConfig
        Heatmap configuration (extents, etc.).
    water_center_xz : tuple
        Water surface laser intersection (x, z) in world coordinates.
    seafloor_center_xz : tuple
        Seafloor laser footprint (x, z) in world coordinates.
    n_backward_total : int
        Total number of backward photons simulated (for normalisation).
    n_bins : int, optional
        Number of radial bins (shared across histograms).
    """

    # ── Detected (from pre-computed heatmaps) ─────────────────────────────
    det_w_rc, det_w_re, det_w_hist, det_w_maxr = _radial_histogram_from_heatmap(
        sampled_heatmaps.water_heatmap,
        heatmap_config.water_extent,
        (0.0, 0.0),
        n_bins,
    )
    det_sf_rc, det_sf_re, det_sf_hist, det_sf_maxr = _radial_histogram_from_heatmap(
        sampled_heatmaps.seafloor_heatmap,
        heatmap_config.seafloor_extent,
        (0, 0),
        n_bins,
    )

    # ── Outgoing (from photon data, same radial range as detected) ─────────
    # Use SURFACE_REFLECTION for water: in handle_enter ALL entering photons
    # are stored as SURFACE_REFLECTION (via subset.copy()) with energy=1.0
    # BEFORE Fresnel attenuation. Using all types would double-count (same
    # photon appears in SURFACE_REFLECTION and later in SCATTER etc.) and
    # use attenuated energies that don't represent energy at the water surface.
    out_w_rc, out_w_re, out_w_hist, _ = _radial_histogram_from_photons(
        photon_maps_data,
        "water",
        water_center_xz,
        det_w_maxr,
        n_bins,
        photon_types=[PhotonType.SURFACE_REFLECTION],
    )
    # Use BOTTOM_REFLECTION for seafloor: in handle_seafloor ALL photons
    # hitting the seafloor are stored as BOTTOM_REFLECTION (via subset.copy())
    # BEFORE seafloor albedo is applied.
    out_sf_rc, out_sf_re, out_sf_hist, _ = _radial_histogram_from_photons(
        photon_maps_data,
        "seafloor",
        seafloor_center_xz,
        det_sf_maxr,
        n_bins,
        photon_types=[PhotonType.BOTTOM_REFLECTION],
    )

    # ── Figure 1: Detected only ────────────────────────────────────────────
    _plot_single_radial_figure(
        [
            (det_w_rc, det_w_re, det_w_hist, "Water Surface"),
            (det_sf_rc, det_sf_re, det_sf_hist, "Seafloor"),
        ],
        "Detected Energy Distribution by Radius\n(photons that reached the sensor)",
    )

    # ── Figure 2: Outgoing only ────────────────────────────────────────────
    _plot_single_radial_figure(
        [
            (out_w_rc, out_w_re, out_w_hist, "Water Surface"),
            (out_sf_rc, out_sf_re, out_sf_hist, "Seafloor"),
        ],
        "Total Outgoing Energy Distribution by Radius\n(all tracked photons)",
    )

    # ── Figure 3: Outgoing vs Detected comparison ─────────────────────────
    # Normalize detected by N_backward so it represents average detected
    # energy per backward ray — comparable to forward-pass energy units.
    _plot_combined_radial_figure(
        water=(det_w_rc, det_w_re, det_w_hist / n_backward_total, out_w_hist),
        seafloor=(det_sf_rc, det_sf_re, det_sf_hist / n_backward_total, out_sf_hist),
    )

    # ── Figure 4 & 5: Directional slices (outgoing, then detected) ────────
    w_ext = heatmap_config.water_extent
    sf_ext = heatmap_config.seafloor_extent

    # Outgoing directional slices
    out_wx = _slice_from_photons(
        photon_maps_data,
        "water",
        water_center_xz,
        w_ext,
        n_bins,
        "x",
        photon_types=[PhotonType.SURFACE_REFLECTION],
    )
    out_wz = _slice_from_photons(
        photon_maps_data,
        "water",
        water_center_xz,
        w_ext,
        n_bins,
        "z",
        photon_types=[PhotonType.SURFACE_REFLECTION],
    )
    out_sfx = _slice_from_photons(
        photon_maps_data,
        "seafloor",
        seafloor_center_xz,
        sf_ext,
        n_bins,
        "x",
        photon_types=[PhotonType.BOTTOM_REFLECTION],
    )
    out_sfz = _slice_from_photons(
        photon_maps_data,
        "seafloor",
        seafloor_center_xz,
        sf_ext,
        n_bins,
        "z",
        photon_types=[PhotonType.BOTTOM_REFLECTION],
    )
    _plot_directional_figure(
        out_wx,
        out_wz,
        out_sfx,
        out_sfz,
        "Outgoing Energy — Directional Slices (X vs Z)\n(all tracked photons)",
    )

    # Detected directional slices
    det_wx = _slice_from_heatmap(
        sampled_heatmaps.water_heatmap,
        w_ext,
        (0.0, 0.0),
        n_bins,
        "x",
    )
    det_wz = _slice_from_heatmap(
        sampled_heatmaps.water_heatmap,
        w_ext,
        (0.0, 0.0),
        n_bins,
        "z",
    )
    det_sfx = _slice_from_heatmap(
        sampled_heatmaps.seafloor_heatmap,
        sf_ext,
        (0, 0),
        n_bins,
        "x",
    )
    det_sfz = _slice_from_heatmap(
        sampled_heatmaps.seafloor_heatmap,
        sf_ext,
        (0, 0),
        n_bins,
        "z",
    )
    _plot_directional_figure(
        det_wx,
        det_wz,
        det_sfx,
        det_sfz,
        "Detected Energy — Directional Slices (X vs Z)\n(photons that reached the sensor)",
    )
