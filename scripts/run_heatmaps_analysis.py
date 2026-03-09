import math
import numpy as np

from alb_sim.config.run import RunConfig
from alb_sim.config.heatmap import HeatmapConfig
from alb_sim.config.sea_floor import SeaFloorConfig
from alb_sim.config.sensor import SensorConfig
from alb_sim.config.water import TurbidityLayerConfig, WaterConfig
from alb_sim.config.simulation import SimulationConfig
from alb_sim.execution.parallel import run_parallel
from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.physics.models.simulation import SimulationModel
from alb_sim.plotting.photon_map_interaction_interactive_heatmap import interactive_photon_map_interaction_heatmap
from alb_sim.plotting.plot_radial_histogram import plot_radial_energy_histograms
from alb_sim.plotting.plot_scatter_radius_heatmap import plot_scatter_radius_heatmap
from alb_sim.plotting.plot_stacked_waveform import plot_stacked_waveform
from alb_sim.plotting.precomputed_interactive_heatmap import interactive_precomputed_heatmap

simulation_config = SimulationConfig(
    water=WaterConfig(
        layers=(
            TurbidityLayerConfig(
                height=7, absorption_coefficient=0.169, scattering_coefficient=1.2
            ),
        )
    ),
    sensor=SensorConfig(sample_rate=2e9, field_of_view=40e-3),
    sea_floor=SeaFloorConfig(albedo=0.1),
    heatmap=HeatmapConfig(
        enabled=True,
        bins=200,
        water_extent=0.16,  # metres, ~3x laser spot radius
        seafloor_extent=2.5,  # metres, larger due to scattering
    ),
)
run_config = RunConfig(
    photons_per_batch_forward=5000,
    photons_per_batch_backward=1250,
    batches_forward=(8) * 1,
    batches_backward=(8) * 1,
    processes=8,
)

simulation_model = SimulationModel(simulation_config)

waveform, photon_maps_data, sampled_heatmaps = run_parallel(simulation_config, run_config)

# ── Waveform stacked plot ──────────────────────────────────────────────────
plot_stacked_waveform(waveform)

# ── Derived geometry ───────────────────────────────────────────────────────

water_intersection = simulation_model.water_surface_intersection
sea_floor_intersection = simulation_model.sea_floor_intersection
print(f"Laser-water intersection: {water_intersection}")

water_center_xz = (water_intersection[0], water_intersection[2])
seafloor_center_xz = (sea_floor_intersection[0], sea_floor_intersection[2])

laser_spot_radius = (
    simulation_config.scene.flying_height
    * math.tan(simulation_config.laser.divergence_angle)
)
print(f"Laser spot radius: {laser_spot_radius:.4f} m")
print(f"Seafloor laser footprint center: {seafloor_center_xz}")

water_depth = -1 * (simulation_model.seafloor_y - simulation_model.water_surface_y)
seafloor_extent = laser_spot_radius * 15 + water_depth * 0.5


# ── Interactive heatmaps (raw photon data — all outgoing photons) ──────────
# Water surface: use SURFACE_REFLECTION only (all entering photons stored
# with energy=1.0 before Fresnel attenuation, no double-counting).
# Seafloor: use BOTTOM_REFLECTION only (all seafloor-hitting photons stored
# before seafloor albedo is applied).

interactive_photon_map_interaction_heatmap(
    photon_maps_data,
    water_center_xz,
    interaction_type="water",
    initial_bin_size=laser_spot_radius / 50,
    initial_extent_radius=laser_spot_radius * 3,
    energy_weighted=False,
    photon_types=[PhotonType.SURFACE_REFLECTION],
)

interactive_photon_map_interaction_heatmap(
    photon_maps_data,
    water_center_xz,
    interaction_type="water",
    initial_bin_size=laser_spot_radius / 50,
    initial_extent_radius=laser_spot_radius * 3,
    energy_weighted=True,
    photon_types=[PhotonType.SURFACE_REFLECTION],
)

interactive_photon_map_interaction_heatmap(
    photon_maps_data,
    seafloor_center_xz,
    interaction_type="seafloor",
    initial_bin_size=seafloor_extent / 50,
    initial_extent_radius=seafloor_extent,
    energy_weighted=False,
    photon_types=[PhotonType.BOTTOM_REFLECTION],
)

interactive_photon_map_interaction_heatmap(
    photon_maps_data,
    seafloor_center_xz,
    interaction_type="seafloor",
    initial_bin_size=seafloor_extent / 50,
    initial_extent_radius=seafloor_extent,
    energy_weighted=True,
    photon_types=[PhotonType.BOTTOM_REFLECTION],
)


# ── Sampled heatmaps (detected energy) ────────────────────────────────────

if sampled_heatmaps is not None:
    print("\n" + "=" * 50)
    print("SAMPLED PHOTON HEATMAPS (Actually Detected Energy)")
    print("=" * 50)

    # ── Verification ──────────────────────────────────────────────────────
    print("\n--- VERIFICATION ---")

    waveform_total = sum(arr.sum() for arr in waveform.values())
    print(f"Total energy in waveform: {waveform_total:.6f}")
    for pt, arr in waveform.items():
        print(f"  {pt.name}: {arr.sum():.6f}")

    water_heatmap_total = sampled_heatmaps.water_heatmap.sum()
    seafloor_heatmap_total = sampled_heatmaps.seafloor_heatmap.sum()
    print(f"\nWater surface heatmap total energy: {water_heatmap_total:.6f}")
    print(f"Seafloor heatmap total energy: {seafloor_heatmap_total:.6f}")

    # The water heatmap captures photons that entered water
    # (SCATTER, BOTTOM_REFLECTION, SURFACE_TRANSMISSION_UP, SURFACE_REFLECTION).
    # The seafloor heatmap only captures photons that hit the seafloor.

    bins = sampled_heatmaps.water_heatmap.shape[0]
    water_bin_size = 2 * simulation_config.heatmap.water_extent / bins
    seafloor_bin_size = 2 * simulation_config.heatmap.seafloor_extent / bins

    print(f"\nHeatmap resolution:")
    print(f"  Water:    {bins}x{bins} bins, {water_bin_size*1000:.2f} mm/bin, ±{simulation_config.heatmap.water_extent} m")
    print(f"  Seafloor: {bins}x{bins} bins, {seafloor_bin_size*1000:.2f} mm/bin, ±{simulation_config.heatmap.seafloor_extent} m")

    water_nonzero = np.count_nonzero(sampled_heatmaps.water_heatmap)
    seafloor_nonzero = np.count_nonzero(sampled_heatmaps.seafloor_heatmap)
    total_bins = bins * bins
    print(f"\nNon-zero bins:")
    print(f"  Water:    {water_nonzero}/{total_bins} ({100*water_nonzero/total_bins:.1f}%)")
    print(f"  Seafloor: {seafloor_nonzero}/{total_bins} ({100*seafloor_nonzero/total_bins:.1f}%)")

    if water_heatmap_total > 0:
        print("\n✓ Water heatmap has accumulated energy")
    else:
        print("\n✗ Water heatmap is empty — check extent or simulation bug")

    if seafloor_heatmap_total > 0:
        print("✓ Seafloor heatmap has accumulated energy")
    else:
        print("✗ Seafloor heatmap is empty — check extent or photon reach")

    # ── Interactive precomputed heatmaps ──────────────────────────────────
    interactive_precomputed_heatmap(
        sampled_heatmaps.water_heatmap,
        simulation_config.heatmap.water_extent,
        "Water Surface: Sampled/Detected Energy",
        crosshair_xz=(0.0, 0.0),
    )

    interactive_precomputed_heatmap(
        sampled_heatmaps.seafloor_heatmap,
        simulation_config.heatmap.seafloor_extent,
        "Seafloor: Sampled/Detected Energy",
        crosshair_xz=(0.0, 0.0),
    )

    # ── Radial energy histograms ──────────────────────────────────────────
    n_backward_total = run_config.batches_backward * run_config.photons_per_batch_backward
    plot_radial_energy_histograms(
        sampled_heatmaps,
        photon_maps_data,
        heatmap_config=simulation_config.heatmap,
        water_center_xz=water_center_xz,
        seafloor_center_xz=seafloor_center_xz,
        n_backward_total=n_backward_total,
    )

    # ── Scatter radius correlation heatmap ────────────────────────────────
    if sampled_heatmaps.scatter_radius_heatmap is not None:
        plot_scatter_radius_heatmap(sampled_heatmaps.scatter_radius_heatmap, simulation_config.heatmap)


else:
    print("\nSampled heatmaps were not computed (heatmap.enabled=False)")