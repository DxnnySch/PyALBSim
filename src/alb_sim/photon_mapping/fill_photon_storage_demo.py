import numpy as np

from alb_sim.photon_mapping.photon_type import PhotonType


def fill_photon_storage_demo(
    storage,
    *,
    n_surface: int = 50_000,
    n_scatter: int = 30_000,
    n_bottom: int = 20_000,
    seed: int = 1234,
) -> None:
    """
    Fill a PhotonStorage with synthetic demo photons.

    This is intended for visualisation and debugging, not physical correctness.

    Parameters
    ----------
    storage
        PhotonStorage-like object to populate.
    n_surface, n_scatter, n_bottom : int, optional
        Number of synthetic photons to generate for each interaction type.
    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # SURFACE photons (clustered near z = 0, downward)
    # --------------------------------------------------
    pos = np.empty((n_surface, 3), dtype=np.float32)
    pos[:, 0] = rng.uniform(-50, 50, n_surface)
    pos[:, 1] = rng.uniform(-50, 50, n_surface)
    pos[:, 2] = rng.normal(loc=0.0, scale=0.05, size=n_surface)

    dirs = np.empty((n_surface, 3), dtype=np.float32)
    dirs[:, 0:2] = rng.normal(0.0, 0.1, size=(n_surface, 2))
    dirs[:, 2] = -np.abs(rng.normal(1.0, 0.05, size=n_surface))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    storage.positions[PhotonType.SURFACE_REFLECTION].append(pos)
    storage.directions[PhotonType.SURFACE_REFLECTION].append(dirs)
    storage.energies[PhotonType.SURFACE_REFLECTION].append(
        rng.uniform(0.5, 1.0, n_surface).astype(np.float32)
    )
    storage.times[PhotonType.SURFACE_REFLECTION].append(
        rng.uniform(0.0, 5.0, n_surface).astype(np.float32)
    )
    storage.already_reflected[PhotonType.SURFACE_REFLECTION].append(
        np.ones(n_surface, dtype=bool)
    )

    # --------------------------------------------------
    # SCATTER photons (volumetric, roughly isotropic)
    # --------------------------------------------------
    pos = rng.uniform(
        [-50, -50, -20],
        [50, 50, 0],
        size=(n_scatter, 3),
    ).astype(np.float32)

    dirs = rng.normal(0.0, 1.0, size=(n_scatter, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    storage.positions[PhotonType.SCATTER].append(pos)
    storage.directions[PhotonType.SCATTER].append(dirs)
    storage.energies[PhotonType.SCATTER].append(
        rng.exponential(scale=0.3, size=n_scatter).astype(np.float32)
    )
    storage.times[PhotonType.SCATTER].append(
        rng.uniform(0.0, 20.0, n_scatter).astype(np.float32)
    )
    storage.already_reflected[PhotonType.SCATTER].append(rng.random(n_scatter) > 0.5)

    # --------------------------------------------------
    # BOTTOM photons (clustered near z = -20, upward)
    # --------------------------------------------------
    pos = np.empty((n_bottom, 3), dtype=np.float32)
    pos[:, 0] = rng.uniform(-50, 50, n_bottom)
    pos[:, 1] = rng.uniform(-50, 50, n_bottom)
    pos[:, 2] = rng.normal(loc=-20.0, scale=0.2, size=n_bottom)

    dirs = np.empty((n_bottom, 3), dtype=np.float32)
    dirs[:, 0:2] = rng.normal(0.0, 0.2, size=(n_bottom, 2))
    dirs[:, 2] = np.abs(rng.normal(1.0, 0.1, size=n_bottom))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    storage.positions[PhotonType.BOTTOM_REFLECTION].append(pos)
    storage.directions[PhotonType.BOTTOM_REFLECTION].append(dirs)
    storage.energies[PhotonType.BOTTOM_REFLECTION].append(
        rng.uniform(0.2, 0.8, n_bottom).astype(np.float32)
    )
    storage.times[PhotonType.BOTTOM_REFLECTION].append(
        rng.uniform(5.0, 25.0, n_bottom).astype(np.float32)
    )
    storage.already_reflected[PhotonType.BOTTOM_REFLECTION].append(
        np.ones(n_bottom, dtype=bool)
    )
