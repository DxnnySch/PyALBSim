from dataclasses import dataclass, field


@dataclass(frozen=True)
class LaserConfig:
    divergence_angle: float = field(
        default=1e-3,
        metadata={
            "unit": "rad",
            "description": "Laser beam divergence half-angle",
        },
    )
    wavelength: float = field(
        default=0.532,
        metadata={
            "unit": "µm",
            "description": "Laser wavelength in vacuum",
        },
    )
    nadir_angle: float = field(
        default=15.0,
        metadata={
            "unit": "rad",
            "description": "Nadir angle from -y axis",
        },
    )
    azimuth_angle: float = field(
        default=0.0,
        metadata={
            "unit": "rad",
            "description": "Azimuth angle around y axis (0 = +x)",
        },
    )
    pulse_fwhm: float = field(
        default=8.3e-9,
        metadata={
            "unit": "s",
            "description": "Full width at half maximum of laser pulse",
        },
    )

    def __post_init__(self):
        if self.nadir_angle >= 90:
            raise ValueError("Nadir angle must be below 90 degrees")
