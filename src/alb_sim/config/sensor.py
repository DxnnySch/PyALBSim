from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the receiver telescope and waveform digitizer."""

    aperture_radius: float = field(
        default=0.1,
        metadata={
            "unit": "m",
            "description": "Receiver telescope radius",
        },
    )
    field_of_view: float = field(
        default=5e-3,
        metadata={
            "unit": "rad",
            "description": "Half-angle of field of view",
        },
    )
    sample_rate: int = field(
        default=int(2e9),  # 0.5 nanoseconds
        metadata={
            "unit": "Hz",
            "description": "Temporal resolution of simulated waveform",
        },
    )

    def __post_init__(self):
        """Validate that sensor parameters are within reasonable bounds."""
        if self.aperture_radius < 0:
            raise ValueError("Aperture radius must be positive")
