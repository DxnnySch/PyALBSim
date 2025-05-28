import numbers

class Laser:
    def __init__(self):
        self.aperture = 0.1 # Receiver Telescope Radius. Unit is m.
        self.laser_divergence_angle = 1 * 1e-3 # unit in rad.
        self.laser_direction = [0.5, -1, 0]
        self.field_of_view = 10 * 1e-3 / 2 # unit in rad
        self.laser_wavelength = 0.532 # Laser wavelength. um