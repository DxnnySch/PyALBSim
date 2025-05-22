import numbers
from utils.vector3 import vec3

class Photon:
    def __init__(self, origin: vec3, direction: vec3, velocity: numbers.Number, energy: numbers.Number, wavelength: numbers.Number):
        self.position = origin
        self.direction = direction
        self.velocity = velocity
        self.energy = energy
        self.wavelength = wavelength