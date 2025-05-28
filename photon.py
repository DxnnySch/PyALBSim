import numbers
from utils.vector3 import vec3
import utils.numpy_vector as np_vec

class Photon:
    def __init__(self, origin: np_vec.Vector, direction: np_vec.Vector, velocity: numbers.Number, energy: numbers.Number, wavelength: numbers.Number):
        self.position = origin
        self.direction = direction
        self.velocity = velocity
        self.energy = energy
        self.wavelength = wavelength