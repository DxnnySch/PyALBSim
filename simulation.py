from camera import Camera
from laser import Laser
from world import World


class Simulation:
    def __init__(self):
        self.number_of_photons = 1e5 # Number of photon packets.
        self.ratio = 3 # Photon packets within ?? spot radii are used for simulation. Here is 3. TODO: Rewrite
        self.photon_survival_threshold_weight = 0.0001 # epsilon
        
        self.laser_settings = Laser()
        self.camera_settings = Camera()
        self.world_settings = World(self.laser_settings)

    def simulate(self):
        pass
        # initialise photons, in batches?
        # step through photons
            # for each photon, determine location and get strategy function
            # put photon through strategy function
            # repeat
            # maybe culling?