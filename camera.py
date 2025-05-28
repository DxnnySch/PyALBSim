class Camera:
    def __init__(self):
        self.flying_height = 300; # Flying height. m
        self.water_depth = 15; # depth. m
        self.distance_seafloor_flying_height = self.flying_height + self.water_depth; # Aircraft altitude plus seabed distance. m
        self.sample_rate = 500_000_000 # 1/s
        # R=FH*tan(sita/2);  % laser beam spot radius on the sea surface. m