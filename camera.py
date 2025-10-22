class Camera:
    def __init__(self, flying_height = 20, water_depth = 2, sample_rate = 5_000_000_000):
        self.flying_height = flying_height # Flying height. m
        self.water_depth = water_depth # depth. m
        self.distance_seafloor_flying_height = self.flying_height + self.water_depth # Aircraft altitude plus seabed distance. m
        self.sample_rate = sample_rate # 1/s
        # R=FH*tan(sita/2);  % laser beam spot radius on the sea surface. m