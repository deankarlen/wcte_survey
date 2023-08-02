class Distance:

    def __init__(self, point1, point2, dist, err=0.002):
        self.point1 = point1
        self.point2 = point2
        self.dist = dist
        self.err = err # perpendicular to surface leads to small errors

    def est_dist(self):
        return self.point1.dist_to(self.point2)

    def deviation(self):
        return self.dist - self.est_dist()