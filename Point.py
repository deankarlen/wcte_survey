import numpy as np


class Point:

    def __init__(self, name, init, err=[0.002, 0.0001, 0.002]):
        self.name = name
        self.coord_init = init  # initial guess
        self.coord_err = err  # uncertainty on prior. Keep y err small, to work in xz plane
        self.coord_est = [init[0], init[1], init[2]]  # current estimate

    def dist_to(self, point):
        sum2 = 0.
        for coord in range(3):
            delta = self.coord_est[coord] - point.coord_est[coord]
            sum2 += delta ** 2
        return np.sqrt(sum2)
