from Point import Point
from Distance import Distance
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle

# Using the locations of the fiducials, find coordinates of
# detectors
# Coordinate system: z is along beam direction, y is vertical
# Use as an origin: Location of Trigger 1 (near beam window) at beam centre.

with open('fiducial_points.p', 'rb') as f:
    fiducial_points = pickle.load(f)

detector_points = {}
distances_by_detector = {}


def add_p(name, x, y, z, err=[0.002, 0.002, 0.5]):
    detector_points[name] = Point(name, [x, y, z], err=err)


def add_d(det_name, fid_name, dist, err=0.01):
    if det_name not in distances_by_detector:
        distances_by_detector[det_name] = []
    distances_by_detector[det_name].append(Distance(detector_points[det_name], fiducial_points[fid_name], dist, err))


# Initial guesses for locations of detectors
add_p('pb-glass', 0., 1.35 + 0.2, 7.)  # half height of pb glass
add_p('HD14.A', 0.35, 1.55, 7., err=[0.05, 0.001, 1.0])
add_p('HD1.A', 0.8, 1.55, 7., err=[0.4, 0.001, 1.0])
add_p('T2', 0.0, 1.35, 5., err=[0.001, 0.005, 2.0])
add_p('magnet', 0., 1.35 + 0.4, 6., err=[0.005, 0.001, 2.0])
add_p('T0', 0., 1.35 + 0.2, 0., err=[0.001, 0.01, 0.1])

# Distances to fiducials
#add_d('pb-glass', 'beam_dump', 3.262 + 0.380)  # width of block
add_d('pb-glass', 'T09.01', 8.046)
add_d('pb-glass', 'T09.02', 9.148)
add_d('pb-glass', 'T09.03', 7.950)
add_d('pb-glass', 'T09.04', 5.457)
add_d('pb-glass', 'T09.05', 5.906)
add_d('pb-glass', 'T09.06', 3.428)
add_d('pb-glass', 'T09.07', 2.780)
add_d('pb-glass', 'T09.08', 6.658)

add_d('HD14.A', 'T09.01', 7.022)
add_d('HD14.A', 'T09.02', 8.118)
add_d('HD14.A', 'T09.03', 7.990)
add_d('HD14.A', 'T09.04', 5.294)
add_d('HD14.A', 'T09.05', 6.342)
add_d('HD14.A', 'T09.06', 4.482)
add_d('HD14.A', 'T09.07', 3.086)
add_d('HD14.A', 'T09.08', 5.864)

add_d('HD1.A', 'T09.03', 6.735)
add_d('HD1.A', 'T09.04', 4.774)
add_d('HD1.A', 'T09.05', 5.881)
add_d('HD1.A', 'T09.06', 4.495)
add_d('HD1.A', 'T09.07', 3.572)
add_d('HD1.A', 'T09.08', 5.861)

add_d('T2', 'T09.04', 6.845 + 0.002)
add_d('T2', 'T09.07', 4.053 + 0.002)
add_d('T2', 'T09.08', 3.435 + 0.002)
#add_d('T2', 'beam_dump', 6.949 - 0.004)

add_d('magnet', 'T09.01', 4.933 + 0.38 / 2.)
add_d('magnet', 'T09.04', 6.481 + 0.38 / 2.)
add_d('magnet', 'T09.05', 8.167 + 0.38 / 2.)
add_d('magnet', 'T09.06', 6.390 + 0.38 / 2.)
add_d('magnet', 'T09.07', 3.612 + 0.38 / 2.)
add_d('magnet', 'T09.08', 3.828 + 0.38 / 2.)
#add_d('magnet', 'beam_dump', 6.376)

add_d('T0', 'T09.03', 6.362)
add_d('T0', 'T09.04', 9.788)
add_d('T0', 'T09.06', 10.866)
add_d('T0', 'T09.07', 7.449)
add_d('T0', 'T09.08', 2.558)
#add_d('T0', 'beam_dump', 10.885)

detector_name = None


def make_plot():
    if detector_name in distances_by_detector:
        distances = distances_by_detector[detector_name]
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))
        for distance in distances:
            fiducial_point = distance.point2
            point_name = fiducial_point.name
            coord_vals = fiducial_point.coord_est

            axis.plot([coord_vals[2]], [coord_vals[0]], ms=5, marker='o')
            axis.text(coord_vals[2], coord_vals[0], point_name)

        detector_point = detector_points[detector_name]
        coord_vals = detector_point.coord_est
        axis.plot([coord_vals[2]], [coord_vals[0]], ms=5, marker='o')
        axis.text(coord_vals[2], coord_vals[0], detector_name)

        for distance in distances:
            coord_p1 = distance.point1.coord_est
            coord_p2 = distance.point2.coord_est
            axis.plot([coord_p1[2], coord_p2[2]], [coord_p1[0], coord_p2[0]])
            dx = coord_p1[0] - coord_p2[0]
            dz = coord_p1[2] - coord_p2[2]
            angle = np.arctan2(dx, dz) * 180. / np.pi
            if 90. < angle < 270.:
                angle -= 180.
            elif -90. > angle > -270.:
                angle += 180.
            init_dist = np.sqrt(dx ** 2 + dz ** 2)
            actual_dist = distance.dist
            info = f'{init_dist:.3f}:{actual_dist:.3f}'
            midpoint_x = (coord_p1[0] + coord_p2[0]) / 2.
            midpoint_z = (coord_p1[2] + coord_p2[2]) / 2.
            axis.text(midpoint_z, midpoint_x, info,
                      horizontalalignment='center', verticalalignment='center', rotation=angle)

        axis.set_ylim(-3., 8.)
        axis.set_xlim(-1., 13.)
        axis.invert_xaxis()
        axis.invert_yaxis()
        axis.set_xlabel('z (m)')
        axis.set_ylabel('x (m)')
        axis.set_title(detector_name + '  Est dist:Actual dist')


def chi2(pars):
    chi2sum = 0.
    detector_points[detector_name].coord_est[0] = pars[0]
    detector_points[detector_name].coord_est[1] = pars[1]
    detector_points[detector_name].coord_est[2] = pars[2]

    distances = distances_by_detector[detector_name]
    for distance in distances:
        deviation = distance.deviation()
        # deviation from measured distance
        chi2sum += (deviation) ** 2 / distance.err ** 2

    point = detector_points[detector_name]
    # deviation from original estimate for detector coordinate
    for coord in [0, 1, 2]:
        chi2sum += (point.coord_est[coord] - point.coord_init[coord]) ** 2 / point.coord_err[coord] ** 2

    return chi2sum


estimated_coordinates = {}
for point_name in detector_points:
    point = detector_points[point_name]
    detector_name = point_name
    par0 = []
    for coord in [0, 1, 2]:
        par0.append(point.coord_init[coord])

    res = minimize(chi2, par0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 100000, 'maxfev': 100000})

    print(res.message)
    print('chi2 =', res.fun)
    print('nit, nfev =', res.nit, res.nfev)
    print(detector_points[detector_name].coord_est)

    estimated_coordinates[detector_name] = detector_points[detector_name].coord_est

    make_plot()
    plt.savefig(detector_name+'_distances.png')
    plt.show()

print()
print('Summary')
for detector_name in estimated_coordinates:
    print(detector_name,estimated_coordinates[detector_name])
