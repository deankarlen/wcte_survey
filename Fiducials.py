from Point import Point
from Distance import Distance
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle

# find the locations of the fiducials
# Coordinate system: z is along beam direction, y is vertical
# Use as an origin: Location of Trigger 1 (near beam window) at beam centre.
# Use measurements from beam dump indication on wall.
# It is measured to be z= 10.885 m downstream and y = 1.354 m above the floor

# define fiducial points:
# The first is the reference coordinate (not to be changed in minimization)
# The second point is the reference y direction (not to be changed in minimization)

fiducial_points = {}
distances = []
masked_distances = []


def add_p(name, x, y, z, err=[4., 0.020, 4, ]):
    fiducial_points[name] = Point(name, [x, y, z], err=err)


def add_d(name1, name2, dist, err=0.01):
    distances.append(Distance(fiducial_points[name1], fiducial_points[name2], dist, err))


# beam dump
add_p('beam_dump', 0., 1.354, 10.885, err=[0.001, 0.001, 0.003])

# Initial guesses for locations of fiducials
add_p('T09.05', 4., 2.10, 10.)
add_p('T09.06', -1., 2.14, 10.)
add_p('T09.07', -2., 2.09, 8.)
add_p('T09.08', -2., 1.975, 3.)
add_p('T09.01', 1., 2.13, 1.)
add_p('T09.02', 5., 2.1, 1.)
add_p('T09.03', 6., 2.09, 3.)
add_p('T09.04', 6., 2.089, 7.)

# Measured distances:
add_d('beam_dump', 'T09.04', 6.065)  # base of device not on wall ?
add_d('beam_dump', 'T09.02', 12.012)
add_d('beam_dump', 'T09.07', 4.795)
add_d('beam_dump', 'T09.06', 1.208)
add_d('beam_dump', 'T09.05', 4.628)

add_d('T09.01', 'T09.02', 4.406)
add_d('T09.01', 'T09.04', 9.735)
add_d('T09.01', 'T09.06', 11.294)
add_d('T09.02', 'T09.03', 2.475)
add_d('T09.02', 'T09.05', 11.267)
add_d('T09.02', 'T09.08', 7.307)
add_d('T09.03', 'T09.04', 5.820)
# removed as it adds the most chi^2
# add_d('T09.03','T09.06',10.90)
add_d('T09.03', 'T09.07', 9.644)
add_d('T09.03', 'T09.08', 8.058)
add_d('T09.04', 'T09.05', 3.229)
add_d('T09.04', 'T09.06', 6.850)
add_d('T09.04', 'T09.07', 8.082)
add_d('T09.05', 'T09.06', 5.489)
add_d('T09.05', 'T09.07', 8.378)
add_d('T09.05', 'T09.08', 11.955)
add_d('T09.06', 'T09.07', 4.230)
add_d('T09.06', 'T09.08', 9.580)
add_d('T09.07', 'T09.08', 5.644)


# make a plot in the x-y plane to check

def make_plot(option):
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    for point_name in fiducial_points:
        coord_vals = fiducial_points[point_name].coord_init
        if option == 'est':
            coord_vals = fiducial_points[point_name].coord_est

        axis.plot([coord_vals[2]], [coord_vals[0]], ms=5, marker='o')
        axis.text(coord_vals[2], coord_vals[0], point_name)

    for distance in distances:
        coord_p1 = distance.point1.coord_init
        coord_p2 = distance.point2.coord_init
        if option == 'est':
            coord_p1 = distance.point1.coord_est
            coord_p2 = distance.point2.coord_est
        axis.plot([coord_p1[2], coord_p2[2]], [coord_p1[0], coord_p2[0]])
        dx = coord_p1[0] - coord_p2[0]
        dy = coord_p1[1] - coord_p2[1]
        dz = coord_p1[2] - coord_p2[2]
        angle = np.arctan2(dx, dz) * 180. / np.pi
        if 90. < angle < 270.:
            angle -= 180.
        elif -90. > angle > -270.:
            angle += 180.
        init_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
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
    axis.set_title(option + ' dist:Actual dist')


def chi2(pars):
    ip = 0
    chi2sum = 0.
    for point_name in fiducial_points:
        point = fiducial_points[point_name]
        for coord in [0, 2]:
            point.coord_est[coord] = pars[ip]
            ip += 1

    for distance in distances:
        if distance not in masked_distances:
            p1 = distance.point1
            p2 = distance.point2
            dx = p1.coord_est[0] - p2.coord_est[0]
            dy = p1.coord_est[1] - p2.coord_est[1]
            dz = p1.coord_est[2] - p2.coord_est[2]
            est_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # deviation from measured distance
            chi2sum += (est_dist - distance.dist) ** 2 / distance.err ** 2

    for point_name in fiducial_points:
        point = fiducial_points[point_name]
        # deviation from original estimate
        for coord in [0, 2]:
            chi2sum += (point.coord_est[coord] - point.coord_init[coord]) ** 2 / point.coord_err[coord] ** 2

    return chi2sum


# make_plot('est')

if 1 == 1:
    par0 = []
    for point_name in fiducial_points:
        point = fiducial_points[point_name]
        for coord in [0, 2]:
            par0.append(point.coord_init[coord])

    res = minimize(chi2, par0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 100000, 'maxfev': 100000})

    print(res.message)
    print('chi2 =', res.fun)
    print('nit, nfev =', res.nit, res.nfev)

    make_plot('est')
    plt.savefig('fiducial_distances.png')
    plt.show()

    with open('fiducial_points.p', 'wb') as f:
        pickle.dump(fiducial_points, f)

# look for largest contributors to chi**2
# remove one distance measurement at a time

for masked_distance in distances:
    masked_distances = [masked_distance]
    par0 = []
    for point_name in fiducial_points:
        point = fiducial_points[point_name]
        for coord in [0, 2]:
            par0.append(point.coord_init[coord])

    res = minimize(chi2, par0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 100000, 'maxfev': 100000})

    name1 = masked_distance.point1.name
    name2 = masked_distance.point2.name
    print('Masked distance between:', name1, name2, res.message)
    print('chi2 =', res.fun)
    print('nit, nfev =', res.nit, res.nfev)
