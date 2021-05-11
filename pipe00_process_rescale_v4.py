import time, datetime
import os
import curve3d
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import pchip_interpolate


## Define _penalty
def projection_penalty(xx, yy, xx0, yy0, dxx0, dyy0, points_weight, length3d):  # standard deviation
    '''
    Assume that xx0, yy0 - are equidistantly spaced in terms of arc length
    '''
    N = len(xx0)
    slist_norm = np.linspace(0, 1, N, endpoint=True)
    slist = curve3d.calc_slist(xx, yy)

    xx_interp = pchip_interpolate(slist / slist[-1], xx, slist_norm)
    yy_interp = pchip_interpolate(slist / slist[-1], yy, slist_norm)
    points_penalty = points_weight * np.sum((xx_interp - xx0) ** 2 + (yy_interp - yy0) ** 2) / N * length3d ** -2

    ## Add penalty for tangents deviation # works as penalty on derivative deviation (see Sobolev space)
    dxx_interp = np.diff(xx_interp)
    dyy_interp = np.diff(yy_interp)
    ds = slist[-1] / (N - 1)  # To normalize; ds in 2D
    tangent_penalty = np.sum((dxx_interp - dxx0) ** 2 + (dyy_interp - dyy0) ** 2) * ds ** -2 / (N - 1)

    return (points_penalty + tangent_penalty)


def tangent_crosses(thetas, psis):
    from itertools import islice
    crosses = []
    for (theta, psi), (theta_next, psi_next) in zip(zip(thetas, psis), islice(zip(thetas, psis), 1, None)):
        cross = curve3d.calc_cross_from_angles(1, theta, psi, 1, theta_next, psi_next)
        crosses.append(cross)

    return np.array(crosses)


def smoothness_penalty(thetas, psis):
    '''
    Cross product of adjecent tangent vectors: Penalty for the magnitude of
    t_(i-1) x t_i - t_i x t_(i+1)
    '''
    crosses = tangent_crosses(thetas, psis)
    crosses_diff = np.diff(crosses, axis=0)

    return np.sum(crosses_diff ** 2) / len(crosses_diff)  # dimensionless; sum of squared differences


def f(params, r0, xx0, yy0, xx20, zz0, dxx0, dyy0, dxx20, dzz0, points_weight, smooth_weight,
      xy_weight_fraction, info):  # Minimization target
    '''
    :param params: [ds, theta_1, psi_1,..]
    :param points_weight: [1]
    :param smooth_weight: [1]
    :param xy_weight_fraction: from 0 to 1; 0.5 for equal treatment of xy and xz; best - set dependent on length ratio
    :param info: dict with fields 'print': True or False; 'num_evals' - set to zero
    :return:
    '''
    if len(params) % 2 == 1:
        ds = params[0]
        thetas = params[1::2]
        psis = params[2::2]
    else:
        raise ValueError

    rs = curve3d.build_curve(r0, ds, thetas, psis)  # reconstructed curve
    xx, yy, zz = rs.T
    length = ds * len(thetas)  # Use it to normalize projection penalties
    xy_penalty = projection_penalty(xx, yy, xx0, yy0, dxx0, dyy0, points_weight, length)
    xz_penalty = projection_penalty(xx, zz, xx20, zz0, dxx20, dzz0, points_weight, length)
    smooth_penalty = smooth_weight * smoothness_penalty(thetas, psis)
    penalty = xy_weight_fraction * xy_penalty + (1 - xy_weight_fraction) * xz_penalty + smooth_penalty

    ## Print info
    if info['print'] == True:
        info['num_evals'] += 1
        if info['num_evals'] == 1 or info['num_evals'] % 5000 == 0:
            print("{:<10}{}".format('num_evals', 'penalty'))
        if info['num_evals'] == 1 or info['num_evals'] % 500 == 0:
            print("{:<10}{:.3g}".format(info['num_evals'], penalty))
    return penalty


N = 60  # Number of data points
M = 60  # Number of fit points

input_folder = 'data/rescale'
init_data_folder = None  # Choose a folder with initial guess (`None` is also possible)
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))  # [13] # , 35, 39#

eps = 10 ** -3
ds_eps = 10 ** -3
points_weight = 100
smooth_weight = 40
output_folder = 'res/pipe00'
output_folder_angles = os.path.join(output_folder, 'angles')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_angles, exist_ok=True)

x = [] # will be a list of arrays: 1st dimension = phase/time; 2nd dimension = position along the curve
y = []
x2 = []
z = []

# Lists for original data; For visualization
x0 = [] # 1st dimension = phase/time; 2nd dimension = position along the curve
x20 = []
y0 = []
z0 = []

for iframe_raw in iframes_raw:
    # Load & save original data
    print("Frame {}".format(iframe_raw))
    xname = os.path.join(input_folder, 'x0-{}.dat'.format(iframe_raw))
    yname = os.path.join(input_folder, 'y0-{}.dat'.format(iframe_raw))
    x2name = os.path.join(input_folder, 'x20-{}.dat'.format(iframe_raw))
    zname = os.path.join(input_folder, 'z0-{}.dat'.format(iframe_raw))

    xx0 = np.loadtxt(xname)
    yy0 = np.loadtxt(yname)
    xx20 = np.loadtxt(x2name)
    zz0 = np.loadtxt(zname)

    x0.append(xx0)
    y0.append(yy0)
    x20.append(xx20)
    z0.append(zz0)

    # Make the points equidistantly spaced in terms of projections arc length
    slist = curve3d.calc_slist(xx0, yy0)
    slist2 = curve3d.calc_slist(xx20, zz0)
    slist_norm = np.linspace(0, 1, N, endpoint=True)

    # Interpolate to the equidistant spacing in terms of projections arc length
    xx1 = pchip_interpolate(slist / slist[-1], xx0, slist_norm)
    yy1 = pchip_interpolate(slist / slist[-1], yy0, slist_norm)
    xx21 = pchip_interpolate(slist2 / slist2[-1], xx20, slist_norm)
    zz1 = pchip_interpolate(slist2 / slist2[-1], zz0, slist_norm)

    # Repeat re-interpolation until it's precise
    for k in range(100):
        slist = curve3d.calc_slist(xx1, yy1)
        dslist = np.diff(slist)
        ds_error = (np.amax(dslist) - np.amin(dslist)) / np.mean(dslist) / 2
        if ds_error > ds_eps:
            xx1 = pchip_interpolate(slist / slist[-1], xx1, slist_norm)
            yy1 = pchip_interpolate(slist / slist[-1], yy1, slist_norm)
        else:
            break

    for k in range(100):
        slist2 = curve3d.calc_slist(xx21, zz1)
        dslist2 = np.diff(slist2)
        ds_error = (np.amax(dslist2) - np.amin(dslist2)) / np.mean(dslist2) / 2
        if ds_error > ds_eps:
            xx21 = pchip_interpolate(slist2 / slist2[-1], xx21, slist_norm)
            zz1 = pchip_interpolate(slist2 / slist2[-1], zz1, slist_norm)
        else:
            break

    dxx1 = np.diff(xx1)
    dyy1 = np.diff(yy1)
    dxx21 = np.diff(xx21)
    dzz1 = np.diff(zz1)

    # Set projections weight ratio according to projections length; shorter - less length.
    xy_weight_fraction = slist[-1] / (slist[-1] + slist2[-1])
    ## Get initial guess
    r0 = (0, 0, 0)
    if init_data_folder is not None:
        xname = os.path.join(init_data_folder, 'x-{}.dat'.format(iframe_raw))
        yname = os.path.join(init_data_folder, 'y-{}.dat'.format(iframe_raw))
        zname = os.path.join(init_data_folder, 'z-{}.dat'.format(iframe_raw))

        xx = np.loadtxt(xname)
        yy = np.loadtxt(yname)
        zz = np.loadtxt(zname)
        rr_init = np.array([xx, yy, zz]).T
    else:
        rr_init = np.array([xx1, yy1, zz1]).T
        xx_init, yy_init, zz_init = rr_init.T  # For visualization

    _, ds_init, thetas_init, psis_init = curve3d.reconstruct_angles(rr_init, M - 1)

    params0 = [ds_init]
    for theta, psi in zip(thetas_init, psis_init):
        params0.append(theta)
        params0.append(psi)

    method = None  # Default BFGS - works fine; Alternative: "Powell", "Nelder-Mead"

    start = time.time()
    res = opt.minimize(f, params0,
                       args=(r0, xx1, yy1, xx21, zz1, dxx1, dyy1, dxx21, dzz1, points_weight, smooth_weight,
                             xy_weight_fraction, {'num_evals': 0, 'print': True}),
                       tol=eps, options={'disp': True}, method=method)
    time_spent = time.time() - start

    print("Time spent:", datetime.timedelta(seconds=time_spent))

    ds = res.x[0]
    thetas = res.x[1::2]
    psis = res.x[2::2]

    rr = curve3d.build_curve(r0, ds, thetas, psis)
    xx, yy, zz = rr.T

    x.append(xx)
    y.append(yy)
    z.append(zz)
    np.savetxt(os.path.join(output_folder, 'x-{}.dat'.format(iframe_raw)), xx)
    np.savetxt(os.path.join(output_folder, 'y-{}.dat'.format(iframe_raw)), yy)
    np.savetxt(os.path.join(output_folder, 'z-{}.dat'.format(iframe_raw)), zz)
    np.savetxt(os.path.join(output_folder_angles, 'ds-{}.dat'.format(iframe_raw)), [ds])
    np.savetxt(os.path.join(output_folder_angles, 'theta-{}.dat'.format(iframe_raw)), thetas)
    np.savetxt(os.path.join(output_folder_angles, 'psi-{}.dat'.format(iframe_raw)), psis)

