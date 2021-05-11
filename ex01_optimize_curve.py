'''
- Apply optimization procedure to 3D curves
- Plots 2D projects of data and the fit. For 3D visualizations, I recommend to use `vedo` (`vtkplotter`) package
'''

import time, datetime
import curve3d
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import pchip_interpolate
import random
from curve3d import calc_slist, calc_length
import matplotlib.pyplot as plt



## Define _penalty
def projection_penalty(xx, yy, xx0, yy0):  # standard deviation
    '''
    Assume that xx0, yy0 - are equidistantly spaced in terms of arc length
    '''
    slist_norm = np.linspace(0, 1, len(xx0), endpoint=True)
    slist = calc_slist(xx, yy)
    xx_interp = pchip_interpolate(slist / slist[-1], xx, slist_norm)
    yy_interp = pchip_interpolate(slist / slist[-1], yy, slist_norm)

    return (np.sum((xx_interp - xx0) ** 2 + (yy_interp - yy0) ** 2) / (len(xx0) - 1)) ** (1 / 2)


def f(params, r0, xx0, yy0, xx20, zz0):  # Minimization target
    '''
    :param params: [ds, theta_1, psi_1,..]
    :param ds: float
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

    xy_penalty = projection_penalty(xx, yy, xx0, yy0)
    xz_penalty = projection_penalty(xx, zz, xx20, zz0)
    # angle_penalty =
    # TODO: normalize by projection length?
    penalty = xy_penalty + xz_penalty
    return penalty


N = 40  # Number of data points
M = 40  # Number of fit points

### Define projections
## Line
ss0 = np.linspace(0, 1, N, endpoint=True)
xx0 = 10 * ss0
yy0 = 5 * ss0
xx20 = 10 * ss0
zz0 = 0 * ss0
r0 = (0, 0, 0)

# # Spiral
ss0 = np.linspace(0, 1, N, endpoint=True)
xx0 = 10 * ss0
yy0 = np.sin(10 * ss0)
xx20 = 10 * ss0
zz0 = np.cos(10 * ss0)
r0 = (0, 0, 1)

## Example 5.1 - Spiral: different projection
ss0 = np.linspace(0, 1, N, endpoint=True)
xx0 = np.cos(10 * ss0)
yy0 = np.sin(10 * ss0)
xx20 = np.cos(10 * ss0)
zz0 = 10 * ss0
r0 = (1, 0, 0)

## Example 6.1 -  Elleptic spiral # L = 57.8840
# ss0 = sp.linspace(0, 1, N, endpoint=True)
# xx0 = 10 * ss0
# yy0 = sp.sin(10 * ss0)
# xx20 = 10 * ss0
# zz0 = 9 * sp.cos(10 * ss0)
# r0 = (0, 0, 9)


## Example 6.2 -  Elleptic spiral # L = 167.4400
# ss0 = sp.linspace(0, 1, N, endpoint=True)
# xx0 = 10 * ss0
# yy0 = sp.sin(10 * ss0)
# xx20 = 10 * ss0
# zz0 = 27 * sp.cos(10 * ss0)
# r0 = (0, 0, 27)

# Make the points equidistantly spaced in terms of projections arc length
slist = calc_slist(xx0, yy0)
slist2 = calc_slist(xx20, zz0)
slist_norm = np.linspace(0, 1, N, endpoint=True)

# Interpolate to the equidistant spaicing in terms of projections arc length
xx1 = pchip_interpolate(slist / slist[-1], xx0, slist_norm)
yy1 = pchip_interpolate(slist / slist[-1], yy0, slist_norm)
xx21 = pchip_interpolate(slist2 / slist2[-1], xx20, slist_norm)
zz1 = pchip_interpolate(slist2 / slist2[-1], zz0, slist_norm)

# ## Test penalty
# # OK: = 0 on the same curve (line)
# # OK: = 0 on the same curve, but different parameterization
# # OK: offshift y -> only changed y penalty
# ss = sp.linspace(0, 1, 2 *N, endpoint=True) ** 2
# xx = 10 * ss
# yy = 5 * ss + 1
# zz = 30 * ss
#
# xy_penalty = projection_penalty(xx, yy, xx0, yy0)
# xz_penalty = projection_penalty(xx, zz, xx20, zz0)
#
# print(xy_penalty, xz_penalty)


## Minimize
random.seed(11)
k = 0.5
L_true = calc_length(xx0, yy0, zz0)

## Initialize with a line + noise
# ds0 = L_true / M * (1 + k * 2 * (random.random() - 0.5))
# params0 = [ds0]
# for n in range(M):
#     params0.append(sp.arctan(1 / 2) + k * 2 * (random.random() - 0.5)) # theta
#     params0.append(0 + k * 2 * (random.random() - 0.5)) # psi

## Initialize with initial curve + noise
rr0 = np.array((xx0, yy0, zz0)).T
_, ds0, thetas0, psis0 = curve3d.reconstruct_angles(rr0, M - 1)
params0 = [ds0 * (1 + k * 2 * (random.random() - 0.5))]
for theta, psi in zip(thetas0, psis0):
    params0.append(theta + k * 2 * (random.random() - 0.5)) # theta
    params0.append(psi + k * 2 * (random.random() - 0.5)) # psi


rr_init = curve3d.build_curve(r0, ds0, params0[1::2], params0[2::2])
xx_init, yy_init, zz_init = rr_init.T

print("Initial parameters:")
print(params0)

method = None  # "Powell"  #

start = time.time()
print("Optimization in progress...")
res = opt.minimize(f, params0, args=(r0, xx1, yy1, xx21, zz1), tol=10 ** -4, options={'disp': True}, method=method)
time_spent = time.time() - start

print("Done. Time spent:", datetime.timedelta(seconds=time_spent))

ds = res.x[0]
thetas = res.x[1::2]
psis = res.x[2::2]

rr = curve3d.build_curve(r0, ds, thetas, psis)
xx, yy, zz = rr.T

print(ds)
print(thetas)
print(psis)

# Show projections
plt.gca().set_aspect(1, 'datalim')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(xx0, yy0, c='black', lw=1)
plt.plot(xx, yy, 'o', c='orange', markersize=2)
plt.show()

plt.gca().set_aspect(1, 'datalim')
plt.xlabel('x2')
plt.ylabel('z')
plt.plot(xx20, zz0, c='black', lw=1)
plt.plot(xx, zz, 'o', c='orange', markersize=2)
plt.show()

