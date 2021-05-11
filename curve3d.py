'''
Describe 3D curve by segments;

each segment described by theta and psi (angle in yz plane):
x_{k+1} = x_k + ds * |sin(theta)| * cos(psi)
y_{k+1} = y_k + ds * |sin(theta)| * sin(psi)
z_{k+1} = z_k + ds * cos(theta)

Taking only absolute values of sine allows us to consider theta from 0 to 2 pi.

NB: It's safe to take psi % (2* pi), but theta % pi is not correct - for that there is a function defined below
'''
import scipy as sp
from math import cos, sin, acos, atan2, pi


def calc_slist(xx, yy=None, zz=None):
    if yy is None:
        yy = sp.zeros_like(xx)
    if zz is None:
        zz = sp.zeros_like(xx)
    slist = sp.zeros_like(xx, dtype=sp.float64)
    sp.cumsum((sp.diff(xx) ** 2 + sp.diff(yy) ** 2 + sp.diff(zz) ** 2) ** (1 / 2), out=slist[1:])
    return slist


def calc_length(xx, yy, zz):
    return sp.sum((sp.diff(xx) ** 2 + sp.diff(yy) ** 2 + sp.diff(zz) ** 2) ** (1 / 2))


def get_ds_angles(dx, dy, dz):
    ds = (dx ** 2 + dy ** 2 + dz ** 2) ** (1 / 2)
    theta = acos(dz / ds)  # From  0 to pi
    psi = atan2(dy, dx) % (2 * pi)  # From 0 to 2pi
    return ds, theta, psi


def get_dr(ds, theta, psi):
    dx = ds * sin(theta) * cos(psi)
    dy = ds * sin(theta) * sin(psi)
    dz = ds * cos(theta)

    return dx, dy, dz


def build_curve(r0, ds, thetas, psis):
    '''
    :param r0: start position
    :param thetas:
    :param psis:
    :param ds: constant size of segment
    :return: rr: sp.array([x0,y0,z0],...])
    '''
    if len(thetas) != len(psis):
        raise ValueError

    r = sp.array(r0, dtype=float)
    rr = [r]

    for theta, psi in zip(thetas, psis):
        dr = get_dr(ds, theta, psi)
        r = r + dr
        rr.append(r)
    rr = sp.array(rr)
    return rr


def reconstruct_angles(rr, num_segments):
    '''
    '''
    from scipy.interpolate import pchip_interpolate
    r0 = rr[0]
    xx, yy, zz = rr.T
    slist = calc_slist(xx, yy, zz)
    slist_norm = sp.linspace(0, 1, num_segments + 1, endpoint=True)
    # Interpolate to equal arc length spacing
    xx_interp = pchip_interpolate(slist / slist[-1], xx, slist_norm)
    yy_interp = pchip_interpolate(slist / slist[-1], yy, slist_norm)
    zz_interp = pchip_interpolate(slist / slist[-1], zz, slist_norm)
    thetas = []
    psis = []
    for dx, dy, dz in zip(sp.diff(xx_interp), sp.diff(yy_interp), sp.diff(zz_interp)):
        _, theta, psi = get_ds_angles(dx, dy, dz)
        thetas.append(theta)
        psis.append(psi)

    ds = slist[-1] / num_segments

    return r0, ds, thetas, psis


def _theta_to_pi_interval(theta):
    theta = theta % (2 * pi)
    if theta >= pi:
        theta = 2 * pi - theta # mirror

    return theta

thetas_to_pi_interval = sp.vectorize(_theta_to_pi_interval)


## Pairwise operations
def calc_cross_from_angles(ds1, theta1, psi1, ds2, theta2, psi2):
    '''
    Return euclidian form of dr1 x dr2, given angles defining dr1 and dr2.
    '''

    dr_cross = sp.array([cos(theta2) * sin(psi1) * sin(theta1) - cos(theta1) * sin(psi2) * sin(theta2),
                         -(cos(psi1) * cos(theta2) * sin(theta1)) + cos(psi2) * cos(theta1) * sin(theta2),
                         -(sin(psi1 - psi2) * sin(theta1) * sin(theta2))])

    return ds1 * ds2 * dr_cross


# # Test reconstruct angles
# if __name__ == '__main__':
#     # OK: get_ds_angles
#     # OK: reconstructed angles on spiral; N=M
#     # NOT OK: reconstructed angles, when interpolated - hypothesis - interpolation is not precise; redo interpolation
#
#     import numpy.random as rnd
#     import quick
#
#     dx, dy, dz = 3, 0, 0
#     # print(get_ds_angles(dx,dy,dz))
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     dx, dy, dz = 0, 5, 0
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     dx, dy, dz = 0, 0, 6
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     dx, dy, dz = 0, 0, -6
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     dx, dy, dz = 1, 1, 1
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     dx, dy, dz = -1, 10, 121
#     res = get_ds_angles(dx, dy, dz)
#     print(get_dr(*res))
#
#     # # Spiral
#     N = 50
#     M = 40
#     r0 = (1, 0, 0)
#     ds = 1
#     thetas = [sp.arccos(0.5) for _ in range(N)]
#     psis = [2 * sp.pi / N * k for k in range(N)]
#     rr = build_curve(r0, ds, thetas, psis)
#
#     r0_rec, ds_rec, thetas_rec, psis_rec = reconstruct_angles(rr, M)
#
#     with quick.Plot() as qp:
#         qp.plot(sp.linspace(0, 1, N, endpoint=True), thetas)
#         qp.plot(sp.linspace(0, 1, M, endpoint=True), thetas_rec, 'o')
#         qp.ylabel(r"$\theta$")
#
#     with quick.Plot() as qp:
#         qp.plot(sp.linspace(0, 1, N, endpoint=True), psis)
#         qp.plot(sp.linspace(0, 1, M, endpoint=True), psis_rec, 'o')
#         qp.ylabel(r"$\psi$")


# ## Test cross product
# if __name__ == '__main__':
#     # OK: orthogonal vectors
#     # OK: right-hand triplet
#     # OK: Anti-symmetry
#     # OK: Length scaling
#     # OK: zero on parallel vectors
#     # OK: non-orthogonal, non-parallel
#
#     r1 = (1, 0, 1)
#     r2 = (0.5, 0.5, 0)
#
#     angles1 = get_ds_angles(*r1)
#     angles2 = get_ds_angles(*r2)
#
#     print(calc_cross_from_angles(*angles1, *angles2))
#     print(calc_cross_from_angles(*angles2, *angles1))


# ## Test thetas to pi interval
# if __name__ == '__main__':
#     # OK: preserves cosine
#     # OK: maps to [0,pi)
#     # OK: works on 2D arrays
#     import quick
#     thetas = sp.linspace(- 2 * pi, 4 * pi, 100, endpoint=True)
#
#     thetas_new = thetas_to_pi_interval(thetas)
#     with quick.Plot() as qp:
#         qp.plot(thetas, sp.cos(thetas))
#         qp.plot(thetas, sp.cos(thetas_new), 'ro')
#     with quick.Plot() as qp:
#         qp.plot(thetas, thetas_new)
