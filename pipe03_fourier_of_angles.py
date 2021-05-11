'''
Calculate and store coefficients of Fourier expansion of angle theta, psi.

- Save coefficients to files
  - Coeffs for basis functions: 1, cos x, sin x, cos2x, sin2x..
- Visual checks are in the end of the file

'''
import os.path
import numpy as np
from math import cos, sin, pi
import scipy.linalg as lin

def get_basis_function(k):
    '''
    0: 1; 1: cos(x); 2: sin(x); 3: cos(2x) ...
    '''
    if k < -1:
        raise ValueError
    elif k == 0:
        return lambda x: 1
    elif k % 2 == 1:
        n = k // 2 + 1
        return lambda x: cos(n * x)
    elif k % 2 == 0:
        n = k // 2
        return lambda x: sin(n * x)


# Input
input_folder = 'res/pipe02'
output_folder = 'res/pipe03'
os.makedirs(output_folder, exist_ok=True)
nharm = 4


# Load data
thetas2D = np.loadtxt(os.path.join(input_folder, 'theta.dat'))
psis2D = np.loadtxt(os.path.join(input_folder, 'psi.dat'))
ds = np.loadtxt(os.path.join(input_folder, 'ds.dat'))
iframes_raw = np.loadtxt(os.path.join(input_folder, 'iframe.dat'), dtype=np.uint)
phis = 2 * pi * (iframes_raw - 1) / max(iframes_raw)

## Linear fit
## fix x_k; treat each point individually
## N = len(iframes_raw)
## B - array (N,2*nharm +1) - matrix of vlaues of basis functions on points where we know values of psi
## c - array (2*nharm +1) - vector with expansion coefficients
## psi - array (N) - values of psi at point x_k; for each of known phases;
## B*c = psi_vals => c =...

# 1. Define basis functions and construct a matrix of values for linear fit
basis_functions = [get_basis_function(k) for k in range(2 * nharm + 1)]
B = np.array([[b(phi) for b in basis_functions] for phi in phis])

# 2. Linear fit to find Fourier coefficients of modified values: psi - slope * phi
#    With weights, proportional to sin(theta);
psi_coeffs = np.full((B.shape[1], psis2D.shape[1]), fill_value=np.nan)
for i, (thetas, psis_mod) in enumerate(zip(thetas2D.T, psis2D.T)):
    weights = abs(np.sin(thetas)) ** (1 / 2)
    weights = np.diag(weights)
    psi_coeffs1D, residuals, rank, singular_vals = lin.lstsq(weights @ B, weights @ psis_mod)
    psi_coeffs[:, i] = psi_coeffs1D

### Theta

# 3. Linear fit to find Fourier coefficients of theta values
theta_coeffs, residuals, rank, singular_vals = lin.lstsq(B, thetas2D)  #

### Save results
np.savetxt(os.path.join(output_folder, "psi_coeffs.dat"), psi_coeffs)
np.savetxt(os.path.join(output_folder, "theta_coeffs.dat"), theta_coeffs)



#####################
# Visual checks - TODO: update
#####################

x2D = np.loadtxt(os.path.join(input_folder, 'x.dat'))
y2D = np.loadtxt(os.path.join(input_folder, 'y.dat'))
z2D = np.loadtxt(os.path.join(input_folder, 'z.dat'))
rr3D = np.array([x2D, y2D, z2D]).transpose((1, 2, 0))

### psi
# phi0 = sp.linspace(0, 2 * sp.pi)
# for idx in [1, 30, 60, 80, 119]:
#     psi_func  = lambda phi: sp.dot(psi_coeffs.T[idx] , [b(phi) for b in basis_functions])
#     with  quick.Plot() as qp:
#         qp.plot(phis, psis2D.T[idx],'o')
#         qp.plot(phi0, sp.vectorize(psi_func)(phi0))

## psi coeffs
# with quick.Plot() as qp:
#     vals = abs(psi_coeffs)
#     print(sp.amin(vals))
#     norm = quick.SymLogNorm(10 **-4, vmin=10 ** -3, vmax=1)
#     cmap = 'jet'
#     qp.imshow(vals, origin='lower', norm=norm, cmap=cmap)
#     qp.colorbar(norm=norm, cmap=cmap)


### theta
# phi0 = sp.linspace(0, 2 * sp.pi, endpoint=False)
# for idx in [0, 30, 60, 119]:
#     theta_func = lambda phi: sp.dot(theta_coeffs.T[idx], [b(phi) for b in basis_functions])
#     with  quick.Plot() as qp:
#         qp.plot(phis, thetas2D.T[idx],'o')
#         qp.plot(phi0, sp.vectorize(theta_func)(phi0))

## theta coeffs
# with quick.Plot() as qp:
#     vals = abs(theta_coeffs)
#     print(sp.amin(vals))
#     norm = quick.SymLogNorm(10 **-4, vmin=10 ** -3, vmax=1)
#     cmap = 'jet'
#     qp.imshow(vals, origin='lower', norm=norm, cmap=cmap)
#     qp.colorbar(norm=norm, cmap=cmap)

### Shapes
# import mayavi.mlab as mlab
#
# colors = quick.get_colors(len(thetas2D), cmap=quick.default_cyclic_colormap)
# for color, thetas, psis in zip(colors, thetas2D, psis2D):
#     rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#     mlab_color = tuple(color[:3])
#     # mlab.plot3d(rr[:, 2], rr[:, 0], rr[:, 1], color=mlab_color,tube_radius=1)
#     mlab.points3d(rr[:, 2], rr[:, 0], rr[:, 1], color=mlab_color, scale_factor=1)
#
# phis0 =  phis  # sp.linspace(0, 2 * sp.pi, endpoint=False)#
# B0 = sp.array([[b(phi) for b in basis_functions] for phi in phis0])
#
# psis2D0 = B0 @ psi_coeffs  # matmul
# thetas2D0 = B0 @ theta_coeffs
#
# colors0 = quick.get_colors(len(thetas2D0), cmap=quick.default_cyclic_colormap)
# for color, thetas, psis in zip(colors0, thetas2D0, psis2D0):
#     rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#     mlab_color = tuple(color[:3])
#     mlab.plot3d(rr[:, 2], rr[:, 0], rr[:, 1], color=mlab_color, tube_radius=1)
#
# # Plane
# X = [[-100, -100], [100, 100]]
# Y = [[-100, 100], [-100, 100]]
# Z = [[0, 0], [0, 0]]
# mlab.mesh(X, Y, Z)
#
# mlab.show()

## Shapes projections
# from matplotlib.lines import Line2D
#
# phis0 = phis
# B0 = sp.array([[b(phi) for b in basis_functions] for phi in phis0])
#
# psis2D0 = B0 @ psi_coeffs
# thetas2D0 = B0 @ theta_coeffs
# legend_elements = [Line2D([0], [0], color='black', lw=2, label='original'),
#                    Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=15, label='Fourier')]
#
# ms = 2
# filename = None  # os.path.join(output_folder, 'plot_xy_YZ.png')
# with quick.Plot() as qp:
#     qp.set_aspect(1)
#     qp.xlim(-90, 160)
#
#     qp.xlabel('x (Y)')
#     qp.ylabel('y (Z)')
#     colors = quick.get_colors(len(thetas2D), cmap=quick.default_cyclic_colormap)
#     for color, thetas, psis, thetas0, psis0 in zip(colors, thetas2D, psis2D, thetas2D0, psis2D0):
#         rr0 = curve3d.build_curve((0, 0, 0), ds, thetas0, psis0)
#         qp.plot(rr0[:, 0], rr0[:, 1], color=color, markersize=ms, alpha=0.5)
#         rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#         qp.plot(rr[:, 0], rr[:, 1], color='black', lw=0.5)
#
#     qp.legend(handles=legend_elements)
#
# filename = None  # os.path.join(output_folder, 'plot_xy_YZ.png')
# with quick.Plot() as qp:
#     qp.set_aspect(1)
#     qp.xlim(-90, 160)
#
#     qp.xlabel('x (Y)')
#     qp.ylabel('z (X)')
#     colors = quick.get_colors(len(thetas2D), cmap=quick.default_cyclic_colormap)
#     for color, thetas, psis, thetas0, psis0 in zip(colors, thetas2D, psis2D, thetas2D0, psis2D0):
#         rr0 = curve3d.build_curve((0, 0, 0), ds, thetas0, psis0)
#         qp.plot(rr0[:, 0], rr0[:, 2], color=color, markersize=ms, alpha=0.5)
#         rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#         qp.plot(rr[:, 0], rr[:, 2], color='black', lw=0.5)
#
#     qp.legend(handles=legend_elements)
#
# filename = None  # os.path.join(output_folder, 'plot_xy_YZ.png')
# with quick.Plot() as qp:
#     qp.set_aspect(1)
#     qp.xlim(-90, 160)
#
#     qp.ylabel('y (Z)')
#     qp.xlabel('z (X)')
#     colors = quick.get_colors(len(thetas2D), cmap=quick.default_cyclic_colormap)
#     for color, thetas, psis, thetas0, psis0 in zip(colors, thetas2D, psis2D, thetas2D0, psis2D0):
#         rr0 = curve3d.build_curve((0, 0, 0), ds, thetas0, psis0)
#         qp.plot(rr0[:, 2], rr0[:, 1], color=color, markersize=ms, alpha=0.5)
#         rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#         qp.plot(rr[:, 2], rr[:, 1], color='black', lw=0.5)
#
#     qp.legend(handles=legend_elements)



### Tangent trajectories on a unit sphere
# import mayavi.mlab as mlab
#
#
# def get_colors(num, cmap, minval=0.0, maxval=1.0, endpoint=True):
#     '''
#     :param num: How many colors to return
#     :param minval, maxval: truncate colormap by choosing numbers between 0 and 1 (untruncated = [0,1])
#     :param cmap: e.g. 'jet' 'viridis' 'RdBu_r' 'hsv'
#     :return:
#     '''
#     if isinstance(cmap, str):
#         cmap = plt.get_cmap(cmap) #
#     return cmap(np.linspace(minval, maxval, num, endpoint=endpoint))
#
#
# phi0 = np.linspace(0, 2 * np.pi, 100, endpoint=False)
# B0 = np.array([[b(phi) for b in basis_functions] for phi in phi0])
# psis2D0 = B0 @ psi_coeffs   # matmul
# thetas2D0 = B0 @ theta_coeffs
#
# tangents2D0 = np.array(np.vectorize(curve3d.get_dr)(np.ones_like(thetas2D0), thetas2D0, psis2D0))
# tangents2D0 = tangents2D0.transpose((1, 2, 0)) # (21, 120, 3)
#
# # Create a sphere
# r = 1
# theta, psi = np.mgrid[0:pi:101j, 0:2 * pi:101j]
# # Spherical coordinates
# z = r * np.cos(theta)
# x = r * np.sin(theta) * np.cos(psi)
# y = r * np.sin(theta) * np.sin(psi)
#
# s = x
# mlab.mesh(x, y, z, scalars=s, colormap='jet')
#
# # Plot trajectories
# s_ids = list(range(120)) # [0, 30, 70, 119] # 0,4 - knots?! #  # [0, 10, 30, 50, 70, 90, 110, 119]
# colors =  get_colors(len(s_ids)) # From dark to light in viridis
# for c,s_id in zip(colors,s_ids):
#     tangents = tangents2D0[:, s_id, :]
#     color = tuple(c[:3])
#     mlab.plot3d(tangents[:,0], tangents[:,1], tangents[:,2],color=color)
#
#
# tangents2D = np.array(np.vectorize(curve3d.get_dr)(np.ones_like(thetas2D), thetas2D, psis2D))
# tangents2D = tangents2D.transpose((1, 2, 0)) # (21, 120, 3)
# # Plot old trajectories
# colors =  get_colors(len(s_ids)) # From dark to light in viridis
# for c,s_id in zip(colors,s_ids):
#     tangents = tangents2D[:, s_id, :]
#     color = tuple(c[:3])
#     mlab.points3d(tangents[:,0], tangents[:,1], tangents[:,2],color=(1,1,1), scale_factor=0.05)
#
# mlab.show()


### Point positions and velocity (simple finite difference)
# phis0 = sp.linspace(0, 2 * sp.pi, endpoint=False)
# B0 = sp.array([[b(phi) for b in basis_functions] for phi in phis0])
#
# psis2D0 = B0 @ psi_coeffs  # matmul
# thetas2D0 = B0 @ theta_coeffs
#
# rr3D0 = []
# for thetas, psis in zip(thetas2D0, psis2D0):
#     rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
#     rr3D0.append(rr)
#
# rr3D0 = sp.array(rr3D0)
#
# s_id = 60
# # New vals
# rr0 = rr3D0[:, s_id, :]
#
# vel0 = sp.diff(rr0, axis=0, append=[rr0[0]]) / sp.diff(phis0, append=[phis0[0] + 2 * pi])[:,sp.newaxis]
# vel0_norm = lin.norm(vel0, axis=-1)
# # Old vals
# rr = rr3D[:, s_id, :]
# vel = sp.diff(rr, axis=0, append=[rr[0]]) / sp.diff(phis, append=[phis[0] + 2 * pi])[:,sp.newaxis]
# vel_norm = lin.norm(vel, axis=-1)
#
# colors = ['r','g','b']
# # COORDS
# filename = None # os.path.join(output_folder, 'sid_{}/coords_nharm_{}.png'.format(s_id,nharm))
# with quick.Plot(filename) as qp:
#     qp.title("x,y,z, s_id={}, nharm={}".format(s_id,nharm))
#     for i in range(3):
#         qp.plot(phis0, rr0[:,i], c=colors[i])
#         qp.plot(phis, rr[:,i], 'o',c=colors[i])
#
# # # VELS
# # with quick.Plot() as qp:
# #     qp.title("vx,vy,vz, s_id={}".format(s_id))
# #     for i in range(3):
# #         qp.plot(phis0, vel0[:,i], c=colors[i])
# #         qp.plot(phis, vel[:,i], 'o', c=colors[i])
# # # VELS NORM
#
# filename = None # os.path.join(output_folder, 'sid_{}/vel_norm_nharm_{}.png'.format(s_id, nharm))
# with quick.Plot(filename) as qp:
#     qp.title("Velocity_tot, s_id={}, nharm={}".format(s_id,nharm))
#     qp.plot(phis0, vel0_norm)
#     qp.plot(phis, vel_norm, 'o')
