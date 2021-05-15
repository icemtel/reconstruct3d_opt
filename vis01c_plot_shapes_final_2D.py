'''
- Plot the resulting shapes (lines) and the initial data (dots)
- Rescale such that resulting shapes length = 10 micron
- Flip x-label for correct orientation
'''
import numpy as np
import os
from math import sin, cos
import curve3d

# Change font globally - bigger for export
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', size=16)


# Functions
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


def get_rr(phi):
    basis = [b(phi) for b in basis_functions]
    psis = basis @ psi_coeffs
    thetas = basis @ theta_coeffs
    rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
    return rr


def get_colors(num, cmap, minval=0.0, maxval=1.0, endpoint=True):
    '''
    :param num: How many colors to return
    :param minval, maxval: truncate colormap by choosing numbers between 0 and 1 (untruncated = [0,1])
    :param cmap: e.g. 'jet' 'viridis' 'RdBu_r' 'hsv'
    :return:
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)  #
    return cmap(np.linspace(minval, maxval, num, endpoint=endpoint))


# Input
raw_folder = 'data/rescale/'
ds_folder = 'res/pipe02/'
angles_fourier_folder = 'res/pipe03/'
# ===== final =====
### Load Data - from pipe03 step
ds = np.loadtxt(os.path.join(ds_folder, 'ds.dat'))
psi_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "psi_coeffs.dat"))
theta_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "theta_coeffs.dat"))
n_coeffs = psi_coeffs.shape[0]

# Define
basis_functions = [get_basis_function(k) for k in range(n_coeffs)]
phi_shift = 2.16035077 # for colorcode
norm = mpl.colors.Normalize(0, 2 * np.pi)
sm = mpl.cm.ScalarMappable(norm=norm,cmap='hsv')
colors = get_colors(42, 'hsv')
scale = 10 / 170.258  # New legnth  [um] / old length [px]

iframes_raw = np.array(list(range(1, 18, 2)) + [18] + list(range(21, 42, 2)))
phis = 2 * np.pi * (iframes_raw - 1) / max(iframes_raw)
# Get shapes
### Make plots and find local minima
## Make plots; find where to start search for minima
# phis = np.linspace(0, 2 * np.pi, 100, endpoint=False)
B = np.array([[b(phi) for b in basis_functions] for phi in phis])

psis2D = B @ psi_coeffs
thetas2D = B @ theta_coeffs
rr3D = scale * np.array([get_rr(phi) for phi in phis])
### Shift phase
phis_plot = (- phi_shift + phis) % (2 * np.pi)

# For correct axes size
xlim = [-6, 10]
ylim = [0, 10]
zlim = [-2, 8]

gskw = dict(width_ratios= [np.diff(xlim)[0], np.diff(xlim)[0]],
            height_ratios=[np.diff(ylim)[0], np.diff(zlim)[0]])

gs = mpl.gridspec.GridSpec(2, 2, **gskw)
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(gs[0, :], aspect="equal", adjustable='box')
ax2 = fig.add_subplot(gs[1, :], aspect="equal", adjustable='box')

# fig, (ax1, ax2) = plt.subplots(2,1, sharex='all')
for iframe_raw, phi, rr in zip(iframes_raw, phis_plot, rr3D):
    xx, yy, zz = rr.T
    #cc = colors[iframe_raw]  # shapes are starting from 1
    cc = sm.to_rgba(phi)
    # Plot the target points
    xname = os.path.join(raw_folder, 'x0-{}.dat'.format(iframe_raw))
    yname = os.path.join(raw_folder, 'y0-{}.dat'.format(iframe_raw))
    xx0 = scale * np.loadtxt(xname)
    yy0 = scale * np.loadtxt(yname)

    ax1.plot(xx, yy, color=cc)
    ax1.scatter(xx0, yy0, fc=cc, ec='black', s=10, zorder=5)

    # Plot the target points
    x2name = os.path.join(raw_folder, 'x20-{}.dat'.format(iframe_raw))
    zname = os.path.join(raw_folder, 'z0-{}.dat'.format(iframe_raw))
    xx20 = scale * np.loadtxt(x2name)
    zz0 = scale * np.loadtxt(zname)

    ax2.plot(xx, zz, color=cc)
    ax2.scatter(xx20, zz0, fc=cc, ec='black', s=10, zorder=5)

ax1.set_xlim(xlim)
ax2.set_xlim(xlim)
ax1.set_ylim(ylim)
ax2.set_ylim(zlim)

ax2.set_xlabel('$y$')
ax1.set_ylabel('$z$')
ax2.set_ylabel('$- x$')

filename = 'figs/projections'
plt.savefig(f'{filename}.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{filename}.eps', bbox_inches='tight') # without bbox_tight axis labels are cut
plt.savefig(f"{filename}.svg", bbox_inches="tight")
plt.show()
