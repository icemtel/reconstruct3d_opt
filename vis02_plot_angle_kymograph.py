import os.path
import numpy as np
import curve3d
from math import cos, sin

# Matplotlib setup
import matplotlib as mpl
import matplotlib.pyplot as plt
# Change font globally - bigger for export
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
mpl.rc('font', size=16)

# Input
ds_folder = 'res/pipe02'
angles_fourier_folder = 'res/pipe03'
output_folder = 'res/pipe_final'
rescale_coeff = 10 / 170.258  # New legnth  [um] / old length [px]
dphi = 0.001  # for fintie difference
n_phases = 20  # for output

os.makedirs(output_folder, exist_ok=True)
### Load Data
ds = np.loadtxt(os.path.join(ds_folder, 'ds.dat'))
psi_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "psi_coeffs.dat"))
theta_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "theta_coeffs.dat"))
n_coeffs = psi_coeffs.shape[0]


# Define
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


basis_functions = [get_basis_function(k) for k in range(n_coeffs)]


### Make plots and find local minima
## Make plots; find where to start search for minima
phis = np.linspace(0, 2 * np.pi, 100, endpoint=False)
B = np.array([[b(phi) for b in basis_functions] for phi in phis])

psis2D = B @ psi_coeffs
thetas2D = B @ theta_coeffs
print(thetas2D.max(), thetas2D.min())

# theta
filename = None  # os.path.join(output_folder, 'kymograph2_theta.png')
plt.figure(figsize=(5,2))
ax = plt.gca()

plt.title(r'$\theta(s,\varphi)$')
plt.xlabel(r's/L')
plt.ylabel(r'$\varphi$')

cmap = 'plasma'
vals = curve3d.thetas_to_pi_interval(thetas2D)
norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)

plt.imshow(vals, cmap=cmap, norm=norm, aspect='auto', interpolation='none',
           origin='lower', extent=[0, 1, 0, np.pi])


ax.set_yticks(ticks=[0, np.pi / 2, np.pi])
ax.set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

cb = plt.colorbar()
cb.set_ticks(ticks=[0, np.pi / 2, np.pi])
cb.set_ticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$'])


filename = 'figs/theta_kymograph'
plt.savefig(f'{filename}.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{filename}.eps', bbox_inches='tight') # without bbox_tight axis labels are cut
plt.savefig(f"{filename}.svg", bbox_inches="tight")
plt.show()

# psi
plt.figure(figsize=(5,2))
ax = plt.gca()

plt.title(r'$\psi(s,\varphi)$')
plt.xlabel(r's/L')
plt.ylabel(r'$\varphi$')
norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
cmap = 'hsv'
vals = psis2D  % (2 * np.pi)
plt.imshow(vals, cmap=cmap, norm=norm, aspect='auto', interpolation='none',
           origin='lower', extent=[0, 1, 0, 2* np.pi])

ax.set_yticks(ticks=[0, np.pi,  2 * np.pi])
ax.set_yticklabels(['$0$', r'$\pi$', r'$2 \pi$'])

cb = plt.colorbar()
cb.set_ticks(ticks=[0, np.pi,  2 * np.pi])
cb.set_ticklabels(['$0$', r'$\pi$', r'$2 \pi$'])

filename = 'figs/psi_kymograph'
plt.savefig(f'{filename}.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{filename}.eps', bbox_inches='tight') # without bbox_tight axis labels are cut
plt.savefig(f"{filename}.svg", bbox_inches="tight")
plt.show()
