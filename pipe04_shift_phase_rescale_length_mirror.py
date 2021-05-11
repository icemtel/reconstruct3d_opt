'''
- Was used in original pipeline
- Replaced by a new script so that the axes are immediately correctly aligned.

- Define where the velocity has local minima.
  - Shift phases to start from one of that points.
- Rescale to 10 um
- Swap axes in output, s.t. the cilium stands at xy-plane and beat in y direction.

- UPD: IMPORANT: also the projections data was incorrectly mirrored.
    In this file I mirror the 3D shapes back, to correctly obtain the 3D beat pattern, like in Naitoh1984
'''
import os.path
import numpy as np
from math import cos, sin
from scipy.linalg import norm
import curve3d as curve3d
import scipy.optimize as opt


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
ds_folder = 'res/pipe02'
angles_fourier_folder = 'res/pipe03'
output_folder = 'res/pipe_final'
rescale_coeff = 10 / 170.258  # New legnth  [um] / old length [px]
dphi = 0.001 # for fintie difference
n_phases = 20 # for output

os.makedirs(output_folder, exist_ok=True)
### Load Data
ds = np.loadtxt(os.path.join(ds_folder, 'ds.dat'))
psi_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "psi_coeffs.dat"))
theta_coeffs = np.loadtxt(os.path.join(angles_fourier_folder, "theta_coeffs.dat"))
n_coeffs = psi_coeffs.shape[0]

# Define
basis_functions = [get_basis_function(k) for k in range(n_coeffs)]


def get_rr(phi):
    basis = [b(phi) for b in basis_functions]
    psis = basis @ psi_coeffs
    thetas = basis @ theta_coeffs
    rr = curve3d.build_curve((0, 0, 0), ds, thetas, psis)
    return rr


def get_drrdphi(phi):
    drrdphi = (get_rr(phi + dphi) - get_rr(phi - dphi)) / dphi / 2
    return drrdphi


def get_drrdphi_norm(phi):
    return norm(get_drrdphi(phi))


def get_drrdphi_int(phi):
    return norm(np.mean(get_drrdphi(phi), axis=0))


### Make plots and find local minima
## Make plots; find where to start search for minima
phis = np.linspace(0, 2 * np.pi, 100, endpoint=False)
B = np.array([[b(phi) for b in basis_functions] for phi in phis])

psis2D = B @ psi_coeffs
thetas2D = B @ theta_coeffs
rr3D = np.array([get_rr(phi) for phi in phis])

drrdphi3D = np.array([get_drrdphi(phi) for phi in phis])
drrdphi_norm = norm(drrdphi3D, axis=-1)
drrdphi_int = np.array([get_drrdphi_int(phi) for phi in phis])
#s_id = 119
# with quick.Plot() as qp:
#     qp.plot(phis, drrdphi_norm[:,s_id])
# with quick.Plot() as qp:
#     qp.plot(phis, drrdphi_int)

## Determine minima of velocity - use it as a start of a power/recovery stroke
optRes1 = opt.minimize(get_drrdphi_int, x0=2)
recovery_phase = optRes1.x  # 2.16

optRes2 = opt.minimize(get_drrdphi_int, x0=6)
power_phase = optRes2.x  # 5.46

### Shift phase
shift = recovery_phase
phis_new = (shift + np.linspace(0, 2 * np.pi, n_phases, endpoint=False) ) % (2 * np.pi)

rr3D_new = rescale_coeff * np.array([get_rr(phi) for phi in phis_new])
drrdphi3D_new = rescale_coeff * np.array([get_drrdphi(phi) for phi in phis_new])


### Export
## NB: Swap axes: x_new = z ; y_new = x; z_new = y
## Mirror axes: x -> -x
np.savetxt(os.path.join(output_folder, 'x-data'), - rr3D_new[:, :, 2])
np.savetxt(os.path.join(output_folder, 'y-data'), rr3D_new[:, :, 0])
np.savetxt(os.path.join(output_folder, 'z-data'), rr3D_new[:, :, 1])

np.savetxt(os.path.join(output_folder, 'philist'), phis_new - shift)
np.savetxt(os.path.join(output_folder, 'dxdphi-data'), - drrdphi3D_new[:, :, 2])
np.savetxt(os.path.join(output_folder, 'dydphi-data'), drrdphi3D_new[:, :, 0])
np.savetxt(os.path.join(output_folder, 'dzdphi-data'), drrdphi3D_new[:, :, 1])


## Plot new derivatives
import matplotlib.pyplot as plt

plt.plot(phis_new, norm(np.mean(drrdphi3D_new, axis=1), axis=-1), 'o')
plt.plot(phis, rescale_coeff * drrdphi_int)
plt.show()