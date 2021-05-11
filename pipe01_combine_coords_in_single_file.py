import os.path
import numpy as np
import curve3d

folder = 'res/pipe00'
folder_angles = os.path.join(folder, 'angles')
output_folder = 'res/pipe01'
os.makedirs(output_folder, exist_ok=True)
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))

x = []
y = []
z = []

ds_list = []
psis_list = []
thetas_list = []
for iframe_raw in iframes_raw:
    xname = os.path.join(folder, 'x-{}.dat'.format(iframe_raw))
    yname = os.path.join(folder, 'y-{}.dat'.format(iframe_raw))
    zname = os.path.join(folder, 'z-{}.dat'.format(iframe_raw))

    xx = np.loadtxt(xname)
    yy = np.loadtxt(yname)
    zz = np.loadtxt(zname)
    rr = np.array([xx, yy, zz]).T

    x.append(xx)
    y.append(yy)
    z.append(zz)

    dsname = os.path.join(folder_angles, 'ds-{}.dat'.format(iframe_raw))
    thetaname = os.path.join(folder_angles, 'theta-{}.dat'.format(iframe_raw))
    psiname = os.path.join(folder_angles, 'psi-{}.dat'.format(iframe_raw))

    ds = np.loadtxt(dsname)
    thetas = curve3d.thetas_to_pi_interval(np.loadtxt(thetaname))
    psis = np.loadtxt(psiname) % (2 * np.pi)

    ds_list.append(ds)
    thetas_list.append(thetas)
    psis_list.append(psis)



np.savetxt(os.path.join(output_folder, 'x.dat'), x)
np.savetxt(os.path.join(output_folder, 'y.dat'), y)
np.savetxt(os.path.join(output_folder, 'z.dat'), z)

np.savetxt(os.path.join(output_folder, 'ds.dat'), ds_list)
np.savetxt(os.path.join(output_folder, 'theta.dat'), thetas_list)
np.savetxt(os.path.join(output_folder, 'psi.dat'), psis_list)
np.savetxt(os.path.join(output_folder, 'iframe.dat'), iframes_raw, fmt='%d')

## Combine angles