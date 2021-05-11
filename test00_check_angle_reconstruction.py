'''
Result: visual inspection -> alright
'''

import os.path
import numpy as np
import curve3d
import numpy.testing as nptest

output_folder = 'res/rescale_new_angles'
output_folder_angles = os.path.join(output_folder, 'angles')
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))

ds_list = []
psis_list = []
thetas_list = []
colors = []

for iframe_raw in iframes_raw:
    # if iframe_raw in  [5, 13, 18, 35]: # iframes_raw: #in [31]: #[13, 35, 39]:
    #     pass
    # else:
    #     next(all_colored)
    #     continue

    dsname = os.path.join(output_folder_angles, 'ds-{}.dat'.format(iframe_raw))
    thetaname = os.path.join(output_folder_angles, 'theta-{}.dat'.format(iframe_raw))
    psiname = os.path.join(output_folder_angles, 'psi-{}.dat'.format(iframe_raw))

    ds = np.loadtxt(dsname)
    thetas = np.loadtxt(thetaname) % np.pi
    psis= np.loadtxt(psiname) % (2 * np.pi)

    ds_list.append(ds)
    thetas_list.append(thetas)
    psis_list.append(psis)


    xname = os.path.join(output_folder, 'x-{}.dat'.format(iframe_raw))
    yname = os.path.join(output_folder, 'y-{}.dat'.format(iframe_raw))
    zname = os.path.join(output_folder, 'z-{}.dat'.format(iframe_raw))

    xx = np.loadtxt(xname)
    yy = np.loadtxt(yname)
    zz = np.loadtxt(zname)
    rr = np.array([xx, yy, zz]).T

    _, ds_rec, thetas_rec, psis_rec = curve3d.reconstruct_angles(rr, len(xx) - 1)


    nptest.assert_allclose(ds, ds_rec, 10**-5, 10**-6)
    nptest.assert_allclose(thetas, thetas_rec, 10**-5, 10**-6)
    nptest.assert_allclose(psis, psis_rec, 10**-5, 10**-6)