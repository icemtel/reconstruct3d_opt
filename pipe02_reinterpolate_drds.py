'''
2019-04 A Solovev
- Cut lengths;
- Reinterpolate / extrapolate to 120 points and equal shapes length
- Fix discontinuity in psi if any
'''

import os.path
import numpy as np
from scipy.interpolate import pchip_interpolate
import curve3d as curve3d
from math import pi

# Input
input_folder = 'res/pipe01'
output_folder = 'res/pipe02'
iframes_raw = list(np.loadtxt(os.path.join(input_folder, 'iframe.dat'), dtype=np.uint))
N_new = 120
length_new = 170.258  # mean across different phases

#
os.makedirs(output_folder, exist_ok=True)
dsname = os.path.join(input_folder, 'ds.dat')
thetaname = os.path.join(input_folder, 'theta.dat')
psiname = os.path.join(input_folder, 'psi.dat')

ds_list = np.loadtxt(dsname)
thetas_list = curve3d.thetas_to_pi_interval(np.loadtxt(thetaname))
psis_list = np.loadtxt(psiname) % (2 * np.pi)

thetas_new_list = []
psis_new_list = []
xlist_new = []
ylist_new = []
zlist_new = []
for iframe in iframes_raw:  # [15]:  #
    print("iframe:", iframe)
    idx = iframes_raw.index(iframe)
    ds = ds_list[idx]
    thetas = thetas_list[idx]
    psis = psis_list[idx]

    tangents = np.array([curve3d.get_dr(1, theta, psi) for theta, psi in zip(thetas, psis)])
    txs, tys, tzs = tangents.T

    N = len(thetas)
    length = ds * N
    #   print("Length:", length)
    slist = ds / 2 + np.linspace(0, length, N, endpoint=False)  # s corresponding to center of segments

    #   print("Interpolate (extrapolate) to length:", length_new)
    ds_new = length_new / N_new
    slist_new = ds_new / 2 + np.linspace(0, length_new, N_new, endpoint=False)

    # Reinterpolate
    txs_new = pchip_interpolate(slist, txs, slist_new)
    tys_new = pchip_interpolate(slist, tys, slist_new)
    tzs_new = pchip_interpolate(slist, tzs, slist_new)
    # Reconstruct angles
    ds_theta_psi_new_list = np.array(
        [curve3d.get_ds_angles(tx, ty, tz) for (tx, ty, tz) in zip(txs_new, tys_new, tzs_new)])
    tangent_lengths, thetas_new, psis_new = ds_theta_psi_new_list.T

    eps = 0.001
    counter = 0
    counter_max = 10
    while abs(max(tangent_lengths) - 1) > eps \
            or abs(min(tangent_lengths) - 1) > eps \
            and counter < counter_max:
        # Keep track of iterations number
        counter += 1
        if counter == counter_max:
            assert False
        # Fix lengths
        slist_new_real = ds_new * (
                np.cumsum(tangent_lengths) - tangent_lengths / 2)  # Offset to the middle of tangent segments
        txs_new = txs_new / tangent_lengths
        tys_new = tys_new / tangent_lengths
        tzs_new = tzs_new / tangent_lengths

        txs_new = pchip_interpolate(slist_new_real, txs_new, slist_new)
        tys_new = pchip_interpolate(slist_new_real, tys_new, slist_new)
        tzs_new = pchip_interpolate(slist_new_real, tzs_new, slist_new)

        # Reconstruct angles
        ds_theta_psi_new_list = np.array(
            [curve3d.get_ds_angles(tx, ty, tz) for (tx, ty, tz) in zip(txs_new, tys_new, tzs_new)])
        tangent_lengths, thetas_new, psis_new = ds_theta_psi_new_list.T

    if counter > 0:
        print("Counter:", counter)

    rr_new = curve3d.build_curve((0, 0, 0), ds_new, thetas_new, psis_new)
    xx_new, yy_new, zz_new = rr_new.T

    xlist_new.append(xx_new)
    ylist_new.append(yy_new)
    zlist_new.append(zz_new)
    thetas_new_list.append(thetas_new)
    psis_new_list.append(psis_new)

### Fix angle discontinuity
# Similar to unwrap
# For better Fourier expansion quality

def find_optimal_shift(val, val2, period=2 * pi):
    shifts = np.array([-1, 0, 1]) * period

    optimal_shift_idx = np.argmin(abs(val - val2 + shifts))  # Find which shift provides better continuity

    return shifts[optimal_shift_idx]


psis2D = np.array(psis_new_list)
psis2D_new0 = psis2D.copy()

### Fix continuity of all frames
# Idea # 2 - fix only the first phase - then try to fix in the other dimension
psis = psis2D_new0[0]

psis_new = np.full_like(psis, fill_value=np.nan)
psis_pre = iter(psis)
cum_shift = 0
for i, psi in enumerate(psis):
    if i == 0:
        psis_new[0] = psi
        continue

    psi_pre = next(psis_pre)
    opt_shift = find_optimal_shift(psi,
                                   psi_pre)  # Shift next value of psi, if it will make it lie closer to the previous value

    cum_shift += opt_shift
    psis_new[i] = psi + cum_shift

psis2D_new = psis2D_new0.copy()
psis2D_new[0] = psis_new

# Fix the second dimension
psis2D_new = np.array(psis2D_new)
psis2D_new2 = []
for psis in psis2D_new.T:  # Fix one dimension

    psis_new = np.full_like(psis, fill_value=np.nan)
    psis_pre = iter(psis)
    cum_shift = 0
    for i, psi in enumerate(psis):
        if i == 0:
            psis_new[0] = psi
            continue

        psi_pre = next(psis_pre)
        opt_shift = find_optimal_shift(psi, psi_pre)
        # Shift next value of psi, if it will make it lie closer to the previous value

        cum_shift += opt_shift
        psis_new[i] = psi + cum_shift

    psis2D_new2.append(psis_new)

psis2D_new2 =  np.array(psis2D_new2).T
# Visualize
# for psis, psis_new in zip(psis2D, psis2D_new2):
#     with quick.Plot() as qp:
#         qp.plot(psis,'o')
#         qp.plot(psis_new)

psis2D = psis2D_new2

## Theta doesn't need a fix
## Because the axes are chosen s.t. thetas is not coming close to interval borders: 0 and pi
## Check it
# Visualize
# for thetas in thetas2D.T:
#     with quick.Plot() as qp:
#         qp.plot(thetas,'o')



np.savetxt(os.path.join(output_folder, 'theta.dat'), thetas_new_list)
np.savetxt(os.path.join(output_folder, 'psi.dat'), psis2D)
np.savetxt(os.path.join(output_folder, 'ds.dat'), [ds_new])
np.savetxt(os.path.join(output_folder, 'iframe.dat'), iframes_raw, fmt='%d')

np.savetxt(os.path.join(output_folder, 'x.dat'), xlist_new)
np.savetxt(os.path.join(output_folder, 'y.dat'), ylist_new)
np.savetxt(os.path.join(output_folder, 'z.dat'), zlist_new)
