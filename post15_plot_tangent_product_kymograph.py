'''
Checked: the same when used angles directly from saved results.

Result:
- tangents change slowly
- curvature calculation is wrong
'''

import os.path
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

input_folder = 'data/rescale'
output_folder =  'res/rescale_new_angles'
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))


t = []
t_prod = []
kappalist = []

for iframe_raw in iframes_raw: #[15]: #
    xname = os.path.join(output_folder, 'x-{}.dat'.format(iframe_raw))
    yname = os.path.join(output_folder, 'y-{}.dat'.format(iframe_raw))
    zname = os.path.join(output_folder, 'z-{}.dat'.format(iframe_raw))

    xx = np.loadtxt(xname)
    yy = np.loadtxt(yname)
    zz = np.loadtxt(zname)
    rr = np.array([xx, yy, zz]).T

    drr = np.diff(rr, axis=0) # OK
    tt = np.array([drrr / norm(drrr) for drrr in drr]) # OK
    tangent_products = np.sum(tt[:-1] * tt[1:], axis=1)


    t_prod.append(tangent_products)
    t.append(tt)

    # Test: curvature; formula from  Taylor expansion of (t_i, t_{i+1}) at i-th point; + Frenet formulas
    # Result: doesn't work well; The reason is probably that the formula has O(ds ** (1/2)) accuracy
    assert np.std(norm(drr, axis=1)) < 0.0001
    ds = np.mean(norm(drr, axis=1))
    kappa = (2 * tangent_products) ** (1 / 2) / ds
    kappalist.append(kappa)


# tangent
filename = None  # os.path.join(output_folder, 'kymograph_psi.png')
plt.title(r'$t_i \cdot t_{i+1}$')
plt.xlabel(r'icoord')
plt.ylabel(r'Frame')
norm = None # quick.MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

cmap = 'Reds'
vals = t_prod
plt.imshow(vals, cmap=cmap, norm=norm, origin='lower')
plt.colorbar()
plt.show()

# Constant along a shape ??
cmap = 'Reds'
vals = kappalist
plt.imshow(vals, cmap=cmap, norm=norm, origin='lower')
plt.colorbar()
plt.show()
