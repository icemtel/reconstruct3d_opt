'''
- Move from tangents which correspond to segments to tangents, which correspond to each of the points
- Get normals to build the flagellum
  NB: This not a Frenet normal and binormal;
  instead get u = t x ex v = t x ex (before rotation of the system this is ez)- they change smoothly
'''
import os.path
import numpy as np
from math import cos, sin
from scipy.linalg import norm
import curve3d as curve3d
import scipy.optimize as opt

# Input
input_folder = 'res/pipe_final'
output_folder = 'res/pipe_final'
os.makedirs(output_folder, exist_ok=True)

### Load Data
xx = np.loadtxt(os.path.join(input_folder, 'x-data'))
yy = np.loadtxt(os.path.join(input_folder, 'y-data'))
zz = np.loadtxt(os.path.join(input_folder, 'z-data'))

rr3D = np.moveaxis(np.array([xx, yy, zz]), 0, 2)
# Tangents, corresponding to segments
tt3D_segments = np.diff(rr3D, axis=1)
tt3D_segments = tt3D_segments / norm(tt3D_segments, axis=-1)[:, :, np.newaxis]
shape = tt3D_segments.shape

# To get a tangent in a node, sum up tangents of adjecent segments; then normalize <-> "mean" tangent on a sphere
# On the first and the last points the tangent remains unchanged.
tt3D = np.zeros((shape[0], shape[1] + 1, shape[2]))
tt3D[:, :-1, :] = tt3D_segments
tt3D[:, 1:, :] += tt3D_segments
tt3D = tt3D / norm(tt3D, axis=-1)[:, :, np.newaxis]

## Get u = t x ex v = t x ex ( before rotation of the system this is ez)
## Use u as normal and v as binormal.
axis = np.array([1, 0, 0])
uu3D = np.cross(tt3D, axis[np.newaxis, np.newaxis, :], axis=-1)
vv3D = np.cross(tt3D, uu3D, axis=-1)
## Normalize
uu3D = uu3D / norm(uu3D, axis=-1)[:, :, np.newaxis]
vv3D = vv3D / norm(vv3D, axis=-1)[:, :, np.newaxis]

## Save results
np.savetxt(os.path.join(output_folder, 'tx-data'), tt3D[:, :, 0])
np.savetxt(os.path.join(output_folder, 'ty-data'), tt3D[:, :, 1])
np.savetxt(os.path.join(output_folder, 'tz-data'), tt3D[:, :, 2])
np.savetxt(os.path.join(output_folder, 'nx-data'), uu3D[:, :, 0])
np.savetxt(os.path.join(output_folder, 'ny-data'), uu3D[:, :, 1])
np.savetxt(os.path.join(output_folder, 'nz-data'), uu3D[:, :, 2])
## It's enough to save t and n to reconstruct b
# sp.savetxt(os.path.join(output_folder, 'bx-data'), vv3D[:, :, 0])
# sp.savetxt(os.path.join(output_folder, 'by-data'), vv3D[:, :, 1])
# sp.savetxt(os.path.join(output_folder, 'bz-data'), vv3D[:, :, 2])



#### Visualization and checkcs
## Is axis good? - OK
## Check if t is not parallel to the axis => length u is close to 1
# if True:
#     u_norms = norm(sp.cross(tt3D, axis[sp.newaxis, sp.newaxis, :], axis=-1),axis=-1)
#     print(sp.amin(u_norms))

## Visualize
# if True:
#     import mayavi.mlab as mlab
#
#     skip = 3
#     phase_skip = 1
#
#     scale_mode = 'none' # quiver
#     scale_factor = 0.5 # quiver
#     # Shapes
#     mlab.points3d(rr3D[::phase_skip,::skip,0],rr3D[::phase_skip,::skip,1],rr3D[::phase_skip,::skip,2], scale_factor=0.1)
#     # Tangents
#     mlab.quiver3d(rr3D[::phase_skip,::skip,0],rr3D[::phase_skip,::skip,1],rr3D[::phase_skip,::skip,2],
#                   tt3D[::phase_skip, ::skip, 0], tt3D[::phase_skip, ::skip, 1], tt3D[::phase_skip, ::skip, 2],
#                   scale_mode=scale_mode, scale_factor=scale_factor, color=(1,0,0))
#     # U
#     mlab.quiver3d(rr3D[::phase_skip,::skip,0],rr3D[::phase_skip,::skip,1],rr3D[::phase_skip,::skip,2],
#                   uu3D[::phase_skip, ::skip, 0], uu3D[::phase_skip, ::skip, 1], uu3D[::phase_skip, ::skip, 2],
#                   scale_mode=scale_mode, scale_factor=scale_factor, color=(0,1,0))
#     # V
#     mlab.quiver3d(rr3D[::phase_skip,::skip,0],rr3D[::phase_skip,::skip,1],rr3D[::phase_skip,::skip,2],
#                   vv3D[::phase_skip, ::skip, 0], vv3D[::phase_skip, ::skip, 1], vv3D[::phase_skip, ::skip, 2],
#                   scale_mode=scale_mode, scale_factor=scale_factor, color=(0,0,1))
#     mlab.show()
#
#
