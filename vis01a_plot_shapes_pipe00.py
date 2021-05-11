###########################
# Plot in 3D #
###########################
import numpy as np
import os
import vedo
import matplotlib.pyplot as plt

folder = 'res/pipe00'  #
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))


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


colors = get_colors(42, 'hsv')
shapes = []
for iframe_raw in iframes_raw:
    xname = os.path.join(folder, 'x-{}.dat'.format(iframe_raw))
    yname = os.path.join(folder, 'y-{}.dat'.format(iframe_raw))
    zname = os.path.join(folder, 'z-{}.dat'.format(iframe_raw))

    xx = np.loadtxt(xname)
    yy = np.loadtxt(yname)
    zz = np.loadtxt(zname)
    cc = colors[iframe_raw]  # shapes are starting from 1
    c = tuple(cc[:3])  # discard opacity
    shape = vedo.Tube(list(zip(zz, xx, yy)), r=1, c=c) # NB: Swapped axes!!
    shapes += [shape]

size = 256 * np.array([4, 3])  # window size # 256
offscreen = False  # Hide
interactive = False  # should be false to execute the whole script
plotter = vedo.plotter.Plotter(pos=(100, 0), interactive=interactive,
                               size=size, offscreen=offscreen)

plane = vedo.Plane(pos=(0, 0, 0), normal=(0, 0, 1), sx=100).alpha(0.5)
shapes += [plane]

plotter.add(shapes, render=False)
plotter.show(interactive=True).close()
