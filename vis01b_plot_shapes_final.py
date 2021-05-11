###########################
# Plot in 3D #
###########################
import numpy as np
import os
import vedo
import matplotlib.pyplot as plt

folder = 'res/pipe_final'  #
iframes_raw = list(range(20))


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


x = np.loadtxt(os.path.join(folder, 'x-data'))
y = np.loadtxt(os.path.join(folder, 'y-data'))
z = np.loadtxt(os.path.join(folder, 'z-data'))

colors = get_colors(len(iframes_raw), 'hsv')
shapes = []
for iframe_raw, xx, yy, zz in zip(iframes_raw, x, y, z):
    cc = colors[iframe_raw]  # shapes are starting from 1
    c = tuple(cc[:3])  # discard opacity
    shape = vedo.Tube(list(zip(xx, yy, zz)), r=0.125, c=c)
    shapes += [shape]

size = 256 * np.array([4, 3])  # window size # 256
offscreen = False  # Hide
interactive = False  # should be false to execute the whole script
plotter = vedo.plotter.Plotter(pos=(100, 0), interactive=interactive,
                               size=size, offscreen=offscreen)

plane = vedo.Plane(pos=(0, 0, 0), normal=(0, 0, 1), sx=10).alpha(0.5)
shapes += [plane]

plotter.add(shapes, render=False)
plotter.show(interactive=True).close()
