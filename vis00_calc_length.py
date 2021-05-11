import os.path
import numpy as np
import curve3d

folder = 'res/pipe00'
iframes_raw = list(range(1, 18, 2)) + [18] + list(range(21, 42, 2))

lengths = []
for iframe_raw in iframes_raw:
    xname = os.path.join(folder, 'x-{}.dat'.format(iframe_raw))
    yname = os.path.join(folder, 'y-{}.dat'.format(iframe_raw))
    x2name = os.path.join(folder, 'x2-{}.dat'.format(iframe_raw))
    zname = os.path.join(folder, 'z-{}.dat'.format(iframe_raw))

    xx = np.loadtxt(xname)
    yy = np.loadtxt(yname)
    zz = np.loadtxt(zname)

    length = curve3d.calc_length(xx, yy, zz)
    lengths.append(length)
    print('{}\t{:.3f}'.format(iframe_raw, length))


print("Mean: {:.3f}\tSTD: {:.3f}".format(np.mean(lengths), np.std(lengths)))
max_rel_deviation = max(np.mean(lengths) -lengths)  / np.mean(lengths)
print("Max relative deviation: {:.3f}".format(max_rel_deviation))

# Plot
import matplotlib.pyplot as plt
import matplotlib as mpl

# Extra matplotlib set-up (can be skipped)
mpl.rc('font', size=16)
square_size = (3.8709677419354835, 3.9735099337748343)
halfsquare_size = (square_size[0], square_size[1] / 2)
pratio = 1.5
pscale = 1.0
portrait_size = (pscale * 3.8709677419354835 * np.sqrt(pratio), pscale * 3.9735099337748343 / np.sqrt(pratio))
mpl.rc('figure', figsize=portrait_size)

# Make a plot
cut_length = np.mean(lengths)
plt.plot(iframes_raw, lengths, 'o')
plt.axhline(y=cut_length, c='black')
plt.xlabel('Frame')
plt.ylabel('Length, px')

outfolder = 'figs/'
filename = 'lengths'
plt.savefig(outfolder + f'{filename}.png', dpi=400, bbox_inches='tight')
plt.savefig(outfolder + f"{filename}.svg", bbox_inches="tight")
plt.show()