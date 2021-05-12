'''
Setup for cilia geometry
'''
import os
import numpy as np

shapes_data_path = os.path.dirname(__file__) + '/shapes_data'
data_key = 'machemer'
num_phases = 20
dphi = 2 * np.pi / num_phases

# Phyiscal values
beat_period = 31.25  # ms; 32 Hz
beat_freq = 2 * np.pi / beat_period
beat_time_step = beat_period / num_phases  # ms
flag_length = 10  # um; from the data


def data_loader(*filenames):
    '''
    Load cilia shapes;
    - skip every n-th data point in the middle of the cilium:
    - keep refined spacing near the tips
    :param filenames: Names of files to contain x,y,z data
    :return: A function which returns np.array: rs = [[x1,y1,z1],[x2,y2,z2],..]
    '''

    def load_data(phase, longitude_grid, near_tip_skip=0):
        '''
        :param phase:
        :param longitude_grid: approximately number of points along the cilium
        :param near_tip_skip: =0 means the end of cilium will be flat
        :return:
        '''
        xs = []
        for filename in filenames:
            x = np.loadtxt(os.path.join(shapes_data_path, filename))[phase]
            xs.append(x)
        vals = np.array(xs).transpose()  # e.g. xs = [xx,yy,zz] => vals = [[x1,y1,z1],[x2,y2,z2],..]
        num_points = len(vals)

        # Explicitly define indices to keep, other data points will not be used
        spacing = (num_points - 2 * near_tip_skip) // longitude_grid
        indices = [0] +  list(np.arange(near_tip_skip, num_points - near_tip_skip, spacing)) + [num_points - 1]
        vals = vals[indices]
        return vals

    return load_data


# Define commands to load coordinates, tangent vectors, normal vectors, and phase-derivative of coordinates dr/dphi
load_rr = data_loader('x-data', 'y-data', 'z-data')
load_tangents = data_loader('tx-data', 'ty-data', 'tz-data')
load_normals = data_loader('nx-data', 'ny-data', 'nz-data')
load_drrdphi = data_loader('dxdphi-data', 'dydphi-data', 'dzdphi-data')