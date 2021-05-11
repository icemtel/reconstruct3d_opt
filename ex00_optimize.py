'''
Consider 1D array x:
Optimize s.t. f(x) -> min

Works fine on simple examples;
Need to make additional test for
- different methods
- more computationally expensive examples
'''

import scipy.optimize as opt
import numpy as np
import random
from scipy.linalg import norm
import matplotlib.pyplot as plt

random.seed(404)

# Fit a curve to random data w(z)
N = 100
k = 3 # noise strength
tol = 10 ** -3
zs = []
ws = []
for n in range(N):
    z = 10 * random.random()
    w = 5 * z + 3 + random.random() * k # linear dependence + noise
    zs.append(z)
    ws.append(w)
zs = np.array(zs)
ws = np.array(ws)

def get_curve(x):
    a,b,c = x
    def curve(z):
        return a * z + b + c * z ** 2

    return curve

def f(x):
    '''
    x = [a,b, c]: in linear fit a z + b + c * z ** 2
    '''
    curve = get_curve(x)

    vals = [curve(z) for z in zs]
    penalty = norm( ws - vals)
    return penalty




x0 = (5, 5, 5)
res = opt.minimize(f,x0,tol=tol)

# Result
if res.success:
    print(res.x)
else:
    print("FAILURE")
    print(res.x)
    print(res.status)
    print(res.message)


# Plot
w_function = get_curve(res.x)
zs0 = np.linspace(0, 10, endpoint=True)

plt.plot(zs, ws, 'o')
plt.plot(zs0, [w_function(z) for z in zs0])
plt.show()
