import numpy as np
import matplotlib.pyplot as plt
from func import *

def draw(f_x):
# allocate grids
    dx = np.arange(-3.0, 2.5, .02)
    dy = np.arange(-4.5, 4.5, .02)
    X, Y = np.meshgrid(dx, dy)

# eval function f_x
    Z = [f_x(np.matrix([[x1], [x2]])) \
        for (x1, x2) in zip(np.hstack(X), np.hstack(Y))]
    Z = np.array(Z)
    Z = Z.reshape(np.shape(X))

# take this pencil
    levels = [0.0, 1, 2, 5, 20, 30, 50, 80, 100, 200, 400]
    #cs = plt.contour(X, Y, Z, levels, colors='k')
    cs = plt.contourf(X, Y, Z, 20, cmap=plt.cm.YlGnBu)
    #plt.clabel(cs, fmt='%.1f', inline=1)
