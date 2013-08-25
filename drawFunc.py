import numpy as np
import matplotlib.pyplot as plt
from func import *

def draw(f_x):
# allocate grids
    dx = np.arange(-1.5, 2, .02)
    dy = np.arange(-3.0, 3.5, .02)
    X, Y = np.meshgrid(dx, dy)

# eval function f_x
    Z = [f_x(np.matrix([[x1], [x2]])) \
        for (x1, x2) in zip(np.hstack(X), np.hstack(Y))]
    Z = np.array(Z)
    Z = Z.reshape(np.shape(X))

# take this pencil
    cs = plt.contour(X, Y, Z, 20, colors='k')
    plt.clabel(cs, inline=1)
