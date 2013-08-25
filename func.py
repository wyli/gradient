import numpy as np

class quadratic:

# trivial (solving linear system)
    A = np.zeros([2, 2])
    b = np.zeros([2, 1])

    def __init__(self):
        pass

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def f_x(self, x):
        f = 0.5 * np.dot(np.dot(x.T, self.A), x) - np.dot(self.b.T, x)
        return f[0, 0]

    def g_x(self, x):
        return np.dot(self.A, x) - self.b

    def G_x(self, x):
        return np.asmatrix(self.A)


class rosenbrock:

    def __init__(self):
        pass

    def f_x(self, x):
        f = 100 * (x[1] - (x[0])**2)**2 + (1-x[0])*2
        return f[0, 0]

    def g_x(self, x):
        g1 = -400.0 * (x[0]*x[1] - x[0]**3) - 2.0 + 2.0*x[0]
        g2 = 200.0 * (x[1] - x[0]**2)
        return np.matrix([[g1[0,0]], [g2[0,0]]])

    def G_x(self, x):
        G11 = 1200.0*x[0]**2 - 400.0*x[1] + 2
        G12 = -400.0*x[0]
        G21 = -400.0*x[0]
        G22 = 200.0
        return np.matrix([[G11[0,0], G12[0,0]], [G21[0,0], G22]])

    def about_alpha(self, x, s):
        # f(a) = f(x + a*s) at point x, direction s
        def along_s(alpha):
            return self.f_x(x+s*alpha)
        return along_s

    def about_alpha_prime(self, x, s):
        def along_s(alpha):
            p = np.dot(self.g_x(x+alpha*s).T, s)
            return p[0, 0]
        return along_s
