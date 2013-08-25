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

# f(x_0, x_1) = 10 * (x_1 - x_0^2)^2 + (x_0 - 1)^2
    def __init__(self):
        pass

    def f_x(self, x):
        f = 10 * (x[1] - (x[0])**2)**2 + (1-x[0])**2
        return f[0, 0]

    def g_x(self, x):
        g0 = -40.0 * (x[0]*x[1] - x[0]**3) - 2.0 + 2.0*x[0]
        g1 = 20.0 * (x[1] - x[0]**2)
        return np.matrix([[g0[0,0]], [g1[0,0]]])

    def G_x(self, x):
        G00 = 120.0 * x[0]**2 - 40.0*x[1] + 2
        G01 = -40.0*x[0]
        G10 = -40.0*x[0]
        G11 = 20.0
        return np.matrix([[G00[0,0], G01[0,0]], [G10[0,0], G11]])

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
