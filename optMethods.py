import numpy as np

def newton(start_point, obj_fun, modified=0, iterations=10):

# with second order information
    x = start_point
    track = x

    k = 0
    while k < iterations:

        if modified > 0:
            vI = modified * np.matrix([[1, 0], [0, 1]])
            G_ = np.linalg.inv(obj_fun.G_x(x) + vI) # inverse?
        else:
            G_ = np.linalg.inv(obj_fun.G_x(x)) # inverse?
        g_ = obj_fun.g_x(x)
        delta = -1.0 * np.dot(G_, g_)
        x = x + delta

        track = np.concatenate((track, x), axis=1)
        k += 1

    return track

def quasi_newton(start_point, obj_fun, iteration=10):

# with first order information
    x = start_point
    track = x

    k = 0
    H = np.matrix([[1, 0], [0, 1]])

    while k < iteration:

        s = -1.0 * np.dot(H, obj_fun.g_x(x))

        alpha_k = _backtracking_line_search(obj_fun, x, s)
        delta_k = alpha_k * s
        x_k_1 = x + delta_k

        gamma_k = obj_fun.g_x(x_k_1) - obj_fun.g_x(x)
        u = delta_k - np.dot(H, gamma_k)

        scale_a = np.dot(u.T, gamma_k)
        if scale_a == 0:
            scale_a = 0.000001
        H = H + np.dot(u, u.T) / scale_a

        track = np.concatenate((track, x), axis=1)
        x = x_k_1
        k += 1

    return track

def _backtracking_line_search(obj_fun, x, s):
    alpha = 1.0
    p = 0.86 # magic number
    c = 0.0001
    diff = 1

    i = 0
    while (diff > 0 & i < 10000):
        f = obj_fun.f_x(x + alpha * s)
        f_x = obj_fun.f_x(x)
        diff = f - f_x - c * alpha * np.dot(obj_fun.g_x(x).T, s)
        alpha *= p
        i += 1
    return alpha
