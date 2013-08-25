import numpy as np

def newton(start_point, obj_fun, modified=0, iterations=5):
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

def quasi_newton(start_point, obj_fun, iteration=10, interpolation=0):
# with first order information
    x = start_point
    track = x

    k = 0
    H = np.matrix([[1, 0], [0, 1]])

    while k < iteration:

        s = -1.0 * np.dot(H, obj_fun.g_x(x))

        if interpolation > 0:
            alpha_k = _backtracking_line_search(obj_fun, x, s)
        else:
            alpha_k = _armijo(obj_fun, x, s)

        delta_k = alpha_k * s
        x_k_1 = x + delta_k

        gamma_k = obj_fun.g_x(x_k_1) - obj_fun.g_x(x)
        u = delta_k - np.dot(H, gamma_k)

        scale_a = np.dot(u.T, gamma_k)
        if scale_a == 0: # :(
            scale_a = 0.000001
        H = H + np.dot(u, u.T) / scale_a

        track = np.concatenate((track, x), axis=1)
        x = x_k_1
        k += 1

    return track

def _backtracking_line_search(obj_fun, x, s, c=1e-4):
# dummy
    alpha = 1.0
    p = 0.8 # magic number
    c = 0.0001
    diff = 1

    f_alpha = obj_fun.about_alpha(x, s)
    g_alpha = obj_fun.about_alpha_prime(x, s)
    i = 0
    while (diff > 0 and i < 50):
        f_x = obj_fun.f_x(x)
        diff = f_alpha(alpha) - f_alpha(0) - c * alpha * g_alpha(0)
        alpha *= p
        i += 1
    return alpha


def _armijo(obj_fun, x, s, c=1e-4):

    alpha0 = 1.0
    amin = 0.0
    f_alpha = obj_fun.about_alpha(x, s)
    g_alpha = obj_fun.about_alpha_prime(x, s)

    if(f_alpha(alpha0) <= f_alpha(0) + c * alpha0 * g_alpha(0)):
        return alpha0

    alpha1 = -(g_alpha(0)) * alpha0**2 / \
            2.0 / (f_alpha(alpha0) - f_alpha(0) - g_alpha(0) * alpha0)
    if(f_alpha(alpha1) <= f_alpha(0) + c * alpha1 * g_alpha(0)):
        return alpha1

    while alpha1 > amin:

        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (f_alpha(alpha1) - f_alpha(0) - g_alpha(0) * alpha1) - \
            alpha1**2 * (f_alpha(alpha0) - f_alpha(0) - g_alpha(0) * alpha0)
        a = a / factor

        b = -alpha0**3 * (f_alpha(alpha1) - f_alpha(0) - g_alpha(0) * alpha1)+ \
            alpha1**3 * (f_alpha(alpha0) - f_alpha(0) - g_alpha(0) * alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * g_alpha(0)))) / (3.0 * a)
        if(f_alpha(alpha2) <= f_alpha(0) + c * alpha2 * g_alpha(0)):
            return alpha2

        if(alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2

    return 0.001
