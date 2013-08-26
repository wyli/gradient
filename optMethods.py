from numpy import *
import pdb

def newton(start_point, obj_fun, modified=0, iterations=5):
# with second order information
    x = start_point
    track = x

    k = 0
    while k < iterations:

        if modified > 0:
            vI = modified * matrix([[1, 0], [0, 1]])
            G_ = linalg.inv(obj_fun.G_x(x) + vI) # inverse?
        else:
            G_ = linalg.inv(obj_fun.G_x(x)) # inverse?
        g_ = obj_fun.g_x(x)
        delta = -1.0 * dot(G_, g_)
        x = x + delta

        track = concatenate((track, x), axis=1)
        k += 1

    return track

def BFGS(start_point, obj_fun, iteration=10, interpolation=0):
# with first order information
    x = start_point
    track = x

    k = 0
    #H = dot(obj_fun.g_x(x), obj_fun.g_x(x).T)
    H = 1.0*eye(2)
    gamma = 1.0

    while k < iteration and linalg.norm(gamma) > 1e-10:

        p = -dot(H, obj_fun.g_x(x))

        if interpolation > 0:
            alpha_k = _backtracking_line_search(obj_fun, x, p)
        else:
            alpha_k = _armijo_line_search(obj_fun, x, p)

        s = alpha_k * p
        g_k = obj_fun.g_x(x)
        x = x + alpha_k * p
        g_k_1 = obj_fun.g_x(x)

        y = g_k_1 - g_k

        z = dot(H, y)
        sTy = dot(s.T, y)
        if sTy > 0:
            H += outer(s, s) * (sTy + dot(y.T, z))[0,0]/(sTy**2) \
                    - (outer(z, s) + outer(s, z))/sTy

        track = concatenate((track, x), axis=1)
        k += 1

    return track

def quasi_newton(start_point, obj_fun, iteration=10, interpolation=0):
# with first order information
    x = start_point
    track = x

    k = 0
    #H = matrix([[.01, 0], [0, .01]])
    H = 1.0*eye(2)

    while k < iteration:

        s = -1.0 * dot(H, obj_fun.g_x(x))

        if interpolation > 0:
            alpha_k = _backtracking_line_search(obj_fun, x, s)
        else:
            alpha_k = _armijo_line_search(obj_fun, x, s)

        delta = alpha_k * s
        x_k_1 = x + delta

        gamma = obj_fun.g_x(x_k_1) - obj_fun.g_x(x)
        u = delta - dot(H, gamma)

        scale_a = dot(u.T, gamma)
        if scale_a == 0: # :(
            scale_a = 0.000001
        H = H + outer(u, u) / scale_a

        track = concatenate((track, x), axis=1)
        x = x_k_1
        k += 1

    return track

def fletcher_reeves(start_point, obj_fun, iteration=10, alpha=0.1):
    x = start_point
    track = x

    k = 0
    while k < iteration:

        if k < 1:
            g = obj_fun.g_x(x)
            beta = 0.0
            s = -g
        else:
            g = obj_fun.g_x(x)

            beta = dot(g.T, g)/dot(g_old.T, g_old)
            beta = beta[0,0]

        s = -g + beta * s
        alpha_k = _armijo_line_search(obj_fun, x, s, alpha0=alpha)
        g_old = obj_fun.g_x(x)
        x = x + alpha_k * s

        track = concatenate((track, x), axis=1)
        k += 1

    return track

def _backtracking_line_search(obj_fun, x, s, c=1e-4):
# dummy
    alpha = 1.0
    p = 0.5 # magic number
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


def _armijo_line_search(obj_fun, x, s, c=1e-4, alpha0=1.0):
# adapted from scipy.optimisation
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

        alpha2 = (-b + sqrt(abs(b**2 - 3 * a * g_alpha(0)))) / (3.0 * a)
        if(f_alpha(alpha2) <= f_alpha(0) + c * alpha2 * g_alpha(0)):
            return alpha2

        if(alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2

    return 0.0001
