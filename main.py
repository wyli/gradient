import optMethods as opt
import drawFunc
import func
import numpy as np
import matplotlib.pyplot as plt
import pdb

def quadratic():

    A = np.matrix([
        [3., 2.],
        [2., 6.]])
    b = np.matrix([[1.], [-5.]])
    obj = func.quadratic(A,b)

    plt.figure()
    drawFunc.draw(obj.f_x)

    x = np.matrix([[0], [2]])
    track = opt.newton(x, obj, 0, 1) # just one iteration
    p1, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0])

    track = opt.quasi_newton(x, obj, 100)
    p2, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    plt.legend([p1, p2], ['Newton', 'Quasi_Newton_Rank_One'])

    plt.show()

def rosen():
    obj = func.rosenbrock()

    plt.figure()

    drawFunc.draw(obj.f_x)
    start_point = np.matrix([[-1.5], [-4.0]])
    plt.annotate('Start', xy=(-1.5, -4.0), xytext=(-1.8, -3.5),
            arrowprops=dict(facecolor='black', shrink=0.02, frac=0.5))
    plt.annotate('Optimal', xy=(1, 1), xytext=(1.1, 1.4),
            arrowprops=dict(facecolor='black', shrink=0.02, frac=0.5))

    #v = 0.0
    #str1 = "Newton"
    #track = opt.newton(start_point, obj, v)
    #p1, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
    #        'o-', linewidth=2.0)

    v = 0.1
    str2 = "Modified_Newton %.2f"%v
    track = opt.newton(start_point, obj, v)
    p2, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    v = 1.5
    str3 = "Modified_Newton %.2f"%v
    track = opt.newton(start_point, obj, v, 50)
    p3, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    str4 = "Quasi_Newton_trivial"
    track = opt.quasi_newton(start_point, obj, 150, 1)
    p4, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    str5 = "Quasi_Newton_Armijo"
    track = opt.quasi_newton(start_point, obj, 450, 0)
    p5, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    str6 = "BFGS_trivial"
    track = opt.BFGS(start_point, obj, 350, 1)
    p6, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    str7 = "BFGS_Armijo"
    track = opt.BFGS(start_point, obj, 350, 0)
    p7, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    v = 0.001
    str8 = "CG_FR_%.5f"%v
    track = opt.fletcher_reeves(start_point, obj, iteration=30, alpha=v)
    p8, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    v = 0.01
    str9 = "CG_FR_%.5f"%v
    track = opt.fletcher_reeves(start_point, obj, iteration=30, alpha=v)
    p9, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)
    #plt.legend([p1, p2, p3, p4, p5, p6, p7],\
    #        [str1, str2, str3, str4, str5, str6, str7])

    plt.legend([p2, p3, p4, p5, p6, p7, p8, p9],\
            [str2, str3, str4, str5, str6, str7, str8, str9])
    plt.show()

#quadratic()
rosen()
