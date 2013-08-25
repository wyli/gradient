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
    plt.axis('equal')

    drawFunc.draw(obj.f_x)
    start_point = np.matrix([[-1.5], [1.0]])
    plt.annotate('Start', xy=(-1.5, 1.0), xytext=(-1.2, 1.2),
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Optimal', xy=(1, 1), xytext=(0.5, 1.2),
            arrowprops=dict(facecolor='black', shrink=0.05))

    #v = 0.0
    #str1 = "Newton"
    #track = opt.newton(start_point, obj, v)
    #p1, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
    #        'o-', linewidth=2.0)

    #v = 0.1
    #str2 = "Modified_Newton %.2f"%v
    #track = opt.newton(start_point, obj, v)
    #p2, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
    #        'o-', linewidth=2.0)

    #v = 1.5
    #str3 = "Modified_Newton %.2f"%v
    #track = opt.newton(start_point, obj, v, 50)
    #p3, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
    #        'o-', linewidth=2.0)

    #str4 = "Quasi_Newton_Dummy_line"
    #track = opt.quasi_newton(start_point, obj, 150, 1)
    #p4, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
    #        'o-', linewidth=2.0)

    str5 = "Quasi_Newton_Armijo"
    track = opt.quasi_newton(start_point, obj, 450, 0)
    p5, = plt.plot(track[0,:].tolist()[0], track[1,:].tolist()[0],
            'o-', linewidth=2.0)

    #plt.legend([p1, p2, p3, p4, p5], [str1, str2, str3, str4, str5])
    plt.show()

#quadratic()
rosen()
