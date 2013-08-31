using TargetFunction
using DrawFunc
using OptMethods
using PyCall
@pyimport matplotlib.pyplot as plt

plt.figure(TargetFunction.rosenbrock.name)
DrawFunc.draw2D(TargetFunction.rosenbrock)
plt.annotate("optimal", xy = (1.0, 1.0), xytext=(1.0, 1.0))


start_x = [-1.5; -4.0]
track = OptMethods.Fletcher_Reeves(start_x, TargetFunction.rosenbrock, 10) 
plt.plot(track[1,:]', track[2,:]', "o--")
println(track[:,end])
println("num_steps: ", size(track, 2))

plt.show()
