module DrawFunc

using TargetFunction
using PyCall
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib.cm as cm

function draw2D(fun::TargetFunction.ObjFun)

    dx = [-3.0:.02:2.5]
    dy = [-4.5:.02:4.5]'

    m, n = length(dx), length(dy)
    dx = repmat(dx, 1, n)[:]
    dy = repmat(dy, m, 1)[:]
    z = [fun.f_x([i;j]) for (i, j) in zip(dx, dy)]

    dx, dy, z = reshape(dx, m, n), reshape(dy, m, n), reshape(z, m, n)
    cs = plt.contourf(dx, dy, z, 20, cmap=cm.get_cmap("YlGnBu"))
end

end
