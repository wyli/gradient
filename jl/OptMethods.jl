module OptMethods
using TargetFunction

function Fletcher_Reeves(

    start_point, fun::TargetFunction.ObjFun, iteration=10, alpha=0.1)

    x = start_point
    track = x

    k = 1
    while k < iteration

        if k < 2

            g = fun.g_x(x)
            beta = 0.0
            s = -g
        else

            g = fun.g_x(x)
            m_old = (g_old'*g_old)[1]
            m_new = (g'*g)[1]
            beta = m_new / m_old
        end

        s = -g + beta * s
        alpha_k = _backtracking_line_search(fun, x, s)
        g_old = fun.g_x(x)
        x = x + alpha_k * s

        k += 1
        track = [track x]
    end
    return track
end
function _backtracking_line_search(fun::TargetFunction.ObjFun, x, s, c=1e-4)

    alpha = 1.0
    p = 0.6
    c = 0.0001
    diff = 1

    f_alpha = fun.about_alpha(x, s)
    g_alpha = fun.about_alpha_prime(x, s)
    f_alpha_0 = f_alpha(0.0)
    g_alpha_0 = g_alpha(0.0)

    i = 0

    while (diff > 0.0 && i < 1000)

        f_x = fun.f_x(x)
        diff = f_alpha(alpha) - f_alpha_0 - c * alpha * g_alpha_0
        diff = diff[1]
        alpha *= p
        i += 1
    end

    return alpha
end

end
