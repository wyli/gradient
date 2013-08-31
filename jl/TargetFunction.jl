module TargetFunction

immutable ObjFun
    name::ASCIIString
    f_x::Function
    g_x::Function
    about_alpha::Function
    about_alpha_prime::Function
end


function f_x(x::Array{Float64, 1})
    f = 10.0 * (x[2] - x[1]*x[1])*(x[2] - x[1]*x[1]) + (1 - x[1])*(1 - x[1])
    return f
end

function g_x(x::Array{Float64, 1})
    g1 = -40.0 * (x[1]*x[2] - x[1]*x[1]*x[1]) - 2.0 + 2.0 * x[1]
    g2 = 20.0 * (x[2] - x[1]*x[1])
    return [g1; g2]
end

function about_alpha(x::Array{Float64, 1}, s::Array{Float64, 1})
    function along_s(alpha::Float64)
        return f_x(x + s*alpha)
    end
    return along_s
end

function about_alpha_prime(x::Array{Float64, 1}, s::Array{Float64, 1})
    function along_s(alpha::Float64)
        return g_x(x + alpha * s)' * s
    end
    return along_s
end

rosenbrock = ObjFun("Rosenbrock", f_x, g_x, about_alpha, about_alpha_prime)

end
