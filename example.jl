using Revise
using AugLag
using LinearAlgebra
using BenchmarkTools
using NLopt

AugLag.debugging() = true

function eq_const(x::Vector{Float64})
    val = (x[1] - 1.0)^2 - x[2]
    grad = transpose([2 * (x[1] - 1.0) -1.0;])
    return [val], grad
    #return Vector{Float64}(undef, 0), Matrix{Float64}(undef, 2, 0)
end

function ineq_const(x::Vector{Float64})
    val = (x[1] - 1.0) - x[2]
    grad = transpose([1. -1.;])
    return [val], grad
    #return Vector{Float64}(undef, 0), Matrix{Float64}(undef, 2, 0)
end

function nlopt_main()
    qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))

    function objective(x::Vector, grad::Vector)
        val, grad_ = qm(x)
        if length(grad) > 0
            grad[:] = grad_
        end
        return val
    end

    function eq_const(x::Vector, grad::Vector)
        val = (x[1] - 1.0)^2 - x[2]
        if length(grad) > 0
            grad[1] = - 2 * (x[1] - 1.0)
            grad[2] = 1.0
        end
        return - val
    end

    function ineq_const(x::Vector, grad::Vector)
        val = (x[1] - 1.0) - x[2]
        if length(grad) > 0
            grad[1] = -1.0
            grad[2] = 1.0
        end
        return -val
    end

    opt = Opt(:LD_SLSQP, 2)
    opt.min_objective = objective
    opt.inequality_constraint = ineq_const
    opt.equality_constraint = eq_const
    opt.xtol_abs = 1e-6
    minf, minx, ret = optimize(opt, [2.0, 2.0])
end

using Profile
function main()
    qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))
    prob = Problem(qm, eq_const, ineq_const, 2)
    x_opt = [2.0, 2.0]
    internal_data = gen_init_data(prob)
    xtol = 1e-6
    for i in 1:20
        x_opt_pre = x_opt
        x_opt = step_auglag(x_opt, prob, internal_data, xtol)
        if maximum(abs.(x_opt - x_opt_pre)) < xtol
            break
        end
    end
end
println("testing auglag")
@profile main()
open("profile.txt", "w") do io
    Profile.print(io, mincount=10)
end
#println("testing nlopt slsqp")
#@benchmark nlopt_main()
