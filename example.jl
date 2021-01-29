using Revise
using AugLag
using LinearAlgebra

AugLag.debugging() = true
qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))

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

function main()
    prob = Problem(qm, eq_const, ineq_const, 2)
    x_opt = [2.0, 2.0]
    internal_data = gen_init_data(prob)
    xtol = 1e-3
    @time for i in 1:20
        x_opt_pre = x_opt
        x_opt = step_auglag(x_opt, prob, internal_data, xtol)
        if maximum(abs.(x_opt - x_opt_pre)) < xtol
            break
        end
        println("xopt:")
        println(x_opt)
    end
end
main()
