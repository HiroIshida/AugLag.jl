using Revise
using AugLag
using LinearAlgebra

AugLag.debugging() = false
qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))

function eq_const(x::Vector{Float64})
    val = (x[1] - 1.0)^2 - x[2]
    grad = transpose([2 * x[1] -1.0;])
    return val, grad
end

function ineq_const(x::Vector{Float64})
    val = x[1] - x[2]
    grad = transpose([1. -1.;])
    return val, grad
end

prob = Problem(qm, eq_const, ineq_const, 2, 1, 1)
x_init = [5.5, 5.0]
internal_data = gen_init_data(prob, x_init)
step_auglag(prob, internal_data)
