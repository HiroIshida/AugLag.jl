using AugLag
using BenchmarkTools
using LinearAlgebra
using Test
import JSON

AugLag.debugging() = true

# the test data set (mpc.json) is created using 
# https://github.com/HiroIshida/robust-tube-mpc
f = open("../data/mpc.json", "r")
qpdata = JSON.parse(read(f, String))
close(f)

function parse_json_matrix(json)
    n = length(json)
    m = length(json[1])
    mat = zeros(Float64, n, m)
    for i in 1:n
        mat[i, :] = json[i]
    end
    return mat
end

C_ineq1 = parse_json_matrix(qpdata["C_ineq1"])
C_ineq2 = Vector{Float64}(qpdata["C_ineq2"])
C_eq1 = parse_json_matrix(qpdata["C_eq1"])
C_eq2 = Vector{Float64}(qpdata["C_eq2"])
H = parse_json_matrix(qpdata["H"])
sol = Vector{Float64}(qpdata["sol"])

using NLopt

function solve_mpc_nlopt(H, C_ineq1, C_ineq2, C_eq1, C_eq2, sol)

    dim = size(H)[1]
    qm = QuadraticModel(0.0, zeros(dim), H)

    function objective(x::Vector, grad::Vector)
        val, grad_ = qm(x)
        if length(grad) > 0
            grad[:] = grad_
        end
        return val
    end

    function eq_const(x::Vector, grad::Vector)
        println("hoge")
        val = C_eq2 - C_eq1 * x
        if length(grad) > 0
            println(grad)
            println("hage")
            grad[:] = - vec(C_eq1)
        end
        return val
    end

    function ineq_const(x::Vector, grad::Vector)
        val = C_ineq2 - C_ineq1 * x
        if length(grad) > 0
            grad = - C_ineq1
        end
        return val
    end

    x_opt = ones(dim)
    opt = Opt(:LD_SLSQP, dim)
    opt.min_objective = objective
    #opt.inequality_constraint = ineq_const
    #opt.equality_constraint = eq_const
    opt.xtol_abs = 1e-3
    optimize(opt, x_opt)
end

function solve_mpc(H, C_ineq1, C_ineq2, C_eq1, C_eq2, sol)

    dim = size(H)[1]
    qm = QuadraticModel(0.0, zeros(dim), H)

    function eq_const(x::Vector{Float64})
        val = C_eq2 - C_eq1 * x
        grad = - C_eq1
        return val, transpose(grad)
    end

    function ineq_const(x::Vector{Float64})
        val = C_ineq2 - C_ineq1 * x
        grad = - C_ineq1
        return val, transpose(grad)
    end

    prob = Problem(qm, eq_const, ineq_const, dim)
    x_opt = zeros(dim)
    internal_data = gen_init_data(prob)

    xtol = 1e-6
    while true
        x_opt_pre = x_opt
        x_opt = step_auglag(x_opt, prob, internal_data, xtol)
        if maximum(abs.(x_opt - x_opt_pre)) < xtol
            println("converged")
            break
        end
    end
    @test maximum(abs.(x_opt - sol)) < 1e-4
    println("mpc solved")
end

@benchmark solve_mpc(H, C_ineq1, C_ineq2, C_eq1, C_eq2, sol)
@benchmark solve_mpc_nlopt(H, C_ineq1, C_ineq2, C_eq1, C_eq2, sol)
