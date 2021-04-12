using Revise
using AugLag
using LinearAlgebra
using BenchmarkTools
using NLopt
import PyPlot

function eq_const(x::Vector{Float64})
    val = (x[1] - 1.0)^2 - x[2]
    grad = transpose([2 * (x[1] - 1.0) -1.0;])
    return [val], grad
    #return Vector{Float64}(undef, 0), Matrix{Float64}(undef, 2, 0)
end

function ineq_const(x::Vector{Float64})
    val = x[1] - 2
    grad = transpose([0 1;])
    return [val], grad
    #return Vector{Float64}(undef, 0), Matrix{Float64}(undef, 2, 0)
end

function debug_plot(ws::Workspace, b_min, b_max, f, g, h, x, Lgrad, newton_direction)
    xlin = range(b_min[1], b_max[1], length=100)
    ylin = range(b_min[2], b_max[2], length=100)
    function func(vec)
        AugLag.evaluate!(ws, vec, f, g, h)
        return AugLag.compute_L(ws)
    end
    fs = [func([x, y]) for x in xlin, y in ylin]
    PyPlot.contourf(xlin, ylin, fs')
    PyPlot.plt.colorbar()
    PyPlot.scatter([x[1]], [x[2]], c="red")
    g = -Lgrad
    PyPlot.scatter([x[1]+g[1]], [x[2]+g[2]], c="blue")
end

function main()
    qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))
    x = [2.0, 2.0]
    ws = Workspace(2, 1, 1)
    cfg = Config()
    try
        for i in 1:1
            x = single_step!(ws, x, qm, ineq_const, eq_const, cfg)
        end
    catch e
        if isa(e, AugLag.MaxLineSearchError)
            println("abort because hit the max line search limit")
            println(e.x)
            #fs = debug_plot(ws, [-0.5, -0.5], [1.5, 1.5], qm, ineq_const, eq_const,
            b_min = e.x .- 0.5
            b_max = e.x .+ 0.5
            fs = debug_plot(ws, b_min, b_max, qm, ineq_const, eq_const,
                           e.x, e.Lgrad, e.newton_direction)
            return fs 
        end
    end
end
main();
