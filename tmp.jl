using Revise
using Aula
using LinearAlgebra
using Test
import JSON

# the test data set (mpc.json) is created using 
# https://github.com/HiroIshida/robust-tube-mpc
f = open("./data/mpc.json", "r")
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

x = zeros(dim)

function main(x, dim, C_eq2, C_ineq2, qm, ineq_const, eq_const)
    ws = Workspace(dim, length(C_ineq2), length(C_eq2))
    cfg = Config()
    for i in 1:20
        x = single_step!(ws, x, qm, ineq_const, eq_const, cfg)
        shoud_abort(ws, cfg) && break
    end
end
using BenchmarkTools
using Profile
@profile main(x, dim, C_eq2, C_ineq2, qm, ineq_const, eq_const)
Profile.print()
