@testset "simple" begin
    function eq_const(x::Vector{Float64})
        val = (x[1] - 1.0)^2 - x[2]
        grad = transpose([2 * (x[1] - 1.0) -1.0;])
        return [val], grad
    end

    function ineq_const(x::Vector{Float64})
        val = x[1] - 2
        grad = transpose([1 0;])
        return [val], grad
    end

    qm = QuadraticModel(0.0, [0, 0], Diagonal([1, 1.]))
    x = [2.0, 2.0]
    ws = Workspace(2, 1, 1)
    cfg = Config()
    for i in 1:100
        x = single_step!(ws, x, qm, ineq_const, eq_const, cfg)
        shoud_abort(ws, cfg) && break
    end
    @test isapprox(x, [2.0, 1.0], atol=1e-3)
end
