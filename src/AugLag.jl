module AugLag

using LinearAlgebra

struct QuadraticModel
    f::Float64
    grad::Vector{Float64}
    hessian::Matrix{Float64}
end

function newton_direction(x::Vector{Float64}, qm::QuadraticModel)
    param_damping = 1.0 # Levenberg–Marquardt damping
    d = - (qm.hessian .+ param_damping)\qm.grad
    return d
end

function (qm::QuadraticModel)(x::Vector{Float64})
    val = x' * qm.hessian * x + dot(qm.grad, x)  + qm.f
    grad = 2 * qm.hessian * x + qm.grad
    return val, grad
end

struct Config
    xtol_internal::Float64
    sigma_alpha_update_minus::Float64
    sigma_alpha_update_plus::Float64
    sigma_line_search::Float64
end
function Config(;
        xtol_internal=1e-2,
        sigma_alpha_update_minus=0.5,
        sigma_alpha_update_plus=1.2,
        sigma_line_search=0.01)
    Config(xtol_internal, sigma_alpha_update_minus, sigma_alpha_update_plus, sigma_line_search)
end

mutable struct Workspace
    n_dim::Int
    n_ineq::Int
    n_eq::Int
    lambda::Vector{Float64} # g dual
    kappa::Vector{Float64} # h dual
    mu::Float64 # g penal
    nu::Float64 # h penal

    # cache
    fval::Union{Float64, Nothing}
    fgrad::Union{Vector{Float64}, Nothing}
    fhessian::Union{Matrix{Float64}, Nothing}
    gval::Union{Vector{Float64}, Nothing}
    gjac::Union{Matrix{Float64}, Nothing}
    hval::Union{Vector{Float64}, Nothing}
    hjac::Union{Matrix{Float64}, Nothing} 
end 

function Workspace(n_dim, n_ineq, n_eq) 
    g_dual = zeros(n_ineq) 
    h_dual = zeros(n_eq)
    g_penalty = 1.0
    h_penalty = 1.0
    Workspace(n_dim, n_ineq, n_eq, g_dual, h_dual, g_penalty, h_penalty, 
        nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

function evaluate!(ws::Workspace, x, f::QuadraticModel, g, h)
    ws.fval, ws.fgrad = f(x) 
    ws.gval, ws.gjac = g(x)
    ws.hval, ws.hjac = h(x)
    ws.fhessian = f.hessian
end

function compute_L(ws::Workspace)
    ineq_term = dot(ws.lambda, ws.gval) + dot(ws.mu * (ws.gval .> 0), ws.gval.^2)
    eq_term = dot(ws.kappa, ws.hval) + ws.nu * sum(ws.hval.^2)
    return ws.fval + ineq_term + eq_term
end

function compute_Lgrad(ws::Workspace)
    # NOTE : (Toussaint 2017) forgot including term ws.gval and ws.hval
    ineq_term = ws.gjac * (ws.lambda + 2 * ws.mu * ws.gval .* (ws.gval .> 0)) 
    eq_term = ws.hjac * (ws.kappa .+ (2 * ws.nu * ws.hval))
    return ws.fgrad + eq_term + ineq_term
end

function compute_approx_Lhessian(ws::Workspace)
    # NOTE : (Toussaint 2017) forgot multiplying 2
    ineq_term = 2 * ws.gjac * Diagonal(ws.mu * (ws.gval .> 0)) * ws.gjac'
    eq_term = 2 * ws.nu * ws.hjac * ws.hjac'
    return ws.fhessian + ineq_term + eq_term
end

function single_step!(ws::Workspace, x::Vector{Float64}, f, g, h, cfg::Config)

    alpha_newton = 1.0
    while true # LM newton method
        evaluate!(ws, x, f, g, h)
        L = compute_L(ws)
        Lgrad = compute_Lgrad(ws)
        Lhess = compute_approx_Lhessian(ws)
        qm = QuadraticModel(L, Lgrad, Lhess)
        direction = newton_direction(x, qm)
        
        function numerical_grad(x0)
            ws_ = deepcopy(ws)
            evaluate!(ws_, x0, f, g, h)
            L0 = compute_L(ws_)
            eps = 1e-7
            grad = zeros(ws_.n_dim)
            for i in 1:ws_.n_dim
                x1 = copy(x0)
                x1[i] += eps
                evaluate!(ws_, x1, f, g, h)
                L1 = compute_L(ws_)
                grad[i] = (L1 - L0)/eps
            end
            return grad
        end
        grad = numerical_grad(x)

        while true # line search
            x_new = x + direction * alpha_newton
            evaluate!(ws, x_new, f, g, h)
            L_new = compute_L(ws)
            isValidAlpha = L_new < L + cfg.sigma_line_search * dot(Lgrad, alpha_newton * direction)
            isValidAlpha && break
            alpha_newton *= cfg.sigma_alpha_update_minus
        end
        dx = alpha_newton * direction
        x += dx
        alpha_newton = min(cfg.sigma_alpha_update_plus * alpha_newton, 1.0)
        maximum(abs.(dx)) < cfg.xtol_internal && break
    end
    ws.lambda = min.(0, ws.lambda + 2 * ws.mu .* ws.gval) # NOTE : min because g > 0 in our case
    ws.kappa += 2 * ws.nu * ws.hval

    ws.mu < 1e8 && (ws.mu *= 10)
    ws.nu < 1e8 && (ws.nu *= 10)
    return x
end

export QuadraticModel, Workspace, Config, single_step!

end
