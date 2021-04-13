module Aula

using LinearAlgebra

include("funcmodels.jl")

function newton_direction(x::Vector{Float64}, qm::QuadraticModel)
    param_damping = 1.0 # Levenberg–Marquardt damping
    d = - (hessian(qm) .+ param_damping)\qm.grad
    return d
end

struct Config
    xtol_internal::Float64
    sigma_alpha_update_minus::Float64
    sigma_alpha_update_plus::Float64
    sigma_line_search::Float64
    mu_minimum::Float64
    dx_threshold::Float64
end
function Config(;
        xtol_internal=1e-2,
        sigma_alpha_update_minus=0.5,
        sigma_alpha_update_plus=1.2,
        sigma_line_search=0.01,
        mu_minimum=1e-6,
        dx_threshold=1e-6)
    Config(xtol_internal, sigma_alpha_update_minus, sigma_alpha_update_plus, sigma_line_search, 
        mu_minimum, dx_threshold)
end

mutable struct Workspace
    n_dim::Int
    n_ineq::Int
    n_eq::Int
    lambda_ineq::Vector{Float64} # g dual
    lambda_eq::Vector{Float64} # h dual
    mu::Float64 # g penal

    # cache
    dx_cache::Union{Vector{Float64}, Nothing}
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
    mu = 1.0
    Workspace(n_dim, n_ineq, n_eq, g_dual, h_dual, mu, 
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

function shoud_abort(ws::Workspace, cfg::Config)
    ws.mu < cfg.mu_minimum && (return true)
    norm(ws.dx_cache) < cfg.dx_threshold && (return true)
    return false
end

function evaluate!(ws::Workspace, x, f::QuadraticModel, g, h)
    ws.fval, ws.fgrad = f(x) 
    ws.gval, ws.gjac = g(x)
    ws.hval, ws.hjac = h(x)
    ws.fhessian = f.hessian
end

function psi(t::Float64, sigma::Float64, mu::Float64)
    if t - sigma/mu < 0.0
        return - sigma * t + 0.5 * t^2 / mu
    else
        return - 0.5 * mu * sigma^2
    end
end

function psi_grad(t::Float64, sigma::Float64, mu::Float64)
    if t - sigma/mu < 0.0
        return - sigma + t/mu
    else
        return 0.0
    end
end

function compute_L(ws::Workspace)
    ineq_term = sum([psi(t, sigma, ws.mu) for (t, sigma) in zip(ws.gval, ws.lambda_ineq)])
    eq_term = -dot(ws.lambda_eq, ws.hval) + sum(ws.hval.^2) / (2 * ws.mu)
    return ws.fval + eq_term  + ineq_term
end

function compute_Lgrad(ws::Workspace)
    ineq_term = ws.gjac * [psi_grad(t, sigma, ws.mu) for (t, sigma) in zip(ws.gval, ws.lambda_ineq)]
    eq_term = ws.hjac * (-ws.lambda_eq .+ ws.hval/ws.mu)
    return ws.fgrad + eq_term + ineq_term
end

function compute_approx_Lhessian(ws::Workspace)
    diag = Diagonal([(t-sigma/ws.mu < 0.0) * (1.0/ws.mu) for (t, sigma) in zip(ws.gval, ws.lambda_ineq)])
    ineq_term = ws.gjac * diag* ws.gjac'
    eq_term = ws.hjac * ws.hjac'/ws.mu
    return ws.fhessian + eq_term + ineq_term
end

function compute_exact_Lhessian(ws_::Workspace, x, f, g, h)
    eps = 1e-7
    ws = deepcopy(ws_) # to keep ws intact
    func(x) = (evaluate!(ws, x, f, g, h); compute_L(ws))

    function dfunc(func, x, i)
        dx = zeros(ws.n_dim)
        dx[i] = eps
        return (func(x + dx) - func(x))/eps
    end

    function ddfunc(func, x, i, j)
        dx = zeros(ws.n_dim)
        dx[i] = eps
        return dfunc((x)->dfunc(func, x, j), x, i)
    end

    hess = zeros(ws.n_dim, ws.n_dim)
    for i in ws.n_dim
        for j in ws.n_dim
            hess[i, j] = ddfunc(func, x, i, j)
        end
    end
    return hess
end

struct MaxLineSearchError <: Exception
    x::Vector{Float64}
    Lgrad::Vector{Float64}
    newton_direction::Vector{Float64}
end

function single_step!(ws::Workspace, x::Vector{Float64}, f, g, h, cfg::Config)
    x0 = copy(x)
    alpha_newton = 1.0
    while true # LM newton method
        evaluate!(ws, x, f, g, h)
        L = compute_L(ws)
        Lgrad = compute_Lgrad(ws)
        Lhess = compute_approx_Lhessian(ws)
        qm = QuadraticModel(L, Lgrad, Lhess)
        direction = newton_direction(x, qm)
        
        while true # line search
            x_new = x + direction * alpha_newton
            evaluate!(ws, x_new, f, g, h)
            L_new = compute_L(ws)
            extra = cfg.sigma_line_search * dot(Lgrad, alpha_newton * direction)
            isValidAlpha = L_new < L + extra
            isValidAlpha && break
            alpha_newton *= cfg.sigma_alpha_update_minus
        end
        dx = alpha_newton * direction
        x += dx
        alpha_newton = min(cfg.sigma_alpha_update_plus * alpha_newton, 1.0)
        norm(Lgrad) < 0.01 && break
    end
    ws.lambda_ineq = max.(0, ws.lambda_ineq - ws.gval/ws.mu)
    ws.lambda_eq = ws.lambda_eq .- ws.hval/ws.mu
    ws.mu > cfg.mu_minimum && (ws.mu *= 0.2) # Toussaint's rai sets 0.2
    ws.dx_cache = x - x0
    return x
end

export QuadraticModel, Workspace, Config, single_step!, shoud_abort

end
