module AugLag

macro debugassert(test)
  esc(:(if $(@__MODULE__).debugging()
    @assert($test)
   end))
end
debugging() = false

using LinearAlgebra

function numerical_grad(fun, x0)
    eps = 1e-7
    dim = length(x0)
    grad_numel = zeros(dim)
    for i in 1:dim
        x1 = copy(x0)
        x1[i] += eps
        grad_numel[i] = (fun(x1) - fun(x0))/eps
    end
    return grad_numel
end

struct QuadraticModel
    f::Float64
    grad::Vector{Float64}
    hessian::Matrix{Float64}
end

function newton_direction(x::Vector{Float64}, qm::QuadraticModel)
    alpha = 1.0
    param_damping = 1.0 # Levenberg–Marquardt damping
    inv_hessian = inv(qm.hessian .+ param_damping)
    d = - inv_hessian * qm.grad
end

function (qm::QuadraticModel)(x::Vector{Float64})
    val = transpose(x) * qm.hessian * x + dot(qm.grad, x)  + qm.f
    grad = 2 * qm.hessian * x + qm.grad
    return val, grad
end

function psi(t::Float64, sigma::Float64, mu::Float64)
    if t - mu * sigma < 0.0
        val = - sigma * t + 1/(2*mu) * t^2
        return val
    else
        val = -0.5 * mu * sigma
        return val
    end
end

function psi_grad(t::Float64, sigma::Float64, mu::Float64)
    if t - mu * sigma < 0.0
        grad = - sigma + 1/mu * t
        return grad
    else
        return 0.0
    end
end

mutable struct AuglagData
    lambda_ceq::Vector{Float64}
    lambda_cineq::Vector{Float64}
    mu_ceq::Float64
    mu_cineq::Float64
end

# we will consider a model with quadratic objective function with 
# general equality and inequality functions
struct Problem
    qm::QuadraticModel
    ceq::Function
    cineq::Function

    n_dim::Int
    n_dim_ceq::Int
    n_dim_cineq::Int
end

function gen_init_data(prob::Problem)
    lambda_ceq = zeros(prob.n_dim_ceq)
    lambda_cineq = zeros(prob.n_dim_cineq)
    mu_ceq = 1.0
    mu_cineq = 1.0
    AuglagData(lambda_ceq, lambda_cineq, mu_ceq, mu_cineq)
end


struct FuncEvals
    val_obj::Float64
    grad_obj::Vector{Float64}
    val_ceq::Vector{Float64}
    grad_ceq::Matrix{Float64}
    val_cineq::Vector{Float64}
    grad_cineq::Matrix{Float64}
end

function FuncEvals(x::Vector{Float64}, prob::Problem)
    val_obj, grad_obj = prob.qm(x)
    val_ceq, grad_ceq = prob.ceq(x)
    val_cineq, grad_cineq = prob.cineq(x)
    FuncEvals(val_obj, grad_obj, val_ceq, grad_ceq, val_cineq, grad_cineq)
end

function Problem(qm::QuadraticModel, ceq, cineq, n_dim)
    x_dummy = zeros(n_dim)
    val_ceq, jac_ceq = ceq(x_dummy)
    n_dim_ceq = length(val_ceq)

    @assert size(jac_ceq) == (n_dim, n_dim_ceq)
    val_cineq, jac_cineq = cineq(x_dummy)

    n_dim_cineq = length(val_cineq)
    @assert size(jac_cineq) == (n_dim, n_dim_cineq)

    Problem(qm, ceq, cineq, n_dim, n_dim_ceq, n_dim_cineq)
end

function compute_auglag(prob::Problem, ad::AuglagData, fe::FuncEvals) 
    # compute function evaluation of the augmented lagrangian
    val_lag = fe.val_obj
    for i in 1:prob.n_dim_ceq
        ineq_lag_mult = ad.lambda_ceq[i] * fe.val_ceq[i]
        ineq_quadratic_penalty = ad.mu_ceq/2.0 * fe.val_ceq[i]^2
        val_lag += (- ineq_lag_mult + ineq_quadratic_penalty)
    end
    for i in 1:prob.n_dim_cineq
        val_lag += psi(fe.val_cineq[i], ad.lambda_cineq[i], ad.mu_cineq)
    end
    return val_lag
end

function compute_auglag_grad(prob::Problem, ad::AuglagData, fe::FuncEvals)
    # compute gradient of the augmented lagrangian
    grad_lag = fe.grad_obj
    for i in 1:prob.n_dim_ceq
        grad_ineq_lag_mult = ad.lambda_ceq[i] * fe.grad_ceq[:, i]
        grad_ineq_quadratic_penalty = ad.mu_ceq/2.0 * 2 * fe.val_ceq[i] * fe.grad_ceq[:, i]
        grad_lag += (- grad_ineq_lag_mult + grad_ineq_quadratic_penalty)
    end
    for i in 1:prob.n_dim_cineq
        grad_lag += psi_grad(fe.val_cineq[i], ad.lambda_cineq[i], ad.mu_cineq) * fe.grad_cineq[:, i]
    end
    return grad_lag
end

function compute_auglag_hessian(prob::Problem, ad::AuglagData, fe::FuncEvals)
    approx_hessian = prob.qm.hessian
    for i in 1:prob.n_dim_ceq
        approx_hessian += 0.5 * ad.mu_ceq * fe.grad_ceq[:, i] * transpose(fe.grad_ceq[:, i])
    end
    for i in 1:prob.n_dim_cineq
        approx_hessian += 0.5 * ad.mu_cineq * fe.grad_cineq[:, i] * transpose(fe.grad_cineq[:, i]) 
    end
    return approx_hessian
end

function step_auglag(x::Vector{Float64}, prob::Problem, ad::AuglagData)
    alpha_newton = 1.0
    for _ in 1:100
        fe = FuncEvals(x, prob)
        # newton step
        val_lag = compute_auglag(prob, ad, fe)
        grad_lag = compute_auglag_grad(prob, ad, fe)
        hessian_lag = compute_auglag_hessian(prob, ad, fe)

        # compute approx hessian
        qm = QuadraticModel(val_lag, grad_lag, hessian_lag)
        direction = newton_direction(x, qm)
        x += direction * 0.01
    end

    val_obj, grad_obj = prob.qm(x)
    val_ceq, grad_ceq = prob.ceq(x)
    val_cineq, grad_cineq = prob.cineq(x)
    for j in 1:prob.n_dim_ceq
        ad.lambda_ceq[j] -= ad.mu_ceq * val_ceq[j]
    end
    for j in 1:prob.n_dim_cineq
        ad.lambda_cineq[j] = max(0.0, ad.lambda_cineq[j] - ad.mu_cineq * val_cineq[j])
    end

    ad.mu_ceq *= 5.0
    ad.mu_cineq *= 5.0
    return x
end

export QuadraticModel, AuglagData, Problem, compute_auglag, gen_init_data, step_auglag

end # module
