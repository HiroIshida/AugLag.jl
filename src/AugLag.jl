module AugLag

macro debugassert(test)
  esc(:(if $(@__MODULE__).debugging()
    @assert($test)
   end))
end
debugging() = false

using LinearAlgebra

struct QuadraticModel
    P::Matrix{Float64}
end

function (qm::QuadraticModel)(x::Vector{Float64})
    val = transpose(x) * qm.P * x
    grad = 2 * qm.P * x
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
    x::Vector{Float64} # current estimate for the solution
    lambda_eq::Vector{Float64}
    lambda_ineq::Vector{Float64}
    mu_eq::Vector{Float64}
    mu_ineq::Vector{Float64}
end

# we will consider a model with quadratic objective function with 
# general equality and inequality functions
struct Problem
    qm::QuadraticModel
    cineq::Function
    ceq::Function

    n_dim::Int
    n_dim_cineq::Int
    n_dim_ceq::Int
end

function gen_inner_data(prob::Problem)
end

function Problem(qm::QuadraticModel, cineq, ceq, n_dim)
    x_dummy = zeros(n_dim)
    val_ceq, jac_ceq = ceq(x_dummy)
    n_dim_ceq = length(val_ceq)

    @assert size(jac_ceq) == (n_dim, n_dim_ceq)
    val_cineq, jac_cineq = cineq(x_dummy)

    n_dim_cineq = length(val_cineq)
    @assert size(jac_cineq) == (n_dim, n_dim_cineq)

    Problem(qm, cineq, ceq, n_dim, n_dim_cineq, n_dim_ceq)
end

function update(prob::Problem, ad::AuglagData) 
    val_obj, grad_obj = prob.qm(ad.x)
    val_ceq, grad_ceq = prob.ceq(ad.x)
    val_cineq, grad_cineq = prob.cineq(ad.x)
    
    # compute function evaluation of the augmented lagrangian
    val_lag = val_obj
    for i in 1:length(ad.lambda_eq)
        val_lag -= ad.lambda_eq[i] * val_ceq[i] + 1.0/(2.0 * ad.mu_eq[i]) * val_ceq[i]^2
    end
    for i in 1:length(ad.lambda_ineq)
        val_lag += psi(val_cineq[i], ad.lambda_ineq[i], ad.mu_ineq[i])
    end

    # compute gradient of the augmented lagrangian
    grad_lag = grad_obj
    for i in 1:length(ad.lambda_eq)
        grad_lag -= ad.lambda_eq[i] * grad_ceq[:, i] + 1.0/(2.0 * ad.mu_eq[i]) * grad_ceq[:, i]^2
    end
    for i in 1:length(ad.lambda_eq)
        grad_lag += psi_grad(val_cineq[i], ad.lambda_ineq[i], ad.mu_ineq[i]) * grad_cineq[:, i]
    end
end

export QuadraticModel, AuglagData, Problem, update

end # module
