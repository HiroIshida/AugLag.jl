
abstract type RealValuedFunction <: Function end
abstract type VectorValuedFunction <: Function end

struct QuadraticModel <: RealValuedFunction
    f::Float64
    grad::Vector{Float64}
    hessian::Matrix{Float64}
end

function (qm::QuadraticModel)(x::Vector{Float64})
    val = x' * qm.hessian * x + dot(qm.grad, x)  + qm.f
    grad = 2 * qm.hessian * x + qm.grad
    return val, grad
end
hessian(qm::QuadraticModel) = qm.hessian

"""
struct VectorValuedNoHess <: VectorValuedFunction
    f::Function
end
function (fmodel::VectorValuedNoHess)(x::Vector{Float64})
    val, jac = fmodel.f(x)
    return val, jac
end
function hessians(fmodel::VectorValuedNoHess)(x0::Vector{Float64})
    _, jac0 = fmodel.f(x0)
    n, m = size(jac0) # m is constraint number
    eps = 1e-7

    hessians = Matrix[] # TODO inefficient because we already know the size of arr

    for i in 1:m
        # TODO this implementation is quite inefficient 
        # because we compute duplicate jacobian multiple times
        hessian = zeros(n, n)
        for j in 1:n
            x1 = copy(x0)
            x1[j] += eps
            _, jac1 = fmodel.f(x1)
            hessian[:, j] = (jac1[:, i] - jac0[:, i])/eps
        end
        push!(hessians, hessian)
    end
    return hessians
end

"""
