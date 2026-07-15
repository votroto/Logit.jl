include("path.jl")

function unilateral_deviations_simple!(
    result::NTuple{N,<:AbstractVector},
    payoffs::NTuple{N,<:AbstractArray},
    x::NTuple{N,<:AbstractVector}
) where {N}
    for i in CartesianIndices(first(payoffs))
        @inbounds w = prod(x[z][i[z]] for z in 1:N)
        @simd for p in 1:N
            @inbounds result[p][i[p]] += (w/x[p][i[p]]) * payoffs[p][i]
        end
    end
    result
end

function H(pi, lambda, ubar, i, j)
    K = eachindex(pi[i])
    m = lambda * maximum(ubar[i])
    exp(lambda*ubar[i][j] - m) - pi[i][j]*sum(exp(lambda*ubar[i][k] - m) for k in K)
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function H(x, lambda, u)
    T = promote_type(eltype(x), typeof(lambda))

    pi = splitviews(x, size(first(u)))

    ubar = ntuple(i -> zeros(T, size(u[i], i)), length(u))
    unilateral_deviations_simple!(ubar, u, pi)

    h = zeros(T, length(x))

    idx = 1
    for p in eachindex(u)
        for a in axes(u[p], p)
            h[idx] = H(pi, lambda, ubar, p, a)
            idx+=1
        end
    end

    h
end

A::Matrix{Float64} = [5 6; 7 8]
B::Matrix{Float64} = [1 2; 3 4]
x::Vector{Float64} = [1e-8,1.0,1.0,1e-8]

S(x,lam) = H(x, lam, (A,B))

guess = [0.5,0.5,0.5,0.5]

x1 = hc(guess, 0.0, 10.0, S)
