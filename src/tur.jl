include("path.jl")



function jac_l(x, lam, u)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    [ubar[p][end] - ubar[p][a]
     for p in eachindex(u)
     for a in eachindex(mu[p])]
end

function point_to_strat(x)
    c = maximum(x)
    ex = exp.(x .- c)
    denom = exp(-c) + sum(ex)
    y = similar(x, length(x)+1)
    for i in eachindex(x)
        y[i] = ex[i] / denom
    end
    y[end] = exp(-c) / denom
    return y
end

function strat_to_point(y)
    x = Vector{eltype(y)}(undef, length(y)-1)
    for i in eachindex(x)
        x[i] = log(y[i] / y[end])
    end
    x
end

function unilateral_deviations_simple(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    x::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    T = promote_type(P, X)

    result = ntuple(i -> zeros(T, size(payoffs[i], i)), N)
    for i in CartesianIndices(first(payoffs))
        @inbounds w = prod(x[z][i[z]] for z in 1:N)
        @simd for p in 1:N
            @inbounds result[p][i[p]] += (w/x[p][i[p]]) * payoffs[p][i]
        end
    end
    result
end

function H(mu, lambda, ubar, i, j)
    mu[i][j] - lambda*(ubar[i][j] - ubar[i][end])
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function H(x, lambda, u)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    [H(mu, lambda, ubar, p, a)
     for p in eachindex(u)
     for a in eachindex(mu[p])]
end


function unilateral_derivatives(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    x::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    T = promote_type(P, X)

    # result[p][q] will hold a matrix of size (size_p, size_q)
    # representing ∂ū_p[j] / ∂π_q[m]
    result = ntuple(p ->
        ntuple(q -> zeros(T, size(payoffs[p], p), size(payoffs[p], q)), N),
        N
    )

    for i in CartesianIndices(first(payoffs))
        # Complete joint probability product
        @inbounds w = prod(x[z][i[z]] for z in 1:N)

        # We need to compute the derivative of player p's expected payoff
        # with respect to player q's strategy
        for p in 1:N
            for q in 1:N
                p == q && continue  # ∂ū_p / ∂π_p is always 0

                # Divide out both player p and player q's probabilities
                @inbounds w_deriv = w / (x[p][i[p]] * x[q][i[q]])

                # Accumulate into the (action_p, action_q) slot
                @inbounds result[p][q][i[p], i[q]] += w_deriv * payoffs[p][i]
            end
        end
    end
    return result
end

function jac_x(x, lam, u, system)
    #fw = ForwardDiff.jacobian(x -> system(x, lam), x)

    # d F_ij / d mu_lk
    J = zeros(length(x), length(x))

    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    dudpi = unilateral_derivatives(u, pi)

    # result[p][q] will hold a matrix of size (size_p, size_q)
    # representing ∂ū_p[j] / ∂π_q[m]

    ij = 1
    for i in eachindex(u)
        for j in eachindex(mu[i])
            lm = 1
            for l in eachindex(u)
                for m in eachindex(mu[l])
                    if l == i # same player
                        J[ij, lm] = (j == m) ? 1 : 0
                    else
                        J[ij, lm] = -lam * sum((dudpi[i][l][j,r] - dudpi[i][l][end,r])  * pi[l][r] * (((r == m) ? 1 : 0) - pi[l][m]) for r in eachindex(pi[l]))
                    end
                    lm += 1
                end
            end
            ij +=1
        end
    end

    #println("start")
    #display(norm(fw - J))
    #println("stop")

    J
end


A::Matrix{Float64} = [0.0 1.0 4.0; 1.0 0.0 1.0; 4.0 1.0 0.0]
B::Matrix{Float64} = -[0.0 1.0 4.0; 1.0 0.0 1.0; 4.0 1.0 0.0]

A = randn(20,20)
B = randn(20,20)

S(x, lam) = H(x, lam, (A, B))
Sl(x, lam) = jac_l(x, lam, (A, B))
Sx(x, lam) = jac_x(x, lam, (A, B), S)

guess_reduced = [strat_to_point(fill(1/size(A,1),size(A,1))); strat_to_point(fill(1/size(A,2),size(A,2)))]

x1 = hc(guess_reduced, 0.0, 100.0, S, Sl, Sx)

mu = splitviews(x1, size(A) .- 1)
pi = point_to_strat.(mu)

