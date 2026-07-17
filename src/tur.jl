include("path.jl")



function jac_l(x, lam, u, system)
    #    fw = ForwardDiff.derivative(lam -> system(x, lam), lam)

    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    J = [ubar[p][end] - ubar[p][a]
     for p in eachindex(u)
     for a in eachindex(mu[p])]


    #println("start")
  #  @assert (norm(fw - J)) <= 1e-9
    #println("stop")

    J
end

function point_to_strat(x)
    T = eltype(x)

    c = max(zero(T), maximum(x))

    ex = exp.(x .- c)
    ref = exp(-c)

    denom = ref + sum(ex)

    y = similar(x, length(x)+1)

    @inbounds for i in eachindex(x)
        y[i] = ex[i] / denom
    end

    y[end] = ref / denom

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
        @simd for p in 1:N
            @inbounds w = prod(x[z][i[z]] for z in 1:N if z != p)
            @inbounds result[p][i[p]] += (w) * payoffs[p][i]
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

    result = ntuple(p ->
            ntuple(q -> zeros(T, size(payoffs[p], p), size(payoffs[p], q)), N),
        N
    )

    for i in CartesianIndices(first(payoffs))
        for p in 1:N
            for q in 1:N
                p == q && continue

                # Compute the joint probability of all players EXCEPT p and q
                w_deriv = one(T)
                for z in 1:N
                    if z != p && z != q
                        @inbounds w_deriv *= x[z][i[z]]
                    end
                end

                @inbounds result[p][q][i[p], i[q]] += w_deriv * payoffs[p][i]
            end
        end
    end
    return result
end

function jac_x(x, lam, u, system)
  #  fw = ForwardDiff.jacobian(x -> system(x, lam), x)

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
#@assert (norm(fw - J)) <= 1e-9
    #println("stop")

    J
end


A = randn(20, 20)
B = randn(20, 20)

S(x, lam) = H(x, lam, (A, B))
Sl(x, lam) = jac_l(x, lam, (A, B), S)
Sx(x, lam) = jac_x(x, lam, (A, B), S)

guess_reduced = [strat_to_point(fill(1/size(A, 1), size(A, 1))); strat_to_point(fill(1/size(A, 2), size(A, 2)))]

x1 = hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)

@time x1 = hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)

mu = splitviews(x1, size(A) .- 1)
pi = point_to_strat.(mu)

println(round.(pi[1]; digits=5))
println(round.(pi[2]; digits=5))
nothing
