include("path.jl")
using ForwardDiff
using BenchmarkTools




# Helper function to recursively generate the nested loops at compile-time.
# It nests loops such that the smallest index is innermost to preserve
# column-major memory access, while hoisting intermediate probability multiplications.
function build_loops(i, dims, idx, prev_p, N)
    d = dims[idx]
    var_ad = Symbol("a", d)

    if idx == length(dims)
        # Innermost loop
        u_args = [Symbol("a", k) for k in 1:N]
        u_idx = Expr(:ref, :U_i, u_args...)
        p_term = prev_p == :one ? :(pi[$d][$var_ad]) : :(pi[$d][$var_ad] * $prev_p)

        body = quote
            s += $u_idx * $p_term
        end

        return quote
            @simd for $var_ad in 1:size(U_i, $d)
                $body
            end
        end
    else
        new_p = Symbol("p", d)
        p_expr = prev_p == :one ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
        inner_loop = build_loops(i, dims, idx + 1, new_p, N)

        return quote
            for $var_ad in 1:size(U_i, $d)
                $new_p = $p_expr
                $inner_loop
            end
        end
    end
end

"""
    unilateral_deviations!(out, U, pi)

Computes the expected utility for each player `i` and each pure action `j`
under the mixed strategy profile `pi` without any heap allocations.
"""
@generated function unilateral_deviations!(
    out::NTuple{N,Vector{T}},
    U::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    if N == 1
        return quote
            @inbounds for a1 in 1:size(U[1], 1)
                out[1][a1] = U[1][a1]
            end
        end
    end

    exprs = []
    for i in 1:N
        # We loop over dimensions in descending order of memory access (N down to 1)
        # excluding the player's own dimension `i`, which is handled by the outermost loop.
        dims = filter(k -> k != i, N:-1:1)
        var_ai = Symbol("a", i)

        inner_loops_expr = build_loops(i, dims, 1, :one, N)

        push!(exprs, quote
            out_i = out[$i]
            U_i = U[$i]
            fill!(out_i, zero(T))
            @inbounds for $var_ai in 1:size(U_i, $i)
                s = zero(T)
                $inner_loops_expr
                out_i[$var_ai] = s
            end
        end)
    end

    return Expr(:block, exprs...)
end




function jac_l(x, lam, u, system)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    J = [ubar[p][end] - ubar[p][a]
         for p in eachindex(u)
         for a in eachindex(mu[p])]
end


function unilateral_deviations_simple(
    payoffs::NTuple{N,<:AbstractArray{F,N}},
    x::NTuple{N,<:AbstractVector{F}}
) where {N,F}
    result = ntuple(i -> zeros(F, size(payoffs[i], i)), Val(N))
    unilateral_deviations!(result, payoffs, x)
    #for i in CartesianIndices(first(payoffs))
    #    @simd for p in 1:N
    #        @inbounds w = prod(x[z][i[z]] for z in 1:N if z != p)
    #        @inbounds result[p][i[p]] += (w) * payoffs[p][i]
    #    end
    #end
    result
end

function H(mu, lambda, ubar, i, j)
    mu[i][j] - lambda*(ubar[i][j] - ubar[i][end])
end



function H(x, lambda, u)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    [H(mu, lambda, ubar, p, a)
     for p in eachindex(u)
     for a in eachindex(mu[p])]
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
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

function unilateral_derivatives!(
    result::NTuple{N,NTuple{N,Matrix{F}}},
    payoffs::NTuple{N,Array{F,N}},
    x::NTuple{N,Vector{F}}
) where {N,F}

    @inbounds @fastmath for i in CartesianIndices(first(payoffs))
        # 1. Hoist probability lookups into a CPU-register tuple.
        # Val(N) forces the compiler to completely unroll this step.
        probs = ntuple(z -> x[z][i[z]], Val(N))

        for p in 1:N
            # 2. Hoist the payoff lookup out of the `q` loop
            pay_p = payoffs[p][i]

            for q in 1:N
                p == q && continue

                w_deriv = one(F)
                for z in 1:N
                    # 3. Because N is a static type parameter, the Julia
                    # compiler will unroll this loop and evaluate this branch
                    # at compile time, eliminating branching overhead entirely.
                    if z != p && z != q
                        w_deriv *= probs[z]
                    end
                end

                result[p][q][i[p], i[q]] += w_deriv * pay_p
            end
        end
    end
    return result
end

function jac_x(x, lam, u::NTuple{N}, system) where {N}
    # fw = ForwardDiff.jacobian(x -> system(x, lam), x)

    # d F_ij / d mu_lk
    J = zeros(length(x), length(x))

    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(u[p], p), size(u[p], q)), N), N)
    unilateral_derivatives!(dudpi, u, pi)

    # result[p][q] holds:
    # ∂ū_p[j] / ∂π_q[m]

    ij = 1

    for i in eachindex(u)          # equation player
        for j in eachindex(mu[i])  # equation action (reference excluded)

            lm = 1

            for l in eachindex(u)  # differentiation player

                if l == i
                    # Own-player block:
                    # ∂(μ_ij - λ(u_ij-u_ref))/∂μ_il
                    # = identity because u does not depend on own strategy
                    for m in eachindex(mu[l])
                        J[ij, lm] = (j == m) ? 1.0 : 0.0
                        lm += 1
                    end

                else
                    c = 0
                    #dot(g, pi[l])
                    for m in eachindex(pi[l])
                        gm = (dudpi[i][l][j, m] - dudpi[i][l][end, m])
                        c += gm * pi[l][m]
                    end

                    for m in eachindex(mu[l])
                        gm = (dudpi[i][l][j, m] - dudpi[i][l][end, m])
                        J[ij, lm] = -lam * pi[l][m] * (gm - c)

                        lm += 1
                    end
                end
            end

            ij += 1
        end
    end


    J
end

using Random
Random.seed!(3462345634)

As = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

S(x, lam) = H(x, lam, As)
Sl(x, lam) = jac_l(x, lam, As, S)
Sx(x, lam) = jac_x(x, lam, As, S)

guess_reduced = [
    strat_to_point(normalize(ones(size(As[1], 1)), 1));
    strat_to_point(normalize(ones(size(As[1], 2)), 1));
    strat_to_point(normalize(ones(size(As[1], 3)), 1));
    strat_to_point(normalize(ones(size(As[1], 4)), 1));
    strat_to_point(normalize(ones(size(As[1], 5)), 1));
]

#hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)
x1 = hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)

@time x1 = hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)

mu = splitviews(x1, size(As[1]) .- 1)
pi = point_to_strat.(mu)

println(round.(pi[1]; digits=5))
println(round.(pi[2]; digits=5))
nothing



#=
using LinearAlgebra
using Gurobi
using JuMP


"""Computes the values and NE strategies for a general-sum game"""
function bilinear_program(us::NTuple{2, AbstractMatrix}, startx, starty, wstart; optimizer=Gurobi.Optimizer)
    m = Model(optimizer)
    #set_silent(m)
    nx, ny = size(us[1])
    @variable(m, xs[i=1:nx], lower_bound=0, upper_bound=1, start=startx[i])
    @variable(m, ys[i=1:ny], lower_bound=0, upper_bound=1, start=starty[i])
    @variable(m, w[i=1:2], lower_bound=minimum(us[i]), upper_bound=maximum(us[i]), start= wstart[i])

    @constraint(m, sum(xs) == 1)
    @constraint(m, sum(ys) == 1)

    @constraint(m, dot(xs, us[1], ys) + dot(xs, us[2], ys) >= sum(w))
    @constraint(m, (us[1] * ys)  .<= w[1])
    @constraint(m, (xs' * us[2]) .<= w[2])

    optimize!(m)

    value.(w), value.(xs), value.(ys), solve_time(m)
end

wstart = dot(pi[1],A,pi[2]), dot(pi[1],B,pi[2])

w,xsb,ysb,t = bilinear_program((A,B), pi[1], pi[2], wstart); nothing
=#

#[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
#[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
