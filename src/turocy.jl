

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



function jacobian_l(x, lam, u)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    J = [ubar[p][end] - ubar[p][a]
         for p in eachindex(u)
         for a in eachindex(mu[p])]
end

function residual(mu, lambda, ubar, i, j)
    mu[i][j] - lambda*(ubar[i][j] - ubar[i][end])
end

function residual(x, lambda, u)
    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    ubar = unilateral_deviations_simple(u, pi)

    [residual(mu, lambda, ubar, p, a)
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
    pi::NTuple{N,Vector{F}}
) where {N,F}

    # result[p][q] holds: ∂ū_p[j] / ∂π_q[m]
    @inbounds @fastmath for i in CartesianIndices(first(payoffs))
        # 1. Hoist probability lookups
        probs = ntuple(z -> pi[z][i[z]], Val(N))

        for p in 1:N
            pay_p = payoffs[p][i]
            ip = i[p] # hoist indexing

            for q in 1:N
                p == q && continue
                iq = i[q] # hoist indexing

                # 2. Branch-free multilinear product
                w_deriv = one(F)
                for z in 1:N
                    # If z is p or q, multiply by 1.0. Otherwise, multiply by the probability.
                    # This compiles to a highly efficient conditional move instruction.
                    w_deriv *= ifelse((z == p) | (z == q), one(F), probs[z])
                end

                result[p][q][ip, iq] += w_deriv * pay_p
            end
        end
    end
    return result
end




# Helper function to generate nested loops and hoist probabilities for derivatives
function build_deriv_loops(dims, idx, prev_p, p, q, N)
    d = dims[idx]
    var_ad = Symbol("a", d)

    if idx == length(dims)
        # Innermost loop
        u_args = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :pay_p, u_args...)

        # If dimension is p or q, we don't multiply a probability
        p_term = if d == p || d == q
            prev_p
        else
            prev_p === nothing ? :(pi[$d][$var_ad]) : :(pi[$d][$var_ad] * $prev_p)
        end

        var_ap = Symbol("a", p)
        var_aq = Symbol("a", q)

        body = if p_term === nothing
            quote
                @inbounds res_pq[$var_ap, $var_aq] += $pay_idx
            end
        else
            quote
                @inbounds res_pq[$var_ap, $var_aq] += $pay_idx * $p_term
            end
        end

        return quote
            @simd for $var_ad in 1:size(pay_p, $d)
                $body
            end
        end
    else
        # Not innermost
        if d == p || d == q
            # Skip probability multiplication for this dimension, just pass state down
            inner_loop = build_deriv_loops(dims, idx + 1, prev_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $inner_loop
                end
            end
        else
            new_p = Symbol("p", d)
            p_expr = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            inner_loop = build_deriv_loops(dims, idx + 1, new_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end



"""
    unilateral_derivatives3!(result, payoffs, pi)

Fully unrolled and hoisted derivative kernel.
Resolves cache thrashing by pulling p and q out, and hoists probability products.
"""
@generated function unilateral_derivatives2!(
    result::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    exprs = []

    # Put p and q on the OUTSIDE to prevent cache thrashing on the result matrices
    for p in 1:N
        for q in 1:N
            p == q && continue

            # Loop dimensions N down to 1 (column-major friendly)
            dims = N:-1:1
            # Pass `nothing` instead of `:one`
            inner_loops_expr = build_deriv_loops(dims, 1, nothing, p, q, N)

            push!(exprs, quote
                res_pq = result[$p][$q]
                pay_p = payoffs[$p]
                $inner_loops_expr
            end)
        end
    end

    return Expr(:block, exprs...)
end

function jacobian_x(x, lam, u::NTuple{N}) where {N}
    # fw = ForwardDiff.jacobian(x -> system(x, lam), x)

    # d F_ij / d mu_lk
    J = zeros(length(x), length(x))

    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(u[p], p), size(u[p], q)), N), N)
    unilateral_derivatives2!(dudpi, u, pi)

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


function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
