
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

@generated function unilateral_derivatives!(
    result::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    exprs = []

    for p in 1:N
        for q in 1:N
            p == q && continue
            dims = N:-1:1
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

function build_deviation_loops(i, dims, idx, prev_p, N)
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
        inner_loop = build_deviation_loops(i, dims, idx + 1, new_p, N)

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

        inner_loops_expr = build_deviation_loops(i, dims, 1, :one, N)

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

function jacobian_t!(J, ubar, mu, u)
    idx = 1
    for p in eachindex(u)
        for a in eachindex(mu[p])
            J[idx] = ubar[p][end] - ubar[p][a]
            idx += 1
        end
    end
    J
end

function residual!(out, mu, ubar, x, lambda, u)
    idx = 1
    for p in eachindex(u)
        for a in eachindex(mu[p])
            out[idx] = mu[p][a] - lambda*(ubar[p][a] - ubar[p][end])
            idx += 1
        end
    end
    out
end

function jacobian_x!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_i = 1

    for eq_p in eachindex(u)
        for eq_a in 1:(size(u[eq_p], eq_p)-1)
            pd_i = 1

            for pd_p in eachindex(u)
                if pd_p == eq_p
                    # Own-player identity block
                    for pd_a in 1:(size(u[eq_p], eq_p)-1)
                        J[eq_i, pd_i] = (eq_a == pd_a)

                        pd_i += 1
                    end
                else
                    c = 0.0
                    for pd_a in eachindex(pi[pd_p])
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
                        c += gm * pi[pd_p][pd_a]
                    end

                    for pd_a in 1:(size(u[pd_p], pd_p)-1)
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
                        J[eq_i, pd_i] = -lam * pi[pd_p][pd_a] * (gm - c)

                        pd_i += 1
                    end
                end
            end

            eq_i += 1
        end
    end

    return J
end

function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
