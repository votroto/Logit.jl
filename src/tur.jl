function unilateral_deviations_simple!(
    result::NTuple{N,Vector{F}},
    payoffs::NTuple{N,Array{F,N}},
    x::NTuple{N,Vector{F}}
) where {N,F}
    for i in CartesianIndices(first(payoffs))
        @inbounds w = prod(x[z][i[z]] for z in 1:N)
        @simd for p in 1:N
            @inbounds result[p][i[p]] += (w/x[p][i[p]]) * payoffs[p][i]
        end
    end
    result
end


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



using BenchmarkTools

# 1. Setup game parameters
N = 4
S = (8, 8, 8, 8) # S_1, S_2, S_3, S_4

# 2. Pre-allocate arrays
pi_probs = ntuple(i -> rand(Float64, S[i]), N)
U = ntuple(i -> rand(Float64, S...), N)

out = ntuple(i -> zeros(Float64, S[i]), N)
unilateral_deviations_nogen!(out, U, pi_probs)

out = ntuple(i -> zeros(Float64, S[i]), N)
@benchmark unilateral_deviations_nogen!($out, U, pi_probs)