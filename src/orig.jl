


using LinearAlgebra
using ForwardDiff

"""
    logit_strategy_solve(payoffs::NTuple{N, Array{Float64, N}};
                         lambda_max=100.0, steps=100, max_regret=1e-5)

Reimplements Gambit's logit solver.
Uses an unconstrained log-odds space transformation and terminates early
if the maximum regret drops below `max_regret`.
"""
function logit_strategy_solve(payoffs::NTuple{N,Array{Float64,N}};
    lambda_max::Float64=1000.0, steps::Int=1000,
    max_regret::Float64=1e-6) where N

    strat_sizes = [size(payoffs[1])...]
    free_dims = [s - 1 for s in strat_sizes]
    total_free_dim = sum(free_dims)

    # Precompute offsets for 1D indexing to avoid allocations in the hot loop
    offsets = [0; cumsum(strat_sizes)[1:(end-1)]]

    function weights_to_probabilities(y::AbstractVector{T}, λ::Float64) where T
        # Allocate exactly one flat array per call
        p_flat = Vector{T}(undef, sum(strat_sizes))
        idx_y = 1

        for i in 1:N
            si = strat_sizes[i]
            fi = free_dims[i]

            # Find max without allocating a slice
            max_w = zero(T)
            if fi > 0
                for k in 1:fi
                    max_w = max(max_w, y[idx_y+k-1])
                end
            end

            # Base strategy anchored at 0
            sum_exp = zero(T)
            exp_0 = exp(-max_w)
            p_flat[offsets[i]+1] = exp_0
            sum_exp += exp_0

            # Remaining strategies
            if fi > 0
                for k in 1:fi
                    val = exp(y[idx_y+k-1] - max_w)
                    p_flat[offsets[i]+1+k] = val
                    sum_exp += val
                end
                idx_y += fi
            end

            # Normalize in-place
            for k in 1:si
                p_flat[offsets[i]+k] /= sum_exp
            end
        end
        return p_flat
    end

    function compute_expected_payoffs(p::AbstractVector{T}) where T
        # Allocate exactly one flat array per call
        u_flat = zeros(T, sum(strat_sizes))

        for I in CartesianIndices(payoffs[1])
            for i in 1:N
                prob_others = one(T)
                for j in 1:N
                    if j != i
                        prob_others *= p[offsets[j]+I[j]]
                    end
                end
                u_flat[offsets[i]+I[i]] += payoffs[i][I] * prob_others
            end
        end
        return u_flat
    end

    # Pass 'res' as the first argument to mutate it in place
    function residual!(res::AbstractVector{T}, y::AbstractVector{T}, λ) where T
        p = weights_to_probabilities(y, λ)
        u = compute_expected_payoffs(p)

        idx_y = 1
        for i in 1:N
            fi = free_dims[i]
            if fi > 0
                u_1 = u[offsets[i]+1]
                for k in 1:fi
                    u_k = u[offsets[i]+1+k]
                    res[idx_y+k-1] = y[idx_y+k-1] - λ * (u_k - u_1)
                end
                idx_y += fi
            end
        end
        return nothing
    end
    # --- Initialization ---
    y = zeros(total_free_dim)
    λ = 0.0
    dt = lambda_max / steps

    # Preallocate memory for ForwardDiff
    R_cache = zeros(total_free_dim)
    J_cache = zeros(total_free_dim, total_free_dim)

    # Create the ForwardDiff tape/config exactly ONCE
    cfg = ForwardDiff.JacobianConfig(nothing, R_cache, y)

    # --- Predictor-Corrector Homotopy Loop ---
    for step in 1:steps
        λ_target = step * dt

        # Corrector Loop
        for iter in 1:15
            # Manually calculate current residual into R_cache
            residual!(R_cache, y, λ_target)
            if norm(R_cache) < 1e-7
                break
            end

            # Compute Jacobian strictly in-place
            ForwardDiff.jacobian!(J_cache,
                (r, vars) -> residual!(r, vars, λ_target),
                R_cache, y, cfg)

            dy = try
                J_cache \ -R_cache
            catch
                break
            end
            y .+= dy
        end

        λ = λ_target

        # --- MAXIMUM REGRET CHECK ---
        current_p = weights_to_probabilities(y, λ)
        current_u = compute_expected_payoffs(current_p)

        current_max_regret = 0.0
        for i in 1:N
            p_i = current_p[(offsets[i]+1):(offsets[i]+strat_sizes[i])]
            u_i = current_u[(offsets[i]+1):(offsets[i]+strat_sizes[i])]

            best_response_payoff = maximum(u_i)
            expected_payoff = sum(p_i .* u_i)

            player_regret = best_response_payoff - expected_payoff
            current_max_regret = max(current_max_regret, player_regret)
        end

        if current_max_regret <= max_regret
            break
        end

        if !all(isfinite, y)
            break
        end
    end

    # --- Clean Outputs ---
    final_p = weights_to_probabilities(y, λ)
    result_profiles = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        si = strat_sizes[i]
        result_profiles[i] = round.(final_p[(offsets[i]+1):(offsets[i]+si)], digits=6)
    end

    return Tuple(result_profiles)
end
