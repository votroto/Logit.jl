using LinearAlgebra

# --- Standard Game Utilities Helpers (Same as before but used natively) ---

function compute_expected_utilities(π, payoffs, player_indices)
    U = zeros(eltype(π), length(π))
    if length(player_indices) == 2
        r1, r2 = player_indices[1], player_indices[2]
        mul!(view(U, r1), payoffs[1], view(π, r2))
        mul!(view(U, r2), payoffs[2]', view(π, r1))
    end
    return U
end

function compute_utility_gradients(π, payoffs, player_indices)
    dU = [zeros(length(r), length(π)) for r in player_indices]
    if length(player_indices) == 2
        r1, r2 = player_indices[1], player_indices[2]
        dU[1][:, r2] .= payoffs[1]
        dU[2][:, r1] .= payoffs[2]'
    end
    return dU
end
"""
    turocy_system!(F, π, λ, payoffs, player_indices)

Computes Turocy's log-transformed QRE homotopy system.
- `π`: The strategy profile vector (probabilities directly).
- `player_indices`: Vector of ranges mapping each player's strategy block in `π`.
"""
function turocy_system!(F, π, λ, payoffs, player_indices)
    # 1. Compute expected utilities (Polynomial in π, no exponentials!)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]  # Action 1 as reference

        # Reference action row enforces the simplex constraint
        F[ref_idx] = sum(view(π, range)) - 1.0

        # Remaining actions enforce the log-utility condition
        for idx in range[2:end]
            F[idx] = log(π[idx]) - log(π[ref_idx]) - λ * (U[idx] - U[ref_idx])
        end
    end
    return F
end

"""
    turocy_jacobian!(J, π, λ, payoffs, player_indices)

Computes the exact analytical Jacobian ∂F/∂π for the log-transformed system.
"""
function turocy_jacobian!(J, π, λ, payoffs, player_indices)
    fill!(J, 0.0)
    dU = compute_utility_gradients(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]

        # 1. Derivatives for the simplex constraint row (j = 1)
        for idx_k in range
            J[ref_idx, idx_k] = 1.0
        end

        # 2. Derivatives for the log-utility rows (j > 1)
        for (local_j, idx_j) in enumerate(range[2:end])
            # Adjust local index because we skipped the first element
            j_idx_in_dU = local_j + 1

            # Derivative of the log terms: ∂/∂π
            J[idx_j, idx_j] += 1.0 / π[idx_j]
            J[idx_j, ref_idx] -= 1.0 / π[ref_idx]

            # Derivative of the utility terms across all game strategies k
            for idx_k in 1:length(π)
                ∂Utility = dU[i][j_idx_in_dU, idx_k] - dU[i][1, idx_k]
                J[idx_j, idx_k] -= λ * ∂Utility
            end
        end
    end
    return J
end

"""
    turocy_partial_lambda!(dF_dλ, π, λ, payoffs, player_indices)

Computes the clean partial derivative with respect to λ (∂F/∂λ).
"""
function turocy_partial_lambda!(dF_dλ, π, λ, payoffs, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]

        # The simplex constraint does not depend on λ
        dF_dλ[ref_idx] = 0.0

        # For j > 1, the derivative is simply the negative utility difference
        for idx in range[2:end]
            dF_dλ[idx] = -(U[idx] - U[ref_idx])
        end
    end
    return dF_dλ
end


"""
    predict!(sys_jac!, sys_param!, x, t, lastdx, lastdt, Fx, Ft)

Predicts the next tangent step using analytical derivatives.
Reuses the pre-allocated Jacobian `Fx` and parameter derivative vector `Ft`.
"""
function predict!(sys_jac!, sys_param!, x, t, lastdx, lastdt, Fx, Ft)
    # 1. Compute analytical derivatives in-place
    sys_jac!(Fx, x, t)
    sys_param!(Ft, x, t)

    # 2. Compute dxdt = Fx \ Ft
    dxdt = Fx \ Ft

    dtds = 1.0 / sqrt(1.0 + dot(dxdt, dxdt))
    dxds = -dtds .* dxdt

    # Maintain orientation along the path
    if dot(dxds, lastdx) + dtds * lastdt < 0
        @. dxds = -dxds
        dtds = -dtds
    end

    return dxds, dtds
end

"""
    correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds, cache; iters=30, eps=1e-6)

Corrects the predicted point back to the homotopy path using an allocation-free bordered system solve.
"""
function correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds, cache; iters=100, eps=1e-6)
    # Unpack pre-allocated memory cache to avoid garbage collection loops
    @show "cr", ds
    r_sys, Fx, Ft, v, w = cache.r_sys, cache.Fx, cache.Ft, cache.v, cache.w

    xpred = xlast + ds .* dx
    tpred = tlast + ds * dt

    x = copy(xpred)
    t = tpred

    i = 0
    while true
        sys!(r_sys, x, t)
        r_con = dot(x .- xpred, dx) + (t - tpred) * dt

        if dot(r_sys, r_sys) + r_con^2 < eps^2
            @show i
            if i <= 1
                return x, t, ds*2
            else
                return x, t, ds
            end
        end
        if i >= iters
            if ds >= 1e-8
                return correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds/2, cache; iters=30, eps=1e-6)
            else
                error("corrector reached minimum step size")
            end
        end

        # Calculate analytical derivatives in-place
        sys_jac!(Fx, x, t)
        sys_param!(Ft, x, t)

        @show Fx
        # CRITICAL OPTIMIZATION: Factorize Fx once instead of doing Fx \ Ft and Fx \ (-r_sys) separately
        Fx_fact = lu!(Fx)

        ldiv!(v, Fx_fact, Ft)            # v = Fx \ Ft
        @. r_sys = -r_sys
        ldiv!(w, Fx_fact, r_sys)         # w = Fx \ (-r_sys)

        dt_step = (-r_con - dot(dx, w)) / (dt - dot(dx, v))

        # In-place updates to state variables
        @. x += w - dt_step * v
        t += dt_step
        i += 1
    end

    return x, t, ds
end

"""
    hc_turocy(startx, startt, endt, sys!, sys_jac!, sys_param!)

Main driver loop for pseudo-arclength continuation using analytical functions.
"""
function hc_turocy(startx, startt, endt, sys!, sys_jac!, sys_param!; max_iters=4000)
    N = length(startx)

    # Pre-allocate all linear algebra buffers inside a dedicated cache object
    cache = (
        r_sys = zeros(N),
        Fx    = zeros(N, N),
        Ft    = zeros(N),
        v     = zeros(N),
        w     = zeros(N)
    )

    x = copy(startx)
    t = startt
    dx = zero(startx)
    dt = sign(endt - startt)
    ds = 0.01
    i = 0

    while sign(endt - startt) * (t - endt) <= 1e-3 && i <= max_iters
        # Pass buffers into prediction step
        dx, dt = predict!(sys_jac!, sys_param!, x, t, dx, dt, cache.Fx, cache.Ft)
        # Pass buffers into correction step
        x, t, ds = correct!(sys!, sys_jac!, sys_param!, x, t, dx, dt, ds, cache)
        @show t,i,ds
        i += 1
    end
    @show i
    return x
end


# 1. Define your game structure (e.g., a 2-player game with 3 actions each)
player_indices = [1:3, 4:6]
#player_indices = [1:5, 6:10]
payoffs_p1 = [0.0 1.0 4.0; 1.0 0.0 1.0; 4.0 1.0 0.0]
#payoffs_p1 = [0.0 0.25 1.0 2.25 4.0; 0.25 0.0 0.25 1.0 2.25; 1.0 0.25 0.0 0.25 1.0; 2.25 1.0 0.25 0.0 0.25; 4.0 2.25 1.0 0.25 0.0]
payoffs_p2 = -payoffs_p2 #rand(3, 3)
payoffs = (payoffs_p1, payoffs_p2)

# 2. Build the analytical closures expected by the non-AD solver
sys_closure!(F, x, t)        = turocy_system!(F, x, t, payoffs, player_indices)
jac_closure!(J, x, t)        = turocy_jacobian!(J, x, t, payoffs, player_indices)
param_closure!(dF_dt, x, t)  = turocy_partial_lambda!(dF_dt, x, t, payoffs, player_indices)

# 3. Formulate your starting strategy vector (Centroid of the simplex at λ = 0)
guess = [1/3, 1/3, 1/3,  1/3, 1/3, 1/3]

#guess = [fill(1/5,5); fill(1/5,5)]

# 4. Execute the path tracker from λ = 0.0 to λ = 10.0
final_qre = hc_turocy(guess, 0.0, 1000000.0, sys_closure!, jac_closure!, param_closure!)
