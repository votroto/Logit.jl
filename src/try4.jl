using LinearAlgebra

# ============================================================
# Expected utilities
# ============================================================

function compute_expected_utilities(π, payoffs, player_indices)
    U = zeros(eltype(π), length(π))

    if length(player_indices) == 2
        r1, r2 = player_indices

        mul!(view(U, r1), payoffs[1], view(π, r2))
        mul!(view(U, r2), payoffs[2]', view(π, r1))
    end

    return U
end

# ============================================================
# Utility gradients
# ============================================================

function compute_utility_gradients(π, payoffs, player_indices)
    dU = [zeros(length(r), length(π)) for r in player_indices]

    if length(player_indices) == 2
        r1, r2 = player_indices

        dU[1][:, r2] .= payoffs[1]
        dU[2][:, r1] .= payoffs[2]'
    end

    return dU
end

# ============================================================
# Convert log strategy -> probability strategy
# ============================================================

function exp_strategy(ell, player_indices)
    π = similar(ell)
    for r in player_indices
        @views π[r] .= exp.(ell[r])
    end
    return π
end

# ============================================================
# Stable log-sum-exp
# ============================================================

function logsumexp(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

# ============================================================
# Full softmax Turocy system in log coordinates
#
# ell = log(pi)
# ============================================================

function turocy_system_logsumexp!(F, ell, λ, payoffs, player_indices)
    π = exp_strategy(ell, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        # -------------------------------------------------
        # QRE equations
        # omit last action to remove redundancy
        # -------------------------------------------------
        actions = range[1:(end-1)]
        values = λ .* U[range]
        L = logsumexp(values)

        for idx in actions
            F[idx] = ell[idx] - λ * U[idx] + L
        end

        # -------------------------------------------------
        # normalization equation
        # -------------------------------------------------
        last_idx = last(range)
        F[last_idx] = sum(exp.(view(ell, range))) - 1.0
    end

    return F
end

# ============================================================
# Jacobian
# ============================================================

function turocy_jacobian_logsumexp!(J, ell, λ, payoffs, player_indices)
    fill!(J, 0.0)

    π = exp_strategy(ell, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)
    dU = compute_utility_gradients(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        # probabilities inside logit expectation
        psoft = exp.(λ .* U[range])
        psoft ./= sum(psoft)

        # ------------------------------------------
        # QRE rows
        # ------------------------------------------
        for idx in range[1:(end-1)]
            # direct ell derivative
            J[idx, idx] += 1.0

            for k in 1:length(ell)
                # derivative of -lambda Ui
                du1 = dU[i][idx-first(range)+1, k]

                # derivative of logsumexp term: lambda sum psoft_b dUb
                du2 = 0.0
                for (b, a) in enumerate(range)
                    du2 += psoft[b] * dU[i][b, k]
                end

                # chain rule: dpi/dell = pi
                J[idx, k] += λ * π[k] * (du2 - du1)
            end
        end

        # ------------------------------------------
        # normalization row
        # ------------------------------------------
        last_idx = last(range)
        for k in range
            J[last_idx, k] = π[k]
        end
    end

    return J
end

# ============================================================
# lambda derivative
# ============================================================

function turocy_partial_lambda_logsumexp!(dF, ell, λ, payoffs, player_indices)
    π = exp_strategy(ell, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        values = λ .* U[range]
        psoft = exp.(values)
        psoft ./= sum(psoft)

        for idx in range[1:(end-1)]
            expected = sum(psoft .* U[range])
            dF[idx] = expected - U[idx]
        end
        last_idx = last(range)
        dF[last_idx] = 0.0
    end

    return dF
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
    @show Fx

    # 2. Compute dxdt = Fx \\ Ft
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
            if i <= 1
                return x, t, ds * 2
            else
                return x, t, ds
            end
        end
        if i >= iters
            if ds >= 1e-8
                return correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds / 4, cache; iters=30, eps=1e-6)
            else
                error("corrector reached minimum step size")
            end
        end

        # Calculate analytical derivatives in-place
        sys_jac!(Fx, x, t)
        sys_param!(Ft, x, t)

        # @show Fx
        # CRITICAL OPTIMIZATION: Factorize Fx once instead of doing Fx \\ Ft and Fx \\ (-r_sys) separately
        Fx_fact = lu!(Fx)

        ldiv!(v, Fx_fact, Ft)            # v = Fx \\ Ft
        @. r_sys = -r_sys
        ldiv!(w, Fx_fact, r_sys)         # w = Fx \\ (-r_sys)

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
        r_sys=zeros(N),
        Fx=zeros(N, N),
        Ft=zeros(N),
        v=zeros(N),
        w=zeros(N)
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
        @show t, i, ds
        i += 1
    end
    @show i
    return x
end

# ============================================================
# Example game
# ============================================================

player_indices = [1:3, 4:6]

payoffs_p1 = [
    0.0 1.0 4.0;
    1.0 0.0 1.0;
    4.0 1.0 0.0
]

payoffs_p2 = -payoffs_p1
payoffs = (payoffs_p1, payoffs_p2)

# ============================================================
# Continuation closures
# ============================================================

sys_closure!(F, x, t) = turocy_system_logsumexp!(F, x, t, payoffs, player_indices)
jac_closure!(J, x, t) = turocy_jacobian_logsumexp!(J, x, t, payoffs, player_indices)
param_closure!(dF, x, t) = turocy_partial_lambda_logsumexp!(dF, x, t, payoffs, player_indices)

# ============================================================
# Initial condition: lambda=0 centroid (ell = log(pi))
# ============================================================

guess = log.([
    1/3, 1/3, 1/3,
    1/3, 1/3, 1/3
])

# ============================================================
# Run homotopy
# ============================================================

ell_qre = hc_turocy(
    guess,
    0.0,
    1000000.0,
    sys_closure!,
    jac_closure!,
    param_closure!
)

# Convert back to probabilities
π_qre = exp_strategy(ell_qre, player_indices)

println("QRE probabilities:")
println(π_qre)