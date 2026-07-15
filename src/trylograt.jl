using LinearAlgebra

# ============================================================
# Reduced log-ratio coordinates
#
# y[a] = log(pi[a]/pi[ref])
# ============================================================

function logits_to_probs(y, player_indices)
    π = zeros(length(y) + length(player_indices))
    offset = 0

    for r in player_indices
        m = length(r)
        z = zeros(m)

        # non-reference actions
        z[1:(m-1)] .= y[(offset+1):(offset+m-1)]

        # reference action logit = 0
        z[m] = 0.0

        # stable softmax
        mx = maximum(z)
        ez = exp.(z .- mx)
        ez ./= sum(ez)

        π[r] .= ez
        offset += m - 1
    end

    return π
end

# ============================================================
# Expected utilities
# ============================================================

function compute_expected_utilities(π, payoffs, player_indices)
    U = zeros(eltype(π), length(π))
    r1, r2 = player_indices

    mul!(view(U, r1), payoffs[1], view(π, r2))
    mul!(view(U, r2), payoffs[2]', view(π, r1))

    return U
end

# ============================================================
# Utility gradients
# ============================================================

function compute_utility_gradients(π, payoffs, player_indices)
    dU = [zeros(length(r), length(π)) for r in player_indices]
    r1, r2 = player_indices

    dU[1][:, r2] .= payoffs[1]
    dU[2][:, r1] .= payoffs[2]'

    return dU
end

# ============================================================
# Stable softmax
# ============================================================

function softmax_stable(x)
    m = maximum(x)
    y = exp.(x .- m)
    y ./= sum(y)
    return y
end

# ============================================================
# Reduced QRE equations
#
# y_a - lambda(U_a-U_ref)=0
# ============================================================

function turocy_reduced_system!(F, y, λ, payoffs, player_indices)
    π = logits_to_probs(y, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)
    offset = 0

    for r in player_indices
        ref = last(r)

        for a in r[1:(end-1)]
            F[offset+a-first(r)+1] = y[offset+a-first(r)+1] - λ * (U[a] - U[ref])
        end

        offset += length(r) - 1
    end

    return F
end

# ============================================================
# Jacobian wrt reduced coordinates
# ============================================================

function turocy_reduced_jacobian!(J, y, λ, payoffs, player_indices)
    fill!(J, 0.0)
    π = logits_to_probs(y, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)
    dU = compute_utility_gradients(π, payoffs, player_indices)
    offset = 0

    for (i, r) in enumerate(player_indices)
        m = length(r)
        ref = last(r)

        # derivative of probabilities wrt logits:
        # dpi_a/dy_b = pi_a(delta_ab-pi_b)
        for a_local in 1:m-1
            row = offset + a_local
            a = r[a_local]
            J[row, row] += 1.0

            for b_global in 1:length(y)
                # map logit coordinate to action
                if b_global <= offset + m - 1 && b_global > offset
                    b_local = b_global - offset
                    b = r[b_local]
                    dp = π[a] * ((a == b ? 1.0 : 0.0) - π[b])
                else
                    # opponent coordinates still affect utilities
                    dp = 0.0
                end

                du = 0.0
                for k in 1:length(r)
                    du += dU[i][a_local, b_global] * dp
                end

                # utility derivative: d(Ua-Uref)/dy
                du_ref = 0.0
                for k in 1:length(r)
                    du_ref += dU[i][m, k] * dp
                end

                J[row, b_global] -= λ * (du - du_ref)
            end
        end

        offset += m - 1
    end

    return J
end

# ============================================================
# lambda derivative
# ============================================================

function turocy_reduced_lambda!(dF, y, λ, payoffs, player_indices)
    π = logits_to_probs(y, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)
    offset = 0

    for r in player_indices
        ref = last(r)

        for a in r[1:(end-1)]
            dF[offset+a-first(r)+1] = -(U[a] - U[ref])
        end

        offset += length(r) - 1
    end

    return dF
end

# ============================================================
# Predictor-Corrector Homotopy Continuation
# ============================================================

"""
    predict!(sys_jac!, sys_param!, x, t, lastdx, lastdt, Fx, Ft)

Predicts the next tangent step using analytical derivatives.
Reuses the pre-allocated Jacobian `Fx` and parameter derivative vector `Ft`.
"""
function predict!(sys_jac!, sys_param!, x, t, lastdx, lastdt, Fx, Ft)
    # 1. Compute analytical derivatives in-place
    sys_jac!(Fx, x, t)
    sys_param!(Ft, x, t)

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
    correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds, cache; iters=100, eps=1e-6)

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
            @show i
            if i <= 2
                return x, t, ds * 2
            elseif i>= 16
                return x, t, ds / 2
            else
                return x, t, ds
            end
        end

        if i >= iters
            if ds >= 1e-8
                println("ouch")
                return correct!(sys!, sys_jac!, sys_param!, xlast, tlast, dx, dt, ds / 4, cache)
            else
                error("corrector reached minimum step size")
            end
        end

        # Calculate analytical derivatives in-place
        sys_jac!(Fx, x, t)
        sys_param!(Ft, x, t)

        # Factorize Fx once to minimize operations
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
    hc_turocy(startx, startt, endt, sys!, sys_jac!, sys_param!; max_iters=4000)

Main driver loop for pseudo-arclength continuation using analytical functions.
"""
function hc_turocy(startx, startt, endt, sys!, sys_jac!, sys_param!; max_iters=2000)
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
       @show i, x, t, ds
       #@show i,ds
        i += 1
    end
    return x
end

# ============================================================
# Problem Setup & Initialization
# ============================================================

player_indices = [1:3, 4:6]

payoffs_p1 = [
    0.0 1.0 4.0;
    1.0 0.0 1.0;
    4.0 1.0 0.0
]

payoffs_p2 = -payoffs_p1
payoffs = (payoffs_p1, payoffs_p2)

sys_closure!(F, x, t) = turocy_reduced_system!(F, x, t, payoffs, player_indices)
jac_closure!(J, x, t) = turocy_reduced_jacobian!(J, x, t, payoffs, player_indices)
param_closure!(dF, x, t) = turocy_reduced_lambda!(dF, x, t, payoffs, player_indices)

guess = zeros(4)

# ============================================================
# Run Homotopy
# ============================================================

y_qre = hc_turocy(
    guess,
    0.0,
    1000000.0,
    sys_closure!,
    jac_closure!,
    param_closure!
)

π_qre = logits_to_probs(y_qre, player_indices)

println(π_qre)