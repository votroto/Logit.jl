include("turocy.jl")
include("spaces.jl")
using LinearAlgebra

# ---------- workspace helpers ----------

# zero every array in nested tuples
_zero_nested!(x::AbstractArray) = fill!(x, zero(eltype(x)))
function _zero_nested!(t::Tuple)
    for ti in t
        _zero_nested!(ti)
    end
end

# allocate all scratch storage once
function make_hc_workspace(x_template::Vector{Float64}, utils::NTuple{N}) where {N}
    T = eltype(x_template)
    n = length(x_template)
    rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))

    pi    = ntuple(i -> Vector{T}(undef, size(utils[i], i)), Val(N))
    res   = Vector{T}(undef, rsize)
    ubar  = ntuple(i -> zeros(T, size(utils[i], i)), Val(N))
    dudpi = ntuple(p -> ntuple(q -> zeros(T, size(utils[p], p), size(utils[p], q)), Val(N)), Val(N))

    Fx = zeros(T, n, n)
    Ft = Vector{T}(undef, rsize)

    v = Vector{T}(undef, rsize)
    w = Vector{T}(undef, rsize)

    return (; pi, res, ubar, dudpi, Fx, Ft, v, w)
end

function predict!(
    x::Vector{Float64},
    t::Float64,
    lastdx::Vector{Float64},
    lastdt::Float64,
    utils::NTuple{N},
    ws
) where {N}
    lam = t
    hx = ws.Fx
    ht = ws.Ft
    pi = ws.pi
    ubar = ws.ubar
    dudpi = ws.dudpi

    _zero_nested!(ubar)
    _zero_nested!(dudpi)

    mu = splitviews(x, size(first(utils)) .- 1)
    point_to_strat!.(pi, mu)

    unilateral_derivatives!(dudpi, utils, pi)
    jacobian_x!(hx, pi, lam, dudpi, utils)

    unilateral_deviations!(ubar, utils, pi)
    jacobian_l!(ht, ubar, mu, lam, utils)

    dxdt = (hx \ ht)  # if this is still hot, you can reuse a factorization path too

    dtds = 1 / sqrt(1 + dot(dxdt, dxdt))
    dxds = -dtds * dxdt

    if dot(dxds, lastdx) + dtds * lastdt < 0
        dxds = -dxds
        dtds = -dtds
    end

    return dxds, dtds
end

function correct!(
    xlast::Vector{Float64},
    tlast::Float64,
    dx::Vector{Float64},
    dt::Float64,
    ds::Float64,
    utils::NTuple{N},
    ws;
    iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12
) where {N}
    xpred, tpred = xlast + ds * dx, tlast + ds * dt
    x, t = copy(xpred), copy(tpred)

    pi    = ws.pi
    res   = ws.res
    ubar  = ws.ubar
    dudpi = ws.dudpi
    Fx    = ws.Fx
    Ft    = ws.Ft
    v     = ws.v
    w     = ws.w

    i = 0
    while true
        fill!(res, 0.0)
        _zero_nested!(ubar)

        mu = splitviews(x, size(first(utils)) .- 1)
        point_to_strat!.(pi, mu)

        unilateral_deviations!(ubar, utils, pi)
        residual!(res, mu, ubar, x, t, utils)

        r_con = dot(x - xpred, dx) + (t - tpred) * dt

        if dot(res, res) + r_con^2 < abs_tol^2
            return x, t, ds
        elseif i >= iters
            if ds >= 1e-4
                return correct!(xlast, tlast, dx, dt, ds * 0.5, utils, ws;
                                iters=iters, abs_tol=abs_tol, rel_tol=rel_tol)
            else
                error("Progress along path stalled!")
            end
        end

        _zero_nested!(dudpi)

        unilateral_derivatives!(dudpi, utils, pi)
        jacobian_x!(Fx, pi, t, dudpi, utils)
        jacobian_l!(Ft, ubar, mu, t, utils)

        res .*= -1

        lu_res = lu!(Fx)       # in-place LU; Fx overwritten
        ldiv!(v, lu_res, Ft)
        ldiv!(w, lu_res, res)

        dt_step = (-r_con - dot(dx, w)) / (dt - dot(dx, v))
        dx_step = w - dt_step * v

        step_norm = sqrt(dot(dx_step, dx_step) + dt_step^2)
        val_norm  = sqrt(dot(x, x) + t^2)

        if step_norm < rel_tol * val_norm
            return x, t, ds
        end

        x .+= dx_step
        t += dt_step
        i += 1
    end
end

function hc(utils; max_iters=1000, init_x=uniform_xprofile(utils), init_t=0.0, end_t=1e6)
    x = init_x
    t = init_t
    dx = zero(init_x)
    dt = one(init_t)
    ds = 0.01
    i = 0
    succs = 0

    ws = make_hc_workspace(x, utils)

    while t <= end_t && i <= max_iters
        dx, dt = predict!(x, t, dx, dt, utils, ws)
        x, t, nds = correct!(x, t, dx, dt, ds, utils, ws)

        if nds == ds
            succs += 1
        else
            succs = 0
            ds = nds
        end
        if succs >= 4
            succs = 0
            ds *= 2
        end

        i += 1
    end

    return x
end