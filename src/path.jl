include("turocy.jl")
include("spaces.jl")
using LinearAlgebra

function predict(
    x::Vector{Float64},
    t::Float64,
    lastdx::Vector{Float64},
    lastdt::Float64,
    utils::NTuple{N}
) where {N}
    u = utils
    lam = t
    hx = zeros(length(x), length(x))

    mu = splitviews(x, size(first(u)) .- 1)
    pi = point_to_strat.(mu)

    dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(u[p], p), size(u[p], q)), N), N)
    unilateral_derivatives!(dudpi, u, pi)

    jacobian_x!(hx, pi, lam, dudpi, u)



    rsize = sum(size(first(u), i) - 1 for i in eachindex(u))
    ht = Vector{eltype(x)}(undef, rsize)

    ubar = ntuple(i -> zeros(eltype(x), size(u[i], i)), Val(N))
    unilateral_deviations!(ubar, u, pi)

    jacobian_l!(ht, ubar, mu, lam, u)

    dxdt = (hx \ ht)

    dtds = 1/sqrt(1+dot(dxdt, dxdt))
    dxds = - dtds*dxdt

    if dot(dxds, lastdx) + dot(dtds, lastdt) < 0
        dxds = -dxds
        dtds = -dtds
    end

    return dxds, dtds
end

function correct(
    xlast::Vector{Float64},
    tlast::Float64,
    dx::Vector{Float64},
    dt::Float64,
    ds::Float64,
    utils::NTuple{N};
    iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12
) where {N}
    xpred, tpred = xlast + ds * dx, tlast + ds * dt
    x, t = xpred, tpred

    i = 0
    while true
        rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))
        res = Vector{eltype(x)}(undef, rsize)
        ubar = ntuple(i -> zeros(eltype(x), size(utils[i], i)), Val(N))

        mu = splitviews(x, size(first(utils)) .- 1)
        pi = point_to_strat.(mu)

        unilateral_deviations!(ubar, utils, pi)

        residual!(res, mu, ubar, x, t, utils)



        r_con = dot(x - xpred, dx) + (t - tpred) * dt

        # 1. Absolute residual check
        if dot(res, res) + r_con ^ 2 < abs_tol^2
            return x, t, ds
        elseif i >= iters
            if ds >= 1e-4
                return correct(xlast, tlast, dx, dt, ds * 0.5, utils; iters=iters, abs_tol=abs_tol, rel_tol=rel_tol)
            else
                error("Progress along path stalled!")
            end
        end


        Fx = zeros(length(x), length(x))
        dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(utils[p], p), size(utils[p], q)), N), N)
        unilateral_derivatives!(dudpi, utils, pi)
        jacobian_x!(Fx, pi, t, dudpi, utils)



        Ft = Vector{eltype(x)}(undef, rsize)
        jacobian_l!(Ft, ubar, mu, t, utils)


        v = similar(Ft)
        w = similar(res)
        @. res = -res

        lu_res = lu!(Fx)
        ldiv!(v, lu_res, Ft)
        ldiv!(w, lu_res, res)

        dt_step = (-r_con - dot(dx, w)) / (dt - dot(dx, v))
        dx_step = w - dt_step * v

        # 2. Relative step size check (smaller than Float64 noise)
        # we have converged as far as the hardware allows.
        step_norm = sqrt(dot(dx_step, dx_step) + dt_step^2)
        val_norm = sqrt(dot(x, x) + t^2)

        if step_norm < rel_tol * val_norm
            return x, t, ds
        end

        x = x + dx_step
        t = t + dt_step
        i += 1
    end

    return x, t, ds
end

function hc(utils; max_iters=1000, init_x=uniform_xprofile(utils), init_t=0.0, end_t=1e6)
    x = init_x
    t = init_t
    dx = zero(init_x)
    dt = one(init_t)
    ds = 0.01
    i = 0
    succs = 0
    while t <= end_t && i <= max_iters
        dx, dt = predict(x, t, dx, dt, utils)
        x, t, nds = correct(x, t, dx, dt, ds, utils)

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

        i+=1
    end

    return x
end
