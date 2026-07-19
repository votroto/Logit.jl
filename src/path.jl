using ForwardDiff
using LinearAlgebra
using UnicodePlots

function predict(system, jac_l, jac_x, x::Vector{Float64}, t::Float64, lastdx::Vector{Float64}, lastdt::Float64)
    hx = jac_x(x, t)
    ht = jac_l(x, t)

    dxdt = (hx \ ht)

    dtds = 1/sqrt(1+dot(dxdt, dxdt))
    dxds = - dtds*dxdt

    if dot(dxds, lastdx) + dot(dtds, lastdt) < 0
        dxds = -dxds
        dtds = -dtds
    end

    return dxds, dtds
end

function correct(system, jac_l, jac_x, xlast::Vector{Float64}, tlast::Float64, dx::Vector{Float64}, dt::Float64, ds::Float64; iters::Int=3, abs_tol::Float64=1e-6, rel_tol::Float64=1e-12)
    xpred, tpred = xlast + ds * dx, tlast + ds * dt
    x, t = xpred, tpred

    i = 0
    while true
        r_sys = system(x, t)
        r_con = dot(x - xpred, dx) + (t - tpred) * dt

        # 1. Absolute residual check
        if dot(r_sys, r_sys) + r_con ^ 2 < abs_tol^2
            return x, t, ds
        elseif i >= iters
           # println("decel")
            if ds >= 1e-4
                return correct(system, jac_l, jac_x, xlast, tlast, dx, dt, ds * 0.5; iters=iters, abs_tol=abs_tol, rel_tol=rel_tol)
            else
                error("can't follow path: step size too small")
            end
        end

        Fx = jac_x(x, t)
        Ft = jac_l(x, t)

        v = similar(Ft)
        w = similar(r_sys)
        @. r_sys = -r_sys

        lu_res = lu!(Fx)
        ldiv!(v, lu_res, Ft)
        ldiv!(w, lu_res, r_sys)

        dt_step = (-r_con - dot(dx, w)) / (dt - dot(dx, v))
        dx_step = w - dt_step * v

        # 2. Relative step size check (NEW)
        # If the step we are about to take is smaller than Float64 noise at this scale,
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

function hc(startx, startt, endt, system, jac_l, jac_x; max_iters=1000)
    x = startx
    t = startt
    dx = zero(startx)
    dt = sign(endt - startt)
    ds = 0.01
    i=0
    succs = 0
    while sign(endt-startt) * (t - endt) <= 1e-3 # && i <= max_iters
        #println("pred")
        dx, dt = predict(system, jac_l, jac_x, x, t, dx, dt)
        #println("corr")
        x, t, nds = correct(system, jac_l, jac_x, x, t, dx, dt, ds)
        #println(".")
        if nds == ds
   #         print(".")
            succs += 1
        else
         #   @show x, t, nds
            succs = 0
            ds = nds
        end
        if succs >= 4
            succs = 0
            ds *= 2
        end

        i+=1
    end
    @show i

    return x
end

#=
P = 0.1
T = 0.5

f(v) = (P .+ 3/(v .^ 2)) .* (v .- 1/3) .- T*8/3
g(v) = P*v .- T*8/3

H(v, lam) = lam*f(v[1])+(1-lam)*g(v[1])

guess = 8/3*T/P

x1, xs1, ts1 = hc(guess, 0.0, 2.0, H)

scatterplot(xs1, ts1, xlabel="x", ylabel="t")=#


