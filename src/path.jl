using ForwardDiff
using LinearAlgebra
using UnicodePlots

function predict(system, x, t, lastdx, lastdt)
    hx = ForwardDiff.jacobian(x -> system(x, t), x)
    ht = ForwardDiff.derivative(t -> system(x, t), t)

    dxdt = (hx \ ht)

    dtds = 1/sqrt(1+dot(dxdt, dxdt))
    dxds = - dtds*dxdt

    if dot(dxds, lastdx) + dot(dtds, lastdt) < 0
        dxds = -dxds
        dtds = -dtds
    end

    return dxds, dtds
end

function update_stepsize(ds, iters; target_iters=2)
    ds * (1.1 + (target_iters - iters)/4)
end

function correct(system, xlast, tlast, dx, dt, ds; iters=3, eps=1e-6)
    xpred, tpred = xlast + ds * dx, tlast + ds * dt
    x, t = xpred, tpred

    i = 0
    while true
        r_sys = system(x, t)
        r_con = dot(x - xpred, dx) + (t - tpred) * dt

        if dot(r_sys, r_sys) + r_con ^ 2 < eps^2
            return x, t, update_stepsize(ds, i)
        elseif i >= iters
            return correct(system, xlast, tlast, dx, dt, ds * 0.5)
        end

        Fx = ForwardDiff.jacobian(A -> system(A, t), x)
        Ft = ForwardDiff.derivative(B -> system(x, B), t)

        v = Fx \ Ft
        w = Fx \ (-r_sys)

        dt_step = (-r_con - dot(dx, w)) / (dt - dot(dx, v))
        dx_step = w - dt_step * v

        x = x + dx_step
        t = t + dt_step
        i += 1
    end


    return x, t, ds
end

function hc(startx, startt, endt, system; max_iters=300)
    x = startx
    t = startt
    dx = zero(startx)
    dt = sign(endt - startt)
    ds = 0.01
    i=0
    while sign(endt-startt) * (t - endt) <= 1e-3 && i <= max_iters
        dx, dt = predict(system, x, t, dx, dt)
        x, t, ds = correct(system, x, t, dx, dt, ds)
        i+=1
    end
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