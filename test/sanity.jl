using LinearAlgebra
using Gurobi
using JuMP


"""Computes the values and NE strategies for a general-sum game"""
function bilinear_program(us::NTuple{2, AbstractMatrix}, startx, starty, wstart; optimizer=Gurobi.Optimizer)
    m = Model(optimizer)
    #set_silent(m)
    nx, ny = size(us[1])
    @variable(m, xs[i=1:nx], lower_bound=0, upper_bound=1, start=startx[i])
    @variable(m, ys[i=1:ny], lower_bound=0, upper_bound=1, start=starty[i])
    @variable(m, w[i=1:2], lower_bound=minimum(us[i]), upper_bound=maximum(us[i]), start= wstart[i])

    @constraint(m, sum(xs) == 1)
    @constraint(m, sum(ys) == 1)

    @constraint(m, dot(xs, us[1], ys) + dot(xs, us[2], ys) >= sum(w))
    @constraint(m, (us[1] * ys)  .<= w[1])
    @constraint(m, (xs' * us[2]) .<= w[2])

    optimize!(m)

    value.(w), value.(xs), value.(ys), solve_time(m)
end

wstart = dot(pi[1],A,pi[2]), dot(pi[1],B,pi[2])

w,xsb,ysb,t = bilinear_program((A,B), pi[1], pi[2], wstart); nothing