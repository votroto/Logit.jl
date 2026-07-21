using LinearAlgebra
using Gurobi
using JuMP
using UnicodePlots
include("../src/path.jl")

"""Computes the values and NE strategies for a general-sum game"""
function bilinear_program(us::NTuple{2, AbstractMatrix}, startx=normalize(ones(size(us[1],1),1)), starty=normalize(ones(size(us[2],2),1)), wstart=zeros(2); optimizer=Gurobi.Optimizer)
    m = Model(optimizer)
    set_silent(m)
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

    #NaN, zeros(nx), zeros(ny), solve_time(m)
    value.(w), value.(xs), value.(ys), solve_time(m)
end

#wstart = dot(pi[1],A,pi[2]), dot(pi[1],B,pi[2])

#w,xsb,ysb,t = bilinear_program((A,B), pi[1], pi[2], wstart); nothing

function test_size(n)
    Us = ( randn(n, n), randn(n, n) )

    stats = @timed xys = hc(Us)

    mu = splitviews(xys, size(Us[1]) .- 1)
    pi = point_to_strat.(mu)

    wstart = dot(pi[1],Us[1],pi[2]), dot(pi[1],Us[2],pi[2])

    w,xsb,ysb,t = bilinear_program(Us, pi[1], pi[2], wstart); nothing

    @assert norm(xsb-pi[1])<=1e-6
    @assert norm(ysb-pi[2])<=1e-6
    #w,xs,ys,t = bilinear_program(Us)


    #for p in 1:2
    #println(round.(pi[p]; digits=5))
    #end
    #println()
    #println(round.(xs;digits=5))
    #println(round.(ys;digits=5))
    nothing

    stats.time, t
end

results = [(test_size(n),n) for n in 2:50 for i in 1:4]

sizes = [n for ((th, tg), n) in results]
ths = [th for  ((th, tg), n) in results]
tgs = [tg for  ((th, tg), n) in results]


#scatterplot!(incs, sizes, tgs, color=:red, name="gurobi")

incs = Plot(; xlim=(0,maximum(sizes)), ylim=(0,0.2), title="runtime vs gamesize", ylabel="runtime (s)", xlabel="game size (NxN)")
scatterplot!(incs, sizes, ths, color=:blue, name="hc")
display(incs)
