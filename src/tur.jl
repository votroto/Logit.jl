include("path.jl")
using ForwardDiff
using BenchmarkTools




using Random
Random.seed!(3462345634)

As = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)



guess_reduced = [
    strat_to_point(normalize(ones(size(As[1], 1)), 1));
    strat_to_point(normalize(ones(size(As[1], 2)), 1));
    strat_to_point(normalize(ones(size(As[1], 3)), 1));
    strat_to_point(normalize(ones(size(As[1], 4)), 1));
    strat_to_point(normalize(ones(size(As[1], 5)), 1));
]


@time x1 = hc(guess_reduced, 0.0, 1000000.0, H, jac_l, jac_x, As)


#@btime x2 = hc(guess_reduced, 0.0, 1000000.0, H, jac_l, jac_x, Bs) setup=(Bs=( randn(5, 5, 5, 5, 5), randn(5, 5, 5, 5, 5), randn(5, 5, 5, 5, 5), randn(5, 5, 5, 5, 5), randn(5, 5, 5, 5, 5) ))


mu = splitviews(x1, size(As[1]) .- 1)
pi = point_to_strat.(mu)

for p in 1:5
println(round.(pi[p]; digits=5))
end
nothing

#=

x = [
    strat_to_point(normalize(rand(size(As[1], 1)), 1));
    strat_to_point(normalize(rand(size(As[1], 2)), 1));
    strat_to_point(normalize(rand(size(As[1], 3)), 1));
    strat_to_point(normalize(rand(size(As[1], 4)), 1));
    strat_to_point(normalize(rand(size(As[1], 5)), 1));
]

mu = splitviews(x, size(first(As)) .- 1)
pi = point_to_strat.(mu)


dudpi = ntuple(p -> ntuple(q -> zeros(size(As[p], p), size(As[p], q)), 5), 5)
unilateral_derivatives!(dudpi, As, pi)


dudpi2 = ntuple(p -> ntuple(q -> zeros(size(As[p], p), size(As[p], q)), 5), 5)
unilateral_derivatives2!(dudpi2, As, pi)

@show norm(dudpi[2][3] - dudpi2[2][3])



dudpi = ntuple(p -> ntuple(q -> zeros(size(As[p], p), size(As[p], q)), 5), 5)
@btime unilateral_derivatives!($dudpi, $As, pi)

dudpi2 = ntuple(p -> ntuple(q -> zeros(size(As[p], p), size(As[p], q)), 5), 5)
@btime unilateral_derivatives2!($dudpi2, $As, pi)

nothing
=#

#Profile.clear()
#@profile x1 = hc(guess_reduced, 0.0, 1000000.0, S, Sl, Sx)




#=
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
=#

#[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
#[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
