include("../src/turocy.jl")
using LinearAlgebra
using BenchmarkTools
using Random
Random.seed!(3462345634)

u = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

x = [
    strat_to_point(normalize(rand(size(u[1], 1)), 1));
    strat_to_point(normalize(rand(size(u[1], 2)), 1));
    strat_to_point(normalize(rand(size(u[1], 3)), 1));
    strat_to_point(normalize(rand(size(u[1], 4)), 1));
    strat_to_point(normalize(rand(size(u[1], 5)), 1));
]

N=5

mu = splitviews(x, size(first(u)) .- 1)
pi = point_to_strat.(mu)

dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(u[p], p), size(u[p], q)), N), N)
unilateral_derivatives!(dudpi, u, pi)


dudpi = ntuple(p -> ntuple(q -> zeros(eltype(x), size(u[p], p), size(u[p], q)), N), N)
@btime unilateral_derivatives!($dudpi, u, pi)
