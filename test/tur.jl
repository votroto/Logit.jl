include("../src/path.jl")
using BenchmarkTools




using Random
Random.seed!(3462345634)

Us = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

@time x1 = hc(Us)


mu = splitviews(x1, size(Us[1]) .- 1)
pi = point_to_strat.(mu)

for p in 1:5
println(round.(pi[p]; digits=5))
end
nothing
