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

ff(x,y)=(x-y)^2

nash(Us)

#A = [ff(x,y) for x in -1.0:0.5:1.0, y in -1.0:0.5:1.0]

#Us = (A,-A)
@time pi = nash(Us)
#
#for p in eachindex(pi)
#    println(round.(pi[p]; digits=5))
#end
#nothing

 b = [0.0 for i in 1:3, j in 1:1]
#
@time nash((b,b))

#=
[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
[0.0, 0.0, 0.0, 0.73337, 0.26663]
[0.32376, 0.0, 0.0, 0.34957, 0.32667]
[0.105, 0.45382, 0.0, 0.0, 0.44118]


=#