include("../src/spaces.jl")
include("../src/turocy.jl")
include("../src/path.jl")

using Random
using LinearAlgebra

function covariant_game(actions::NTuple{N,Int}, r) where {N}
    @assert -1/(N-1) <= r <= 1

    Σ = Matrix{Float64}(I, N, N)
    for i in 1:N, j in 1:N
        if i != j
            Σ[i,j] = r
        end
    end

    L = cholesky(Symmetric(Σ)).L

    G = Array{Float64}(undef, actions..., N)

    for I in CartesianIndices(actions)
        @views G[Tuple(I)..., :] .= L * randn(N)
    end

    slices = eachslice(G, dims=N)
    ntuple(i-> Array(slices[i]), N)
end
using Random
#Random.seed!(3462345634)

Us = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)


Us10 = (
    randn(10,10,10,10,10),
    randn(10,10,10,10,10),
    randn(10,10,10,10,10),
    randn(10,10,10,10,10),
    randn(10,10,10,10,10)
)

#ff(x,y)=(x-y)^2


nash(Us)

ut = covariant_game((10,10,10,10,10), -0.2)

@time pi, status = nash(Us10)
#for p in eachindex(pi)
#    println(round.(pi[p]; digits=5))
#end

#=
[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
[0.0, 0.0, 0.0, 0.73337, 0.26663]
[0.32376, 0.0, 0.0, 0.34957, 0.32667]
[0.105, 0.45382, 0.0, 0.0, 0.44118]


=#