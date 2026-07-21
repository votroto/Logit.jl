# Logit.jl

Computes a Nash Equilibrium of an N-player general sum game by a continuation over a logit equilibrium.

## Usage

```julia
include("../src/path.jl")

using Random
Random.seed!(3462345634)

Us = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

x1 = hc(Us)

mu = splitviews(x1, size(Us[1]) .- 1)
pi = point_to_strat.(mu)

for p in eachindex(pi)
println(round.(pi[p]; digits=5))
end
nothing
```
should print
```
[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
[0.0, 0.0, 0.0, 0.73337, 0.26663]
[0.32376, 0.0, 0.0, 0.34957, 0.32667]
[0.105, 0.45382, 0.0, 0.0, 0.44118]
```

## Notes
- *The project is a work-in-progress. Feedback is welcome, so is help.*
- *The Jacobian Kernels are specialized per the number of players. The first time an N-player game is solved will incur a compilation time penalty.*
- *There is no specialization for zero-sum games. Using a reasonable linear program will always be ten times faster.*

## Acknowledgements

Many thanks go to `BifurcationKit.jl`, `HomotopyContinuation.jl`, `bertini`, `gambit-logit` for their source code and manuals; to ChatGPT for deriving the jacobians; to Gemini and GitHub Copilot for refactoring; to Mosek and Gurobi for their academic licences for testing, and to AIC for their inexplicable continued support.

Based on the papers:

> Turocy, T. L. (2005). A dynamic homotopy interpretation of the logistic quantal response equilibrium correspondence. *Games and Economic Behavior, 51*(2), 243–263. https://doi.org/10.1016/j.geb.2004.04.003

> Timme, S. (2021). Mixed precision path tracking for polynomial homotopy continuation. Advances in Computational Mathematics, 47, 75. https://doi.org/10.1007/s10444-021-09899-y
