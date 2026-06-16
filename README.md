# Logit.jl

Pure Julia implementation of logit quantal response equilibrium (QRE) solver for simultaneous strategic games.

Replaces `gambit-logit -q -e -m1e-8` without any Gambit dependencies.

## Usage

```julia
using Logit

# Define payoff arrays (one per player)
A = [1.0 0.0; 3.0 2.0]  # Player 1
B = [2.0 3.0; 0.0 1.0]  # Player 2

# Solve QRE with regret tolerance 1e-8
result = solve_qre((A, B), 1e-8)

# Access results
println("Lambda: $(result.lambda)")
println("Player 1 mixed strategy: $(result.strategies[1])")
println("Player 2 mixed strategy: $(result.strategies[2])")
println("Regret: $(result.regret)")
```

## API

```julia
solve_qre(payoffs::NTuple{N, Array{Float64, N}}, regret_tol::Float64=1e-8) -> QREResult
```

**Arguments:**
- `payoffs`: Tuple of payoff arrays, one per player. For N-player game, each array is N-dimensional.
- `regret_tol`: Regret tolerance for termination (default 1e-8)

**Returns:**
- `QREResult` with fields:
  - `lambda::Float64`: precision parameter at terminal equilibrium
  - `strategies::Vector{Vector{Float64}}`: mixed strategy for each player
  - `regret::Float64`: maximum regret

## Implementation

Solves the system:
- For player i, strategy j≥2: `log(pᵢⱼ) - log(pᵢ₁) = λ(uᵢⱼ - uᵢ₁)`
- For player i: `Σⱼ pᵢⱼ = 1`

Traces along the QRE path from λ=0 until maximum regret ≤ regret_tol.
