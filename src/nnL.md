You are 100% right, and that was a massive blind spot on my part. What a clumsy way to implement it! Defeating the whole purpose of Turocy's formulation by throwing `exp` right back into the loop is exactly what *not* to do.

Turocy does not want you to change the state variables to logs and shuffle them back and forth. Instead, the core insight of his paper is to **transform the equations of the system itself into log-form** while keeping the strategy profile $\pi$ as the primary state variable.

By taking the logarithm of the standard QRE condition, you completely banish the dangerous $\exp(\lambda U)$ term from your code. This is what gives the homotopy solver its absolute numerical immunity against overflow as $\lambda \to \infty$.

### The True Turocy Log-Linear System

For each player $i$, we choose a reference action (typically the first action, $j=1$). Instead of the probability matching an exponential, we rewrite the equilibrium condition as a log-utility relationship for all other actions $j > 1$:

$$F_{ij}(\pi, \lambda) = \ln(\pi_{ij}) - \ln(\pi_{i1}) - \lambda \left( U_{ij}(\pi) - U_{i1}(\pi) \right) = 0$$

To complete the system for player $i$, the equation for the reference slot ($j=1$) simply enforces the standard linear probability simplex constraint:

$$F_{i1}(\pi, \lambda) = \sum_{k} \pi_{ik} - 1 = 0$$

Notice the magic here: $U_{ij}(\pi)$ is just a basic multilinear polynomial of the opponent strategies. **There is not a single `exp()` function required anywhere in this entire setup.**

---

### Correct, High-Efficiency Julia Implementation

Here is the clean, mathematically faithful version of Turocy's system, its exact analytical Jacobian, and its partial derivative with respect to $\lambda$.

```julia
using LinearAlgebra

"""
    turocy_system!(F, π, λ, payoffs, player_indices)

Computes Turocy's log-transformed QRE homotopy system.
- `π`: The strategy profile vector (probabilities directly).
- `player_indices`: Vector of ranges mapping each player's strategy block in `π`.
"""
function turocy_system!(F, π, λ, payoffs, player_indices)
    # 1. Compute expected utilities (Polynomial in π, no exponentials!)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]  # Action 1 as reference

        # Reference action row enforces the simplex constraint
        F[ref_idx] = sum(view(π, range)) - 1.0

        # Remaining actions enforce the log-utility condition
        for idx in range[2:end]
            F[idx] = log(π[idx]) - log(π[ref_idx]) - λ * (U[idx] - U[ref_idx])
        end
    end
    return F
end

"""
    turocy_jacobian!(J, π, λ, payoffs, player_indices)

Computes the exact analytical Jacobian ∂F/∂π for the log-transformed system.
"""
function turocy_jacobian!(J, π, λ, payoffs, player_indices)
    fill!(J, 0.0)
    dU = compute_utility_gradients(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]

        # 1. Derivatives for the simplex constraint row (j = 1)
        for idx_k in range
            J[ref_idx, idx_k] = 1.0
        end

        # 2. Derivatives for the log-utility rows (j > 1)
        for (local_j, idx_j) in enumerate(range[2:end])
            # Adjust local index because we skipped the first element
            j_idx_in_dU = local_j + 1

            # Derivative of the log terms: ∂/∂π
            J[idx_j, idx_j] += 1.0 / π[idx_j]
            J[idx_j, ref_idx] -= 1.0 / π[ref_idx]

            # Derivative of the utility terms across all game strategies k
            for idx_k in 1:length(π)
                ∂Utility = dU[i][j_idx_in_dU, idx_k] - dU[i][1, idx_k]
                J[idx_j, idx_k] -= λ * ∂Utility
            end
        end
    end
    return J
end

"""
    turocy_partial_lambda!(dF_dλ, π, λ, payoffs, player_indices)

Computes the clean partial derivative with respect to λ (∂F/∂λ).
"""
function turocy_partial_lambda!(dF_dλ, π, λ, payoffs, player_indices)
    U = compute_expected_utilities(π, payoffs, player_indices)

    for (i, range) in enumerate(player_indices)
        ref_idx = range[1]

        # The simplex constraint does not depend on λ
        dF_dλ[ref_idx] = 0.0

        # For j > 1, the derivative is simply the negative utility difference
        for idx in range[2:end]
            dF_dλ[idx] = -(U[idx] - U[ref_idx])
        end
    end
    return dF_dλ
end

# --- Standard Game Utilities Helpers (Same as before but used natively) ---

function compute_expected_utilities(π, payoffs, player_indices)
    U = zeros(eltype(π), length(π))
    if length(player_indices) == 2
        r1, r2 = player_indices[1], player_indices[2]
        mul!(view(U, r1), payoffs[1], view(π, r2))
        mul!(view(U, r2), payoffs[2]', view(π, r1))
    end
    return U
end

function compute_utility_gradients(π, payoffs, player_indices)
    dU = [zeros(length(r), length(π)) for r in player_indices]
    if length(player_indices) == 2
        r1, r2 = player_indices[1], player_indices[2]
        dU[1][:, r2] .= payoffs[1]
        dU[2][:, r1] .= payoffs[2]'
    end
    return dU
end

```

---

### Why this structure is beautiful:

1. **Zero Exponentials**: Look at `turocy_system!`. The absolute value of $\lambda$ can scale to $10^8$ or higher, and the function will never throw an `Inf` or `NaN` because it is completely driven by basic subtractions and multiplications.
2. **Natively Handles Boundaries**: If a strategy $\pi_{ij}$ tries to drift to a negative value during a poor predictor step, `log(π[idx])` will instantly error or trigger an invalid step domain flag in your ODE/homotopy line-search solver, forcing the step-size stabilizer to shrink the step and stay safely inside the simplex.

Are you writing a custom Euler-Newton predictor-corrector loop to step through $\lambda$, or are you plugging this directly into an existing differential equation package?