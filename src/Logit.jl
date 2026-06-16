module Logit

using LinearAlgebra
using ForwardDiff

export solve_qre

"""
    QREResult

Result of QRE solving.

Fields:
- `lambda::Float64`: precision parameter
- `strategies::Vector{Vector{Float64}}`: mixed strategy profile per player
- `regret::Float64`: maximum regret at equilibrium
"""
struct QREResult
    lambda::Float64
    strategies::Vector{Vector{Float64}}
    regret::Float64
end

"""
    solve_qre(payoffs::NTuple{N, Array{Float64, N}}, regret_tol::Float64=1e-8) -> QREResult

Solve for the terminal QRE along the principal branch until regret threshold.

Arguments:
- `payoffs::NTuple{N, Array{Float64, N}}`: payoff array for each player
- `regret_tol::Float64`: regret tolerance for termination (default 1e-8)

Returns:
- `QREResult` with final lambda, strategy profile, and regret

Example:
```julia
A = [1.0 0.0; 3.0 2.0]
B = [2.0 3.0; 0.0 1.0]
result = solve_qre((A, B), 1e-8)
println(result.lambda)
println(result.strategies[1])  # Player 1's mixed strategy
```
"""
function solve_qre(payoffs::NTuple{N, Array{Float64, N}}, regret_tol::Float64=1e-8) where N
    # Get strategy counts from array dimensions
    strategy_counts = NTuple{N, Int}(size(payoffs[i], i) for i in 1:N)
    
    # Start at uniform mixture, lambda = 0
    x = initial_point(strategy_counts)
    
    # Trace along path from lambda=0 until regret < tol
    trace_until_regret(x, payoffs, strategy_counts, regret_tol)
end

# ============================================================================
# Internal implementation
# ============================================================================

function initial_point(strategy_counts::NTuple{N, Int}) where N
    """Start point: uniform mixture in log-space, lambda=0."""
    x = Float64[]
    for n in strategy_counts
        append!(x, zeros(n))  # log(1/n) - log(1/n) = 0 for uniform
    end
    push!(x, 0.0)  # lambda = 0
    return x
end

function trace_until_regret(x::Vector, payoffs::NTuple{N, Array{Float64, N}},
                            strategy_counts::NTuple{N, Int}, regret_tol::Float64) where N
    """Trace QRE path from lambda=0 until regret threshold."""
    
    λ = 0.0
    λ_max = 1e6
    h = 0.03  # initial step size
    
    while λ < λ_max
        # Check termination
        strats = get_strategies(x, strategy_counts)
        reg = compute_regret(strats, payoffs)
        
        if reg < regret_tol
            return QREResult(λ, strats, reg)
        end
        
        # Newton step to find F(x, λ+h) = 0
        λ_next = min(λ + h, λ_max)
        try
            x = solve_for_lambda(x, λ_next, payoffs, strategy_counts)
            λ = λ_next
        catch
            # Step failed, reduce step size
            h = h / 1.1
            if h < 1e-8
                break
            end
        end
    end
    
    # Return final point
    strats = get_strategies(x, strategy_counts)
    reg = compute_regret(strats, payoffs)
    return QREResult(λ, strats, reg)
end

function solve_for_lambda(x::Vector, λ::Float64, payoffs::NTuple{N, Array{Float64, N}},
                          strategy_counts::NTuple{N, Int}) where N
    """Solve F(x, λ) = 0 for x using Newton's method."""
    
    for _ in 1:20  # Newton iterations
        F = qre_equations(x, λ, payoffs, strategy_counts)
        J = qre_jacobian(x, λ, payoffs, strategy_counts)
        
        # Check convergence
        if norm(F) < 1e-10
            return x
        end
        
        # Newton step
        Δx = -J \ F
        x = x + Δx
    end
    
    return x
end

function qre_equations(x::Vector, λ::Float64, payoffs::NTuple{N, Array{Float64, N}},
                       strategy_counts::NTuple{N, Int}) where N
    """Evaluate QRE system F(x, λ) = 0.
    
    Variables: x = [log(p₁₁), ..., log(p₁ₘ₁), log(p₂₁), ..., log(p_NₘN)]
    
    Equations:
    1. For player i, strategy j>1: log p_ij - log p_i1 = λ(u_ij - u_i1)
    2. For player i: Σⱼ pᵢⱼ = 1
    """
    
    strats = get_strategies(x, strategy_counts)
    payoffs_strat = expected_payoffs(strats, payoffs)
    
    F = Float64[]
    
    # Logit ratio equations
    for i in 1:N
        for j in 2:strategy_counts[i]
            # log(p_ij) - log(p_i1) = λ(u_ij - u_i1)
            logit_eq = x[idx(i,j,strategy_counts)] - x[idx(i,1,strategy_counts)] -
                       λ * (payoffs_strat[i][j] - payoffs_strat[i][1])
            push!(F, logit_eq)
        end
    end
    
    # Sum-to-one constraints
    for i in 1:N
        sum_eq = sum(strats[i]) - 1.0
        push!(F, sum_eq)
    end
    
    return F
end

function qre_jacobian(x::Vector, λ::Float64, payoffs::NTuple{N, Array{Float64, N}},
                      strategy_counts::NTuple{N, Int}) where N
    """Jacobian via automatic differentiation."""
    
    f(x) = qre_equations(x, λ, payoffs, strategy_counts)
    return ForwardDiff.jacobian(f, x)
end

function get_strategies(x::Vector, strategy_counts::NTuple{N, Int}) where N
    """Convert log-probability vector to normalized strategy vectors."""
    
    strats = Vector{Float64}[]
    pos = 1
    
    for n in strategy_counts
        log_p = @view x[pos:pos+n-1]
        # Numerically stable softmax: exp(log_p - max) / sum(exp(log_p - max))
        max_log_p = maximum(log_p)
        p = exp.(log_p .- max_log_p) ./ sum(exp.(log_p .- max_log_p))
        push!(strats, p)
        pos += n
    end
    
    return strats
end

function expected_payoffs(strats::Vector{Vector{Float64}}, payoffs::NTuple{N, Array{Float64, N}}) where N
    """Compute expected payoff for each strategy."""
    
    payoffs_strat = [similar(s) for s in strats]
    
    # Iterate over all strategy profiles
    _fill_payoffs!(payoffs_strat, strats, payoffs, 1, ones(Int, N))
    
    return payoffs_strat
end

function _fill_payoffs!(payoffs_strat, strats, payoffs, player, idx)
    """Recursively fill payoff arrays."""
    
    if player > length(strats)
        # Base case: we have a complete strategy profile
        for p in 1:length(strats)
            idx_p = @view idx[1:length(strats)]
            payoff_val = payoffs[p][idx_p...]
            contribution = prod(strats[q][idx[q]] for q in 1:length(strats) if q ≠ p)
            payoffs_strat[p][idx[p]] += payoff_val * contribution
        end
        return
    end
    
    # Recursive case
    for s in 1:length(strats[player])
        idx[player] = s
        _fill_payoffs!(payoffs_strat, strats, payoffs, player + 1, idx)
    end
end

function compute_regret(strats::Vector{Vector{Float64}}, payoffs::NTuple{N, Array{Float64, N}}) where N
    """Maximum regret across all players."""
    
    payoffs_strat = expected_payoffs(strats, payoffs)
    
    max_regret = 0.0
    for i in 1:N
        exp_payoff = dot(strats[i], payoffs_strat[i])
        regret_i = maximum(payoffs_strat[i]) - exp_payoff
        max_regret = max(max_regret, regret_i)
    end
    
    return max_regret
end

function idx(player::Int, strategy::Int, strategy_counts::NTuple{N, Int}) where N
    """Linear index into x vector for a given player and strategy."""
    
    offset = 0
    for i in 1:player-1
        offset += strategy_counts[i]
    end
    return offset + strategy
end

end  # module
