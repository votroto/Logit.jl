To derive the partial derivatives of the log-ratio system $F$ with respect to the log-ratio coordinates $\mu$ and the precision parameter $\lambda$, we must first understand the transformation between $\mu$ and the strategy profile $\pi$.

---

## 1. Setting up the Log-Ratio Transformation

For each player $i$, let their actions be $j \in \{1, \dots, J_i\}$. The last action $J_i$ is chosen as the reference action.

The log-ratio coordinates $\mu_{ij}$ are defined for $j \in \{1, \dots, J_i - 1\}$ as:


$$\mu_{ij} = \ln\left(\frac{\pi_{ij}}{\pi_{iJ_i}}\right)$$

By definition, the reference action has a fixed coordinate:


$$\mu_{iJ_i} = 0$$

Using the softmax function, we recover the strategy probabilities $\pi$ from $\mu$:


$$\pi_{ij}(\mu) = \frac{e^{\mu_{ij}}}{\sum_{k=1}^{J_i} e^{\mu_{ik}}} = \frac{e^{\mu_{ij}}}{e^0 + \sum_{k=1}^{J_i-1} e^{\mu_{ik}}} = \frac{e^{\mu_{ij}}}{1 + \sum_{k=1}^{J_i-1} e^{\mu_{ik}}}$$

### Jacobian of the Softmax ($\frac{\partial \pi}{\partial \mu}$)

Since players' strategies are independent, $\frac{\partial \pi_{ij}}{\partial \mu_{lm}} = 0$ if $l \neq i$. For the same player $i$, the standard derivative of the softmax function yields:

* **For $j, m \neq J_i$:**

$$\frac{\partial \pi_{ij}}{\partial \mu_{im}} = \pi_{ij} (\delta_{jm} - \pi_{im})$$



where $\delta_{jm}$ is the Kronecker delta ($\delta_{jm} = 1$ if $j=m$, and $0$ otherwise).
* **For the reference action $j = J_i$:**

$$\frac{\partial \pi_{iJ_i}}{\partial \mu_{im}} = -\pi_{iJ_i} \pi_{im}$$



---

## 2. Deriving the Partials of $F$

The system $F_{ij}(\mu, \lambda) = 0$ is defined for all players $i$ and their non-reference actions $j \in \{1, \dots, J_i - 1\}$:


$$F_{ij}(\mu, \lambda) = \mu_{ij} - \lambda \left[ \bar{u}_{ij}(\pi(\mu)) - \bar{u}_{iJ_i}(\pi(\mu)) \right]$$

---

### A. Partial derivative with respect to $\lambda$

This derivative is straightforward because $\mu$ (and therefore $\pi$) is held constant:

$$\frac{\partial F_{ij}}{\partial \lambda} = -\left( \bar{u}_{ij}(\pi) - \bar{u}_{iJ_i}(\pi) \right)$$

---

### B. Partial derivative with respect to $\mu_{lm}$

To find $\frac{\partial F_{ij}}{\partial \mu_{lm}}$, we use the chain rule via the intermediate strategy variables $\pi$.

$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = \frac{\partial \mu_{ij}}{\partial \mu_{lm}} - \lambda \sum_{r \in \text{all players}} \sum_{s=1}^{J_r} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{rs}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{rs}} \right) \frac{\partial \pi_{rs}}{\partial \mu_{lm}}$$

Because a player's own strategy profile $\pi_r$ does not affect their own expected payoffs $\bar{u}_{rj}$ (since $\bar{u}_{rj}$ is the payoff when playing pure action $j$ against the opponents' strategy $\pi_{-r}$), we have:


$$\frac{\partial \bar{u}_{ij}}{\partial \pi_{is}} = 0 \quad \text{for all } j, s$$

Thus, non-zero payoff derivatives only occur when $r \neq i$.

We analyze the derivative under two cases: **same player** ($l = i$) and **different player** ($l \neq i$).

#### Case 1: Same Player ($l = i$)

When $l = i$, the first term $\frac{\partial \mu_{ij}}{\partial \mu_{im}} = \delta_{jm}$.

For the second term, since $r = i$ yields zero payoff derivatives, the sum over $r$ vanishes:


$$\frac{\partial F_{ij}}{\partial \mu_{im}} = \delta_{jm}$$

Specifically:

* If $j = m$: $\frac{\partial F_{ij}}{\partial \mu_{ij}} = 1$
* If $j \neq m$: $\frac{\partial F_{ij}}{\partial \mu_{im}} = 0$

#### Case 2: Different Player ($l \neq i$)

When $l \neq i$, the direct coordinate derivative $\frac{\partial \mu_{ij}}{\partial \mu_{lm}} = 0$.

The only surviving term in the sum over players is when $r = l$:


$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = -\lambda \sum_{s=1}^{J_l} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{ls}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{ls}} \right) \frac{\partial \pi_{ls}}{\partial \mu_{lm}}$$

Substituting the softmax derivatives $\frac{\partial \pi_{ls}}{\partial \mu_{lm}}$ derived in Section 1:


$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = -\lambda \left[ \sum_{s=1}^{J_l-1} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{ls}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{ls}} \right) \pi_{ls}(\delta_{sm} - \pi_{lm}) + \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lJ_l}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{lJ_l}} \right)(-\pi_{lJ_l}\pi_{lm}) \right]$$

Expanding and grouping terms by factoring out $\pi_{lm}$:


$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = -\lambda \left[ \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{lm}} \right)\pi_{lm} - \sum_{s=1}^{J_l} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{ls}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{ls}} \right)\pi_{ls}\pi_{lm} \right]$$

Factoring out $\pi_{lm}$ simplifies this to:


$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = -\lambda \pi_{lm} \left[ \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{lm}} \right) - \sum_{s=1}^{J_l} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{ls}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{ls}} \right)\pi_{ls} \right]$$

---

## 3. Summary of the Log-Ratio Partials

For $j \in \{1, \dots, J_i - 1\}$ and $m \in \{1, \dots, J_l - 1\}$:

### With respect to $\lambda$:

$$\frac{\partial F_{ij}}{\partial \lambda} = \bar{u}_{iJ_i}(\pi) - \bar{u}_{ij}(\pi)$$

### With respect to $\mu_{lm}$ (for the same player $l = i$):

$$\frac{\partial F_{ij}}{\partial \mu_{im}} = \delta_{jm}$$

### With respect to $\mu_{lm}$ (for a different player $l \neq i$):

$$\frac{\partial F_{ij}}{\partial \mu_{lm}} = -\lambda \pi_{lm} \left[ \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{lm}} \right) - \sum_{s=1}^{J_l} \left( \frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{ls}} - \frac{\partial \bar{u}_{iJ_i}(\pi)}{\partial \pi_{ls}} \right)\pi_{ls} \right]$$


--------
Ah, the classic division trick! By computing the full probability product $w = \prod_{z=1}^N \pi_{z}(a_z)$ first and then dividing out $\pi_p(a_p)$ (written as `x[p][i[p]]` in your Julia code), you avoid having to loop through and compute $N-1$ products for every single coordinate.

That is highly efficient for smaller games, though you have to watch out for division-by-zero if any strategy probability hits exactly $0$ (which, thankfully, in a logit/softmax equilibrium, they never do since $\pi_{ij} > 0$ always).

To compute the partial derivatives of these expected utilities:

$$\frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}}$$

you can extend this exact same tensor-sweep logic.

---

### The Derivative Version of Your Algorithm

When you take the derivative of $\bar{u}_{ij}(\pi)$ with respect to the opponent $l$'s probability of playing $m$, you are essentially **unilaterally fixing both player $i$ to action $j$ and player $l$ to action $m$**, and taking the expected payoff over the remaining $N-2$ players.

Using your product-and-divide approach, the joint probability weight for the remaining players is:


$$\prod_{r \neq i, l} \pi_{r, a_r} = \frac{w}{\pi_{i, a_i} \pi_{l, a_l}}$$

We can write a highly efficient Julia function `unilateral_derivatives` that computes these derivative matrices for all pairs of players $(p, q)$ where $p \neq q$:

```julia
function unilateral_derivatives(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    x::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    T = promote_type(P, X)

    # result[p][q] will hold a matrix of size (size_p, size_q)
    # representing ∂ū_p[j] / ∂π_q[m]
    result = ntuple(p ->
        ntuple(q -> zeros(T, size(payoffs[p], p), size(payoffs[p], q)), N),
        N
    )

    for i in CartesianIndices(first(payoffs))
        # Complete joint probability product
        @inbounds w = prod(x[z][i[z]] for z in 1:N)

        # We need to compute the derivative of player p's expected payoff
        # with respect to player q's strategy
        for p in 1:N
            for q in 1:N
                p == q && continue  # ∂ū_p / ∂π_p is always 0

                # Divide out both player p and player q's probabilities
                @inbounds w_deriv = w / (x[p][i[p]] * x[q][i[q]])

                # Accumulate into the (action_p, action_q) slot
                @inbounds result[p][q][i[p], i[q]] += w_deriv * payoffs[p][i]
            end
        end
    end
    return result
end

```

### Why this works beautifully:

1. **Zero-safeness**: In the Logit Equilibrium pathway, $\pi$ is the output of a softmax, meaning $\pi_{rs} \in (0, 1)$. The denominator `x[p][i[p]] * x[q][i[q]]` is guaranteed to be strictly positive.
2. **Memory Layout**: It computes all the pairwise Jacobian matrices $J_{pq} = \frac{\partial \bar{u}_p}{\partial \pi_q}$ in a single, cache-friendly pass over the payoff tensors.