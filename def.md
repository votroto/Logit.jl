Turocy defines logit equilibrium at parameter $\lambda$ as a strategy profile $\pi$ that solves the system
$$
H_{ij}(\pi, \lambda) = e^{\lambda \bar{u}_{ij}(\pi)}-\pi_{ij}\sum_{k=1}^{J_i}e^{\lambda \bar{u}_{ik}(\pi)} = 0
$$
where $J_i$ is the number of action of player $i$, and $\bar{u}_{ij}(\pi)$ is the expected utility of player $i$ deviating to a pure action $j$ while everyone else plays according to $\pi_{-i}$.

For numerical stability the strategies should be encoded as log-ratio coordinates instead.
Rewrite the system H such that it describes a logit equilibrium for a profile $\mu$ encoded as log-ratio coordinates instead of a profile $\pi$ containing raw probabilities.


u(x,\pi_{-i}) = u(\pi_1, ..., \pi_{i-1}, x, \pi_{i+1}, \pi_{n})