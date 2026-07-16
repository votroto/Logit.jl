Turocy defines logit equilibrium at parameter $\lambda$ as a strategy profile $\pi$ that solves the system
$$
H_{ij}(\pi, \lambda) = e^{\lambda \bar{u}_{ij}(\pi)}-\pi_{ij}\sum_{k=1}^{J_i}e^{\lambda \bar{u}_{ik}(\pi)} = 0
$$
where $J_i$ is the number of action of player $i$, and $\bar{u}_{ij}(\pi)$ is the expected utility of player $i$ deviating to a pure action $j$ while everyone else plays according to $\pi_{-i}$.

For numerical stability the strategies should be encoded as log-ratio coordinates $\mu$ instead with the last actions as references.

The new system F is then
$$
F_{ij}(\mu, \lambda) = \mu_{ij} - \lambda(\bar{u}_{ij}(\pi) - \bar{u}_{iJ_i}(\pi))
$$
with $\pi$ obtained by softmax from $\mu$.

Turocy also defines the partials for $H$ wrt $\pi$ as
$$
\frac{\partial H_{ij}}{\partial \pi_{ij}} = \frac{e^{\lambda\bar{u}_{ij}(\pi)}}{\pi_{ij}}
$$
or for action $k \neq j$ as
$$
\frac{\partial H_{ij}}{\partial \pi_{ik}} = 0
$$
or for player $l \neq i$ as
$$
\frac{\partial H_{ij}}{\partial \pi_{lm}} = e^{\lambda\bar{u}_{ij}(\pi)}\lambda\sum_{k=1}^{J_i}\left(\frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}} - \frac{\partial \bar{u}_{ik}(\pi)}{\partial \pi_{lm}}\right)\pi_{ik}
$$
and wrt $\lambda$ as

$$
\frac{\partial H_{ij}}{\partial \lambda} = e^{\lambda\bar{u}_{ij}(\pi)}\sum_{k=1}^{J_i}\left(\bar{u}_{ij}(\pi) - \bar{u}_{ik}(\pi)\right)\pi_{ik}
$$
but Turocy does not define the same for the log-ratio system.

Derive the partial derivatives.


$$\frac{\partial \bar{u}_{ij}(\pi)}{\partial \pi_{lm}}$$