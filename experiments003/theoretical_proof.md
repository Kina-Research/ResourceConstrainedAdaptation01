# Routing Lemma: Energy-Efficiency under Zipfian Complexity

## Theorem (Routing Lemma)
Assume a vocabulary or token stream where the linguistic complexity $c \in (0, 1]$ follows a Zipfian distribution, where high complexity is exponentially or polynomially less frequent than low complexity. If the routing policy assigns compute capacity based on a sub-linear function $\alpha(c) = c^\beta$ for $0 < \beta < 1$, then the expected energy consumption is strictly bounded below maximum capacity while allocating maximal capacity ($\alpha=1$) only to the highest complexity tokens ($c=1$).

## Proof Sketch

### 1. Assumption of Zipfian Complexity Distribution
Let $c$ represent the normalized token complexity, $c \in (0, 1]$.
In morphologically rich languages, extreme complexity (e.g., highly inflected, rare word forms) follows a Zipfian or power-law distribution. 
Let the probability density function of token complexity be $P(c) \propto c^{-s}$, where $s > 0$ controls the skew towards lower complexities. 
We assume $P(c)$ is normalized such that $\int_{0}^{1} P(c) dc = 1$.

### 2. Resource-Elastic Routing Function
The routing function $\alpha(c)$ defines the fraction of active experts (or active parameter ratio). We define it as:
$$ \alpha(c) = c^\beta, \quad 0 < \beta < 1 $$
This is a concave, monotonically increasing sub-linear function mapping $(0, 1] \rightarrow (0, 1]$.

### 3. Expected Energy Formulation
Let $E_{max}$ be the energy proxy (e.g., maximum FLOPs) when all experts are active ($\alpha = 1$). The expected energy for the resource-elastic model $E_{elastic}$ is given by:
$$ E_{elastic} = \int_{0}^{1} P(c) \cdot \alpha(c) \cdot E_{max} \, dc $$

Substituting $\alpha(c)$:
$$ E_{elastic} = E_{max} \int_{0}^{1} P(c) \cdot c^\beta \, dc $$

### 4. Bounding the Expected Energy
Since $c \in (0, 1]$ and $0 < \beta < 1$, we know that $c^\beta < 1$ for all $c < 1$. 
Given that $P(c)$ is heavily skewed towards small $c$ (due to the Zipfian assumption), the integral of $P(c) c^\beta$ is strictly less than the integral of $P(c) \cdot 1$.

Therefore:
$$ \int_{0}^{1} P(c) \cdot c^\beta \, dc < \int_{0}^{1} P(c) \cdot 1 \, dc = 1 $$

Thus:
$$ E_{elastic} < E_{max} $$
This proves that the sub-linear routing strictly reduces the expected energy.

### 5. Preservation of Peak Capacity
For the most complex tokens where $c \to 1$:
$$ \lim_{c \to 1} \alpha(c) = \lim_{c \to 1} c^\beta = 1 $$
This ensures that maximum model capacity is still fully allocated to the hardest tokens, preserving task performance.

## Conclusion
A sub-linear routing policy $\alpha(c) = c^\beta$ over a Zipfian distribution of token complexity guarantees a reduction in expected compute energy ($E_{elastic} < E_{max}$) while preserving the capacity required to accurately process the most complex linguistic tokens. This theoretical foundation enables Pareto-optimal scaling in morphologically rich language modeling.
