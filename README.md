# Resource-Elastic Amharic LLMs through the Integration of TTT Scaling Laws and Neuromorphic Principles for Sovereign Green Computing

**Authors:** Kindie Alebachew, Durga Prasad Sharma, Messay Samuel, and Mohammed Abebe  
**Affiliation:** Faculty of Software Engineering and Computer Science, Arba Minch Institute of Technology, Arba Minch University, Arba Minch, Ethiopia

---

## Summary

The framework integrates:
1. **Language-Adaptive Pre-Training (LAPT)** on in-domain Amharic text.
2. **Modular Resource-Elastic Transformer (RET)** with $E_{\max}=8$ parameter-efficient experts (`PEFTLinear` supporting `rslora`, `lora`, `dora`, and `none`) governed by a sub-linear Zipfian complexity routing policy $\alpha(c) = c^{\beta}$.
3. **Perplexity-Gated Test-Time Training (TTT)** ($\tau = 250.0$) restricted to low-rank adapter parameters, paired with **Best-of-$N$ Gumbel Candidate Reranking** ($N=5$).
4. **Three-Regime Adaptive Parameter Scheduler** (`AdaptiveParameterScheduler`) that dynamically modulates routing sparsity ($\beta_t$) and TTT steps ($S_{\text{TTT},t}$) across Rich, Conservative, and Panic energy budget regimes.

---
## Results ($N=200$, 3 Seeds)

| Pipeline Configuration | Mean PPL ($\downarrow$) | 95% Bootstrap CI | Active Params (%) | Energy (J/token) | Accuracy (%) | BLEU-1 | CPR ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dense Baseline (SFT)** | `313.83` | `[310.13, 317.55]` | `100.00%` | `2.621e-04` | `0.00%` | `0.5427` | `---` |
| **2. Intermediate Dense (LAPT+SFT)** | `273.02` | `[266.33, 279.41]` | `100.00%` | `2.621e-04` | `22.61%` | `1.0000` | `22.61` |
| **3. Proposed Unconstrained (RET+TTT)** | `273.57` | `[266.92, 280.00]` | `35.74%` | `9.369e-05` | `21.11%` | `0.9950` | `59.07` |
| **4. Proposed Compute-Optimal** | `275.06` | `[268.17, 281.68]` | `24.50%` | `6.422e-05` | `20.10%` | `0.9950` | **`82.04`** |

- **Statistical Significance (Pipeline 1 vs. Pipeline 4)**: Welch's two-sample $t$-test $p = 9.91 \times 10^{-22}$.
- **Complexity-Error Attenuation**: Spearman rank correlation drops by **43.7%** from $\rho = 0.0801$ ($p = 0.0503$) in the Dense Baseline to $\rho = 0.0451$ ($p = 0.2711$) in the Proposed Elastic Model.
- **Carbon-Intensity of Adaptation (CIA)**: `rslora` achieves the lowest $\text{CIA}$ ($1.804 \times 10^{-2}$ at $N=500$; $1.751 \times 10^{-2}$ at $N=200$).

---
