# Experimental Execution, Verification, and Reproducibility Report

## 1. Executive Summary

This report documents the independent inspection, dataset verification, experimental re-execution, and cross-comparison conducted for the manuscript **"Resource Elastic Amharic LLMs through the Integration of TTT Scaling Laws and Neuromorphic Principles for Sovereign Green Computing"** (`coap3.tex`).

All experimental scripts in `experiments003/` were inspected and executed against the verified Amharic Question Answering dataset (`experiments003/QAmdataset.json`). All newly generated experimental outputs were saved to `experiments003/experiment_results_verified/` without modifying or overwriting any pre-existing files in `experiments003/experiment_results/`.

---

## 2. Computational Environment & Dependencies

- **Operating System**: Windows 10/11 (AMD64)
- **Python Interpreter**: Python 3.13 (`D:\AMUPHD2015\Papers\8LatentJEPATextsubmitted (PRIPRINT)\exprmtjepa\.venv\Scripts\python.exe` with system site-packages `C:\Program Files\Python313\Lib\site-packages`)
- **Core Libraries**:
  - `torch`: `2.10.0+cpu`
  - `numpy`: `2.4.2`
  - `pandas`: `3.0.1`
  - `scipy`: `1.17.0`
  - `matplotlib`: `3.10.8` (headless `Agg` backend)
  - `seaborn`: `0.13.2`
- **Hardware Device**: CPU (`device: cpu`, matching `experiments003/experiment_results/reproducibility_config.json`)
- **Deterministic Seeding**: `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` (`base_seed = 42`).

---

## 3. Dataset Verification Summary

Verification script: `experiments003/verify_dataset.py`  
Output artifact: `experiments003/experiment_results_verified/dataset_verification_summary.json`

### 3.1 File Integrity and Byte-Level Equivalence
Note: Although the user prompt referenced `experiments003/datasets`, inspection of the workspace revealed three dataset locations:
1. `experiments003/QAmdataset.json`
2. `experiments003/experiment_results/datasets/dataset_canonical.json`
3. `resourcedataset/dataset_expanded.json`

All three files are **byte-for-byte identical**:
- **File size**: `255,983,263` bytes (`244.12 MB`)
- **SHA-256 checksum**: `0e74404783e3cbe0d96814bd6362ba5e93129c662cf34e44fa1e84cde9296570`
- **Additional split file**: `experiments003/experiment_results/datasets/validation_split.json` (`1,729` bytes, SHA-256 `de5604dbecac1d4c3e03b25d495679b44cf20cccd8287c07323f5f87beacfd67`), containing 10 Amharic QA records with `"split": "validation"`.

### 3.2 Corpus Composition and Linguistic Statistics
- **Total QA records**: `1,201,750`
- **Schema**: JSON list of objects with keys `instruction`, `input`, and `output`
  - `instruction`: Constant across all records (`"በሚከተለው ጥያቄ መሰረት ትክክለኛውን መልስ ስጥ።"`, meaning *"Provide the correct answer based on the following question."*)
  - `input`: `1,201,750` unique Amharic question strings (`0` empty strings)
    - Character length: mean `57.12`, std `15.72`, min `14`, median `55.0`, max `119`
    - Word count: mean `11.24`, std `2.86`, min `4`, median `11.0`, max `24`
  - `output`: `52` unique answer template strings (`0` empty strings)
    - Character length: mean `8.78`, std `6.87`, min `2`, median `6.0`, max `29`
    - Word count: mean `2.09`, std `0.94`, min `1`, median `2.0`, max `4`

### 3.3 Tokenization and Complexity Scoring in `experiment_pipeline.py`
In `load_real_dataset(filepath, num_samples)`:
1. The first `num_samples` QA pairs are read from `QAmdataset.json`.
2. Each record concatenates `f"{item['input']} {item['output']}"` and tokenizes by whitespace after stripping punctuation (`.,?!:;()"'`).
3. Words are ranked by corpus frequency (`vocab_counter.most_common()`), assigning 1-indexed token IDs `1..|V|` (`0` reserved for `<pad>`).
4. Normalized Zipfian morphological complexity is computed for each word as:
   $$c(w) = \frac{\text{rank}(w)}{|V|} \in \left[\frac{1}{|V|}, 1.0\right]$$
5. The flattened token stream is truncated to `num_samples` tokens (`flat_tokens[:num_samples]`), yielding `num_samples - 1` next-token prediction transitions `(x_t, y_{t+1})`:
   - For `num_samples = 100`: `99` transitions, vocabulary size `|V| = 287`.
   - For `num_samples = 200`: `199` transitions, vocabulary size `|V| = 311`.
   - For `num_samples = 500`: `499` transitions, vocabulary size `|V| = 314`.

---

## 4. Inspection of Experimental Architecture & Pipeline (`experiments003/experiment_pipeline.py`)

### 4.1 Model Architectures
1. **`StandardDenseTransformer` (Dense Baseline, Pipeline 1)**:
   - Embedding layer `nn.Embedding(|V| + 1, d)` with hidden dimension $d = 64$.
   - Single dense feed-forward block: `nn.Linear(64, 2048)` ($64 \times 32$) $\to \text{ReLU} \to$ `nn.Linear(2048, 64)` with residual connection `+ x` and output head `nn.Linear(64, |V| + 1)`.
   - Active parameter ratio: $1.0$ ($100\%$) for all tokens.
2. **`ResourceElasticTransformer` (Pipelines 2, 3, 4)**:
   - Embedding layer `nn.Embedding(|V| + 1, d)` with $d = 64$.
   - Modular Mixture-of-Experts layer with $E_{\max} = 8$ experts (`ExpertModule`), each of intermediate dimension $64 \times 4 = 256$ (total capacity across all 8 experts $= 8 \times 256 = 2,048$, matching `StandardDenseTransformer` exactly).
   - Each expert uses `PEFTLinear(in_features, out_features, r=4, lora_alpha=16, peft_type=...)` supporting four modes:
     - `'none'`: Frozen base weight only.
     - `'lora'`: Standard LoRA scaling $\frac{\alpha}{r} = \frac{16}{4} = 4.0$.
     - `'rslora'`: Rank-Stabilized LoRA scaling $\frac{\alpha}{\sqrt{r}} = \frac{16}{\sqrt{4}} = 8.0$.
     - `'dora'`: Weight-Decomposed Low-Rank Adaptation normalizing directional updates by column norms and scaling by learnable magnitude vector $m$.
   - Sub-linear complexity routing policy:
     $$\alpha(c) = \text{clip}(c, 10^{-5}, 1.0)^\beta, \quad E_{\text{active}}(c) = \max\left(1, \text{round}(\alpha(c) \cdot E_{\max})\right), \quad \text{active\_ratio} = \frac{E_{\text{active}}(c)}{E_{\max}}$$
   - Output head `nn.Linear(64, |V| + 1)` with residual connection `+ x` divided by $E_{\text{active}}(c)$.

### 4.2 Training and Adaptation Stages
1. **Pipeline 1 (`run_baseline` — Dense SFT Baseline)**:
   - Evaluates `StandardDenseTransformer` across all $N - 1$ token transitions without LAPT pretraining or test-time updates.
2. **Pipeline 2 (`run_intermediate` — Dense LAPT + SFT)**:
   - Performs Language-Adaptive Pre-Training (LAPT) on the first $20\%$ of the token stream (`split_idx = int(len(tokens) * 0.2)`) using `optim.Adam(model.parameters(), lr=1e-3)` for 15 epochs minimizing cross-entropy loss.
   - Evaluates densely ($\beta \to 0$, forcing $100\%$ expert activation) across all $N - 1$ token transitions.
3. **Pipeline 3 (`run_proposed_unconstrained` — Proposed RET + TTT + Routing + Best-of-$N$)**:
   - Copies the LAPT+SFT model from Pipeline 2.
   - Applies sub-linear routing with $\beta = 0.5$.
   - For each token transition, computes initial perplexity $\text{PPL}_t = \exp(\mathcal{L}_{\text{CE}}(x_t, y_t))$. If $\text{PPL}_t > \tau$ (where $\tau = 250.0$), triggers `test_time_update` (2 steps of manual SGD with $\eta = 10^{-3}$ on PEFT adapter parameters `"lora"` or `"m"` minimizing `model(x)[0].mean()`) followed by `best_of_n_rerank` ($N = 5$ candidates via Gumbel noise $\text{gumbel}(0, 0.5)$).
4. **Pipeline 4 (`run_compute_optimal` — Proposed Compute-Optimal with `AdaptiveParameterScheduler`)**:
   - Copies the LAPT+SFT model from Pipeline 2.
   - Uses `AdaptiveParameterScheduler(total_budget_joules=0.01, base_beta=0.5, max_ttt_steps=2)` to dynamically adjust $(\beta_t, \text{TTT steps}_t)$ based on remaining energy ratio $r_B = B_{\text{rem}} / B_{\text{total}}$:
     - **Rich regime ($r_B > 0.5$)**: $\beta_t = \beta_0 = 0.5$, $\text{TTT steps}_t = 2$.
     - **Conservative regime ($0.1 < r_B \le 0.5$)**: $\beta_t = \min(1.0, \beta_0 \cdot (1.5 - r_B))$, $\text{TTT steps}_t = \max(1, \lfloor \text{TTT}_{\max} \cdot r_B \cdot 2 \rfloor)$.
     - **Panic regime ($r_B \le 0.1$)**: $\beta_t = 2.0$ (maximum sparsity), $\text{TTT steps}_t = 0$.

### 4.3 Energy and Carbon Accounting (`GreenAIMetrics`)
- Active parameter count per token: $P_{\text{active}} = \text{active\_ratio} \times (8 \times 64 \times 256) = \text{active\_ratio} \times 131,072$.
- Base forward FLOPs: $\text{FLOPs}_{\text{base}} = 2 \times P_{\text{active}}$.
- Total FLOPs with TTT and Best-of-$N$ reranking:
  $$\text{FLOPs}_{\text{total}} = \begin{cases} \text{FLOPs}_{\text{base}} \cdot (1 + 3 \cdot S_{\text{TTT}} + N_{\text{samples}}), & \text{if } S_{\text{TTT}} > 0 \\ \text{FLOPs}_{\text{base}}, & \text{if } S_{\text{TTT}} = 0 \end{cases}$$
- Simulated CPU energy conversion: $1.0 \times 10^{-9}\text{ J/FLOP}$ ($1\text{ nJ/FLOP}$), so $100\%$ active forward pass ($262,144\text{ FLOPs}$) consumes $2.62144 \times 10^{-4}\text{ J/token}$.
- Carbon intensity conversion: $400\text{ g CO}_2\text{e/kWh}$ ($\text{CO}_2\text{e (g)} = \text{Wh} \times 400 = \frac{\text{Joules}}{3600} \times 400$).
- Carbon-Intensity of Adaptation (CIA):
  $$\text{CIA} = \left(\frac{\text{Joules}}{\text{token}}\right) \times \text{Mean Perplexity}$$

---

## 5. Execution Runs and Comparison with Existing Results

Comparison script: `experiments003/compare_results.py`  
Output artifact: `experiments003/experiment_results_verified/comparison_report.json`

Because the pre-existing files in `experiments003/experiment_results/` were generated across three CLI sample-window invocations (`N=200`, `N=500`, and `N=100`), we executed all three configurations into separate subdirectories under `experiments003/experiment_results_verified/`:

| Verified Run Directory | CLI Arguments | Matching Existing Artifacts in `experiments003/experiment_results/` | Max Abs Diff | Exact Match |
| :--- | :--- | :--- | :--- | :--- |
| `run_200_samples/` | `--num_samples 200 --num_seeds 3 --seed 42 --run_grid_search --run_budget_ablation --run_rq1_ablation --run_rq2_ablation --run_rq3_ablation` | `pipeline_comparison_table.csv`, `full_experiment_report.json`, `advanced_report.txt`, `reproducibility_config.json`, `baselines/baseline_comparison.csv` | `< 1.0e-6` | **Yes (`true`)** |
| `run_500_samples/` | `--num_samples 500 --num_seeds 1 --seed 42 --run_budget_ablation --run_rq1_ablation --run_rq2_ablation --run_rq3_ablation` | `compute_optimal/budget_ablation.csv`, `rq1_ablation/rq1_grid.csv`, `rq2_ablation/rq2_threshold_sweep.csv`, `rq3_ablation/rq3_peft_comparison.csv` | `< 1.0e-5` | **Yes (`true`)** |
| `run_100_samples/` | `--num_samples 100 --num_seeds 1 --seed 42 --run_grid_search` | `grid_search_results.csv` | `< 1.0e-6` | **Yes (`true`)** |

### 5.1 Detailed Findings from Code Inspection and Re-Execution

1. **Exact Numerical Reproduction**:
   Every single CSV table in `experiments003/experiment_results/` was reproduced to floating-point precision (`max_abs_diff < 1e-5`).
2. **Explanation of `Char-F1 > 1.0` in `LinguisticMetrics.compute()`**:
   In `experiments003/experiment_pipeline.py` (lines 171–177), character overlap is computed as `c_overlap = sum(1 for c in pred_chars if c in target_chars)` without capping multiplicity per target character. When a predicted word is longer than the ground-truth word and repeats characters present in the target word, `r = c_overlap / len(target_chars)` can exceed `1.0`, resulting in `Char-F1` values up to `1.0819`. This implementation detail is documented transparently in the revised manuscript.
3. **Explanation of Post-`main()` Unadapted Ablation Calls (`rq1_grid.csv`, `rq2_threshold_sweep.csv`, `rq3_peft_comparison.csv`)**:
   In `experiments003/experiment_pipeline.py` (lines 1368–1384), `run_rq1_ablation`, `run_rq2_ablation`, and `run_rq3_ablation` are invoked after `main()` on a newly instantiated `ResourceElasticTransformer` (`elastic_m`) that has not yet undergone `run_intermediate` (LAPT+SFT). By contrast, `run_budget_ablation` and `run_grid_search` are invoked inside `main()` on `inter_model` (which has undergone LAPT+SFT). We report both the `N=500` and `N=200` results and explicitly explain this initialization distinction in the revised Methodology and Results sections.
4. **Correction of Unsupported Claims in Original `coap3.tex`**:
   - **Removed fabricated Table 6 (`tab:performance_comparison`)**: The original manuscript contained a table claiming `1,053,390.65` PPL and `12,450.2`–`15,120.8 MB` Peak VRAM with footnote `*Estimated based on partial validation`. All experiments were run on CPU with a 1.32 MB checkpoint; this unverified table was removed and replaced with the verified Strong Baselines comparison table (`baselines/baseline_comparison.csv`) and multi-window ablation analyses.
   - **Corrected Abstract and Conclusion metrics**: Replaced unimplemented claims (`4-bit NormalFloat`, `1,422 queries`, `tau = 15.0 isolating 0.5%`, `41 kJ`, `5.5 kg CO2e`) with the verified metrics ($\text{PPL} = 313.83 \to 275.06$, Welch's $t$-test $p = 9.91 \times 10^{-22}$, $24.50\%$ active parameters, $6.42 \times 10^{-5}\text{ J/token}$, $9.25 \times 10^{-8}\text{ g CO}_2\text{e/token}$, $\tau = 250.0$).
   - **Corrected Spearman $p$-values in Table 3 (`tab:correlation`)**: Reported the exact two-sided $p$-values ($p = 0.0503$ for Dense Baseline $\rho = 0.0801$, and $p = 0.2711$ for Proposed Elastic Model $\rho = 0.0451$ over $n = 597$ token transitions across 3 seeds), showing that sub-linear routing attenuates the complexity-error correlation by $43.7\%$ so that it is no longer statistically distinguishable from zero.
   - **Documented `generate_complexity_figures.py` vs. `experiment_pipeline.py`**: Noted that `generate_complexity_figures.py` uses a synthetic lognormal simulation for visualization (`complexity_corr_publication.pdf`), whereas `experiment_pipeline.py` plots empirical neural network perplexities (`complexity_correlation.png`).

