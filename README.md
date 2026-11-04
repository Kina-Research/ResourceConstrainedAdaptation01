# Resource-Elastic Amharic LLMs through the Integration of TTT Scaling Laws and Neuromorphic Principles for Sovereign Green Computing

**Authors:** Kindie Alebachew, Durga Prasad Sharma, Messay Samuel, and Mohammed Abebe  
**Affiliation:** Faculty of Software Engineering and Computer Science, Arba Minch Institute of Technology, Arba Minch University, Arba Minch, Ethiopia

---

## Overview

This repository contains the official Taylor & Francis (`interact`) LaTeX manuscript, theoretical proof, Amharic Question Answering dataset, and reproducible PyTorch experimental pipeline for **Resource-Elastic Amharic Language Modeling**.

The framework integrates:
1. **Language-Adaptive Pre-Training (LAPT)** on in-domain Amharic text.
2. **Modular Resource-Elastic Transformer (RET)** with $E_{\max}=8$ parameter-efficient experts (`PEFTLinear` supporting `rslora`, `lora`, `dora`, and `none`) governed by a sub-linear Zipfian complexity routing policy $\alpha(c) = c^{\beta}$.
3. **Perplexity-Gated Test-Time Training (TTT)** ($\tau = 250.0$) restricted to low-rank adapter parameters, paired with **Best-of-$N$ Gumbel Candidate Reranking** ($N=5$).
4. **Three-Regime Adaptive Parameter Scheduler** (`AdaptiveParameterScheduler`) that dynamically modulates routing sparsity ($\beta_t$) and TTT steps ($S_{\text{TTT},t}$) across Rich, Conservative, and Panic energy budget regimes.

---

## Repository Structure

```text
.
├── coap3.tex                                  # Main manuscript (Taylor & Francis interact template)
├── coap3.pdf                                  # Compiled manuscript PDF
├── interact.cls                               # Official Taylor & Francis Interact document class
├── references.bib                             # BibLaTeX bibliography (72 references)
├── .latexmkrc                                 # XeLaTeX + Biber build configuration
├── requirements.txt                           # Python package dependencies
├── .gitignore                                 # Git ignore rules for LaTeX and Python build artifacts
├── .gitattributes                             # Git LFS configuration for large dataset & checkpoints
└── experiments003/
    ├── experiment_pipeline.py                 # End-to-end training, routing, TTT & ablation pipeline
    ├── generate_complexity_figures.py         # Publication complexity scatter & expert heatmap generator
    ├── verify_dataset.py                      # Dataset SHA-256 and linguistic statistics verifier
    ├── compare_results.py                     # Automated numerical reproducibility verifier
    ├── theoretical_proof.md                   # Formal proof of the Sub-Linear Zipfian Routing Lemma
    ├── QAmdataset.json                        # Amharic QA corpus (1,201,750 records; 244.1 MB)
    ├── configs/
    │   └── experiment_config.yaml             # Default experimental hyperparameters
    ├── experiment_results/                    # Primary experimental outputs, tables, and figures
    │   ├── pipeline_comparison_table.csv
    │   ├── grid_search_results.csv
    │   ├── full_experiment_report.json
    │   ├── reproducibility_config.json
    │   ├── baselines/                         # Random, Unigram, Dense, and Elastic baseline comparisons
    │   ├── compute_optimal/                   # Log-spaced energy budget ablation (0.001 J - 0.100 J)
    │   ├── rq1_ablation/                      # 36-configuration factorial sweep (Budget x Beta x TTT)
    │   ├── rq2_ablation/                      # Perplexity routing threshold sweep (tau in [5, 500])
    │   ├── rq3_ablation/                      # PEFT comparison (none, lora, rslora, dora) & CIA metric
    │   └── figures/                           # Publication figures included in coap3.tex
    └── experiment_results_verified/           # Independent multi-window verification runs & reports
        ├── REPRODUCIBILITY_AND_EXECUTION_REPORT.md
        ├── dataset_verification_summary.json
        ├── comparison_report.json
        ├── run_100_samples/                   # Verified N=100 grid-search run
        ├── run_200_samples/                   # Verified N=200 3-seed primary run
        └── run_500_samples/                   # Verified N=500 extended ablation run
```

---

## Quick Start & Reproducing Experiments

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Experimental Pipeline
```bash
# Default 3-seed evaluation (N=200 tokens/seed, Seeds 42, 43, 44)
python experiments003/experiment_pipeline.py

# Full ablation suite (N=200, 3 seeds + Grid Search + Budget + RQ1 + RQ2 + RQ3 ablations)
python experiments003/experiment_pipeline.py \
    --num_samples 200 --num_seeds 3 --seed 42 \
    --run_grid_search --run_budget_ablation \
    --run_rq1_ablation --run_rq2_ablation --run_rq3_ablation

# Extended 500-token single-seed ablation suite
python experiments003/experiment_pipeline.py \
    --num_samples 500 --num_seeds 1 --seed 42 \
    --run_budget_ablation --run_rq1_ablation \
    --run_rq2_ablation --run_rq3_ablation
```

### 3. Verify Dataset Integrity and Numerical Reproducibility
```bash
python experiments003/verify_dataset.py
python experiments003/compare_results.py
python experiments003/generate_complexity_figures.py
```

---

## Summary of Verified Results ($N=200$, 3 Seeds)

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

## Compiling the LaTeX Manuscript

The manuscript [coap3.tex](coap3.tex) uses the Taylor & Francis `interact` class ([interact.cls](interact.cls)) and requires `xelatex` (for Ethiopic *Fidel* script support via `polyglossia` and `fontspec`) and `biber`:

```bash
xelatex -interaction=nonstopmode -synctex=1 coap3.tex
biber coap3
xelatex -interaction=nonstopmode -synctex=1 coap3.tex
xelatex -interaction=nonstopmode -synctex=1 coap3.tex
```
Or simply run `latexmk -xelatex coap3.tex` (configured via [.latexmkrc](.latexmkrc)).

---

## GitHub Upload Notes (Git LFS)

Because `experiments003/QAmdataset.json` is `244.1 MB` (exceeding GitHub's standard `100 MB` file limit), [.gitattributes](.gitattributes) is pre-configured to track `experiments003/QAmdataset.json` and `*.pt` files with **Git Large File Storage (Git LFS)**:

```bash
git lfs install
git add .
git commit -m "Initial commit: Resource-Elastic Amharic LLMs manuscript and verified pipeline"
git push origin main
```

