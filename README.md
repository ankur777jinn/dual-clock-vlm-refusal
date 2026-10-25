# Dual-Clock VLM Refusal: Mechanistic Investigation & Empirical Falsification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: Qwen2.5-VL-7B](https://img.shields.io/badge/model-Qwen2.5--VL--7B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset%20Results-yellow.svg)](https://huggingface.co/datasets/Jinn-69/dual-clock-vlm-refusal-results)
[![Pre-registration: v12](https://img.shields.io/badge/preregistration-v12-green.svg)](docs/preregistration_v12.md)
[![Status: Falsified](https://img.shields.io/badge/verdict-FALSIFIED%20(F1)-red.svg)](outputs/final_verdict/final_verdict.json)

An end-to-end, pre-registered mechanistic interpretability campaign investigating cross-modal attention routing, representation back-patching, and refusal dynamics in Vision-Language Models (**Qwen/Qwen2.5-VL-7B-Instruct**, 28 transformer layers, `bfloat16`).

---

## 📌 Executive Summary

Does a Vision-Language Model fail to refuse multimodal hazards because **cross-modal visual reading completes before text-aligned safety recognition emerges** ($L_\text{read} < L_\text{harm}$)?

Through a pre-registered 5,600-evaluation factorial activation patching grid ($9 \times 7$ layers), text-to-image attention knockout curves, cross-modal linear probe sweeps, and an exhaustive suite of confirmatory controls (C1–C5), we demonstrate that **this hypothesis is empirically falsified**:

```
                         Hypothesized Dual Clocks (Prior):
          Layers:  0 ──────── 10 ──────── 14 ──────── 18 ──────── 27
                              ▲                       ▲
                           L_read                  L_harm
                      [Visual Reading]        [Safety Refusal]
                         Gap: L_harm - L_read = +6 to +8 layers

                                        vs.

                          Empirical Ground Truth (Discovered):
          Layers:  0 ──────── 6 ───────── 9 ───────────────────── 27
                              ▲           ▲                       ▲
                           L_harm      L_read                  L_read
                          (Safety)    (Elbow)                  (Floor)
                         Gap: L_harm - L_read = -3 to -20 layers
```

* **Official Verdict:** **`FALSIFIED`** (Chain stopped at Gate 1; **Falsifier F1: Premature Alignment** triggered).
* **The "Early Safety Reflex" Architecture:** In `Qwen2.5-VL-7B`, multimodal hazard discrimination emerges at **Layer 6** ($\text{AUROC} = 0.825$), well before visual reading closes (elbow at **Layer 9**, floor at **Layer 26**). Refusal acts as an early perceptual reflex rather than a late deliberative judgment.
* **Spatial Invariance (Construct Warning Flag):** Permuting the 2D spatial arrangement of image tokens yields **$100.23\%$ effect retention** ($95\%\text{ CI } [98.9\%, 101.6\%]$), proving that VLM safety evaluates **bag-of-tokens semantic presence** rather than spatial geometry.

---

## 📊 Publication Figures

High-resolution vector plots and PNGs generated from the empirical data:

| Figure | Description | Preview |
| :---: | :--- | :---: |
| **Figure 1** | **Diagnostic Clocks:** (a) Attention Knockout Curve ($L_\text{read}$ at 9 / 26) and (b) Cross-Modal Transfer AUROC ($L_\text{harm} = 6$), showing the reversed latency gap ($\Delta L = -3$). | [PNG](outputs/figures/figure1_dual_clocks.png) · [PDF](outputs/figures/figure1_dual_clocks.pdf) |
| **Figure 2** | **Factorial Response Surface:** Full $9 \times 7$ grid heatmaps for Harmful Safety (HS) vs. Benign Safe (SS) showing the Bridge suppression zone ($L_\text{dst} \le 9$) and the TooLate null zone ($L_\text{dst} > 9$). | [PNG](outputs/figures/figure2_grid_heatmaps.png) · [PDF](outputs/figures/figure2_grid_heatmaps.pdf) |
| **Figure 3** | **Controls & Preconditions Suite:** C1a/C1b donor swaps and specificity alongside C1c/C1d/C3 retention ratios. | [PNG](outputs/figures/figure3_controls_summary.png) · [PDF](outputs/figures/figure3_controls_summary.pdf) |

All raw parquet files and execution logs are hosted at 🤗 [Hugging Face: Jinn-69/dual-clock-vlm-refusal-results](https://huggingface.co/datasets/Jinn-69/dual-clock-vlm-refusal-results).

---

## 🔬 Empirical Findings & Results Summary

### 1. Phase 1: Diagnostic Clocks ($N=100$)
* **Visual Attention Knockout ($L_\text{read}$):**  
  Text $\to$ image cross-attention masking identified a sharp inflection elbow at **Layer 9** ($\Delta\text{BIC} = 36.37$). Under a strict $5\%$ baseline floor, visual information transfer persists until **Layer 26**.
* **Harm Emergence Clock ($L_\text{harm}$):**  
  Linear text-safety probes (AdvBench vs. Alpaca) transfer to image tokens at **Layer 6** ($\text{AUROC} = 0.8249 > 0.70$).
* **Reversed Latency Gap:**  
  $$\Delta L = L_\text{harm} - L_\text{read} = 6 - 9 = -3 \text{ layers (or } -20 \text{ layers)}$$
  Because $\Delta L \le -1.5$, **Falsifier F1 (Premature Alignment)** is triggered at Gate 1.

### 2. Phase 2: Factorial Grid Response Surface (5,600 evaluations)
* **TooLate Quadrant ($L_\text{dst} > 9$):**  
  Produces **exact null modulation** ($\Delta M = +0.0004$ for HS, $\Delta M = -0.0002$ for SS). Back-patching after Layer 9 has zero effect on generated completions.
* **Bridge Quadrant ($L_\text{dst} \le 9$):**  
  Late visual tokens injected into early layers **suppress refusal** ($M$ drops from $2.4034 \to 1.2944$, refusal rate drops from $88\% \to 24\%$).
* **Continuous Changepoint Competitor:**  
  Profiles over layers to identify the global optimal changepoint at **$L_\text{dst} = 6.0$** ($\text{Slope}_\text{pre} = +0.7801, \text{Slope}_\Delta = -0.7053$), parametrically validating Phase 1.

### 3. Confirmatory Mixed-Effects Model (LMM)
Fit with centered effect coding ($\pm 0.5$) and cluster-robust sandwich covariance clustered on `image_id`:
* **Interaction Parameter ($\beta_\text{three-way}$ scaled):** **$-0.6591$** ($\text{SE} = 0.0231, 95\%\text{ CI } [-0.7044, -0.6137]$).
* **One-Sided Significance ($H_1: \beta > 0$):** $z = -28.53, p = 1.0000$ (Null not rejected; effect direction inverted).
* **TOST Equivalence:** Non-equivalent within $\delta_\text{equiv} = \pm 0.0372$ ($p_\text{lower} = 1.0$).

### 4. Experimental Controls & Preconditions Battery
* **Precondition C4 (Bitwise Identity):** $\max |\Delta z| = 0.000000e+00$ bitwise exact in `bfloat16`.
* **Precondition C5 (Typographic Calibration):** $100\%$ refusal on typographic attacks ($M = 6.0 \dots 9.2$).
* **Control C1a (Cross-Class Swap):** HS $\to$ SS donor swap shifts $\Delta M = +2.030$; SS $\to$ HS shifts $\Delta M = -1.588$.
* **Control C1b (Within-Pair Specificity):** Paired difference $\Delta M = -0.5666$ across 40 matched pairs.
* **Control C1c (Same-Class Shuffle):** Shuffled donors retain $99.12\%$ of self-patch effect.
* **Control C1d (Spatial Shuffle):** $100.23\%$ effect retention under 2D grid permutation ($CI: [98.89\%, 101.60\%]$), triggering the **Construct Warning Flag**.
* **Control C3 (Norm Rescaling):** Retains $86.27\%$ of effect under norm-matched rescaling, ruling out activation norm explosions (Falsifier F5 NOT triggered).
* **Capability Benchmark:** POPE & MMBench non-inferiority satisfied (`True`).

---

## 🛠️ Repository Structure

```
dual-clock-vlm-refusal/
├── README.md                           # Project overview & empirical findings
├── pyproject.toml                      # Dependencies & package metadata
├── docs/
│   ├── empirical_findings_and_analysis.md  # In-depth scientific synthesis report
│   ├── preregistration_v12.md          # Formal pre-registration protocol
│   └── phase1_empirical_findings.md    # Phase 1 diagnostic report
├── src/
│   ├── config.py                       # Layer grid (55 cells), thresholds, paths
│   ├── patching_engine.py              # Exact M-RoPE activation patching & attention knockout
│   ├── phase1_diagnostics.py           # Knockout curve (1A) & Cross-modal transfer (1B)
│   ├── phase2_grid.py                  # 55-cell factorial grid runner & caching
│   ├── lmm_analysis.py                 # Cluster-robust sandwich LMM & changepoint model
│   ├── controls.py                     # Controls C1-C5, baselines, and capability benchmarks
│   ├── gate_evaluation.py              # Hierarchical decision chain (Gates 1-5, Falsifiers 1-5)
│   └── data_loader.py                  # Dataset manager, minimal pairs, typographic renderer
├── scripts/
│   ├── run_phase1.py                   # Execute Phase 1 diagnostics
│   ├── run_phase2.py                   # Execute Phase 2 grid
│   ├── run_controls.py                 # Execute C1-C5 and capability benchmarks
│   ├── run_analysis.py                 # Fit LMM and evaluate pre-registered decision tree
│   ├── plot_paper_figures.py           # Generate publication figures (Figures 1, 2, 3)
│   └── upload_to_hf.py                 # Upload experimental artifacts to Hugging Face
└── outputs/
    ├── figures/                        # High-resolution PNG and PDF publication figures
    └── final_verdict/                  # Final decision tree verdict JSON & summary
```

---

## 🚀 Reproduction & Execution

### 1. Setup Environment
```bash
git clone https://github.com/ankur777jinn/dual-clock-vlm-refusal.git
cd dual-clock-vlm-refusal
pip install -e .
```

### 2. Run Phase 1 Diagnostics
```bash
python scripts/run_phase1.py \
    --data-dir data \
    --output-dir outputs/phase1 \
    --advbench-path data/advbench.txt \
    --alpaca-path data/alpaca.txt \
    --device cuda
```

### 3. Run Phase 2 Factorial Grid (55 cells + Unpatched)
```bash
python scripts/run_phase2.py \
    --data-dir data \
    --output-dir outputs/validation_sweep \
    --split validation \
    --L-read 9 \
    --L-harm 6 \
    --run-nonsafety \
    --device cuda
```

### 4. Run Preconditions & Controls Suite
```bash
python scripts/run_controls.py \
    --data-dir data \
    --output-dir outputs/controls \
    --src-layer 10 \
    --dst-layer 6 \
    --run-capability \
    --device cuda
```

### 5. Statistical Modeling & Decision Chain Evaluation
```bash
python scripts/run_analysis.py \
    --grid-results outputs/validation_sweep/grid_results_validation.parquet \
    --phase1-results outputs/phase1/phase1_results.json \
    --controls-results outputs/controls/controls_and_baselines_results.json \
    --output-dir outputs/final_verdict
```

### 6. Generate Figures & Upload to Hugging Face
```bash
# Generate vector PDF and 300 DPI PNG plots
python scripts/plot_paper_figures.py

# Push results directly to Hugging Face Hub
python scripts/upload_to_hf.py --token <YOUR_HF_TOKEN>
```

---

## 📜 Citation & Reference

```bibtex
@article{dual_clock_vlm_refusal_2026,
  title={An Empirical Investigation into Depth-Limited Cross-Modal Routing and the Early Safety Reflex in Vision-Language Models},
  author={Dual-Clock Refusal Interpretability Collaboration},
  journal={Pre-registered Mechanistic Interpretability Study},
  year={2026},
  url={https://github.com/ankur777jinn/dual-clock-vlm-refusal}
}
```
