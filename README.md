# Depth-Limited Cross-Modal Routing in VLMs

**The Three-Timeline Framework for Visual Safety Bypass**

This repository implements the experimental pipeline for investigating *why* Vision-Language Models with robust text safety guardrails (>98% refusal) comply with harmful requests when hazards are depicted in natural images (refusal ≤1%).

## The Thesis

The safety bypass occurs because of a **depth-limited routing and semantic translation asynchrony** in the transformer:

1. **Timeline 1 (Visual Encoding):** Low-level visual harm features are decodable from Layer 0 (AUROC ≥ 0.95).
2. **Timeline 2 (Reading Window):** Text tokens read image tokens only during early layers (L ≤ L_read ≈ 10–12).
3. **Timeline 3 (Semantic Translation):** Image tokens acquire text-aligned harm semantics only at deep layers (L ≥ L_harm ≈ 16–18).

The **gap** between Timelines 2 and 3 means safety heads never see text-aligned harm representations while their cross-modal attention is open.

**Back-patching** bridges this gap by injecting late-layer representations into the early reading window.

## Target Model

`Qwen2.5-VL-7B-Instruct` (28 transformer layers, bf16)

## Repository Structure

```
dual-clock-vlm-refusal/
├── pyproject.toml              # Project metadata and dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration (grid, thresholds, paths)
│   ├── data_loader.py          # Dataset management, splits, typographic rendering
│   ├── utils.py                # Common utilities (logit mass, bootstrap, caching)
│   ├── patching_engine.py      # Attention knockout & back-patching hooks
│   ├── phase1_diagnostics.py   # Diagnostic clocks (L_read, L_harm)
│   ├── phase2_grid.py          # Factorial grid execution & judge evaluation
│   ├── lmm_analysis.py         # Mixed-effects models & statistical tests
│   ├── controls.py             # Controls C1–C5, Baselines B1–B5, capability
│   └── gate_evaluation.py      # Hierarchical gatekeeping decision framework
├── scripts/
│   ├── run_phase1.py           # Phase 1 diagnostic execution
│   ├── run_phase2.py           # Phase 2 grid execution
│   ├── run_controls.py         # Controls and baselines
│   ├── run_analysis.py         # Full statistical analysis pipeline
│   └── power_simulation.py     # Power simulation for preregistration
└── tests/
    └── test_config.py          # Smoke tests for grid arithmetic
```

## Execution Order

1. **Preconditions:** Run C4 (bitwise identity) and C5 (typographic calibration).
2. **Phase 1 (Pool, n=200):** Measure L_read and L_harm diagnostic clocks.
3. **Pilot Margins:** Estimate σ_M, run power simulations, set margins.
4. **Preregister:** Commit hash to OSF/AsPredicted.
5. **Phase 2 (Test, n=200):** Run 55-cell factorial grid + unpatched reference.
6. **Analysis:** Fit LMM, evaluate hierarchical gatekeeping chain.
7. **Controls & Capability:** Run C1–C3, baselines B1–B5, POPE/MMBench.

## Key Design Choices

- **Fixed 55-cell grid** (9 src × 7 dst, triangular): not adaptive.
- **Centered effect coding (±0.5):** eliminates artificial multicollinearity.
- **Hierarchical gatekeeping** at one-sided α = 0.025 (6 gates).
- **Margins set from pilot** (Tuning/Validation), committed before Test.
- **Gate 2A/Falsifier 2 mutual exclusion:** Falsifier only evaluated if Gate fails.

## Citation

```
@misc{bendre2026depthlimited,
  title={Depth-Limited Cross-Modal Routing in Vision-Language Models},
  author={Bendre, Arnav},
  year={2026}
}
```
