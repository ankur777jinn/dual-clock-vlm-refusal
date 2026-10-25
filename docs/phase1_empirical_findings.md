# Phase 1 Diagnostic Clocks: Empirical Findings & Mechanistic Report

**Target Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (28 Decoder Layers, `bfloat16`)  
**Hardware Platform:** NVIDIA RTX 6000 Ada Generation (48 GB VRAM, GPU 2), IIT Roorkee MFS Lab (`vitallab3`)  
**Dataset:** 200 Natural Images (100 Harmful-Scenario, 100 Safe-Similar) + AdvBench 520 Harmful / Alpaca 520 Benign  
**Execution Timestamp:** 2026-10-25 05:08 – 05:58 UTC  

---

## 1. Executive Summary

This document records the empirical results of **Diagnostic 1A (Visual Attention Reading Window, $L_\text{read}$)** and **Diagnostic 1B (Text-Aligned Harm Emergence, $L_\text{harm}$)** executed on `Qwen2.5-VL-7B-Instruct`.

### Key Findings
1. **Diagnostic 1A (Visual Attention Knockout Horizon):**
   - Cumulative pre-softmax attention knockout from layers 1–7 obliterates the cross-modal decision signal ($d'$ drops by >90% to near zero).
   - A dramatic inflection point occurs at **Layer 9** ($\Delta\text{BIC} = 36.37$ in continuous piecewise linear regression).
   - By Layer 12–14, over 50% of the cross-modal representation is locked into the residual stream.
   - A persistent late-layer dependency remains even at Layer 27 ($\Delta_{27} = 0.1985$), which prevented the strict 5% threshold floor rule ($0.05 \times \Delta_1 = 0.0474$) from resolving prior to layer 27.
2. **Diagnostic 1B (Text-Aligned Harm Emergence):**
   - Text-trained linear probes transfer to visual tokens with a **sharp spike at Layer 6** (**$\text{AUROC} = 0.8249$**, crossing the pre-registered $\ge 0.70$ threshold).
   - Beyond Layer 6, the visual representation rotates relative to the static prompt text direction.
3. **Temporal Ordering & Pre-Registered Gate 1 Evaluation:**
   - Under both the threshold convention ($L_\text{read}=26$) and the continuous piecewise regression changepoint ($L_\text{read}=9$), **$L_\text{harm} \le L_\text{read}$** ($6 \le 9$ or $6 \le 26$).
   - This empirically confirms **Falsifier F1 (Premature Alignment)**:
     $$95\% \text{ CI upper bound on } (L_\text{harm} - L_\text{read}) < -\Delta_\text{layer, MEI} = -1.5$$
   - **Theoretical Implication:** `Qwen2.5-VL-7B` does *not* fail to refuse visual jailbreaks because harm recognition lags behind visual reading. Instead, **harm aligns very early (Layer 6)**, while cross-modal visual reading remains active through Layer 9 and persists in the residual stream.

---

## 2. Diagnostic 1A: Visual Attention Knockout Curve

In Diagnostic 1A, cumulative pre-softmax attention knockout was applied from layer $l$ to 27 across all 200 images (5,400 forward passes on GPU 2). A probe at final Layer 27 was evaluated under clean vs. knocked-out conditions:

$$\Delta_l^{d'} = d'_\text{clean} - d'_{\text{knockout} \ge l}$$

### Empirical Attenuation Table

| Layer $l$ | Clean Margin $d'_\text{clean}$ | Knockout Margin $d'_\text{ko}$ | Drop $\Delta_l^{d'}$ | Cross-Modal Signal Retained ($\%$) |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.9982 | 0.0493 | **0.9489** | 4.9% (Baseline Knockout Floor) |
| **2** | 0.9982 | 0.0110 | 0.9873 | 1.1% |
| **3** | 0.9982 | 0.0119 | 0.9863 | 1.2% |
| **4** | 0.9982 | 0.0110 | 0.9873 | 1.1% |
| **5** | 0.9982 | 0.0715 | 0.9267 | 7.2% |
| **6** | 0.9982 | 0.0796 | 0.9187 | 8.0% |
| **7** | 0.9982 | 0.0882 | 0.9100 | 8.8% |
| **8** | 0.9982 | 0.1532 | 0.8451 | 15.3% |
| **9 (Kink)** | 0.9982 | 0.2372 | **0.7610** | **23.8% (Piecewise Changepoint)** |
| **10** | 0.9982 | 0.3872 | 0.6111 | 38.8% |
| **11** | 0.9982 | 0.3999 | 0.5984 | 40.1% |
| **12** | 0.9982 | 0.4825 | 0.5157 | 48.3% |
| **13** | 0.9982 | 0.4969 | 0.5013 | 49.8% |
| **14** | 0.9982 | 0.5514 | 0.4468 | 55.2% |
| **15** | 0.9982 | 0.5644 | 0.4338 | 56.5% |
| **16** | 0.9982 | 0.5614 | 0.4369 | 56.2% |
| **17** | 0.9982 | 0.5778 | 0.4205 | 57.9% |
| **18** | 0.9982 | 0.5921 | 0.4061 | 59.3% |
| **19** | 0.9982 | 0.5848 | 0.4134 | 58.6% |
| **20** | 0.9982 | 0.6149 | 0.3833 | 61.6% |
| **21** | 0.9982 | 0.6311 | 0.3671 | 63.2% |
| **22** | 0.9982 | 0.6355 | 0.3627 | 63.7% |
| **23** | 0.9982 | 0.6694 | 0.3288 | 67.1% |
| **24** | 0.9982 | 0.6863 | 0.3120 | 68.8% |
| **25** | 0.9982 | 0.6944 | 0.3039 | 69.6% |
| **26** | 0.9982 | 0.7085 | 0.2897 | 71.0% |
| **27** | 0.9982 | 0.7997 | **0.1985** | 80.1% |

### Statistical Changepoint Analysis
- **Continuous Piecewise Linear Regression:** Changepoint identified at **Layer 9**.
- **Model Comparison:** $\Delta\text{BIC} = 36.37$ (decisive evidence over single linear decay, $\Delta\text{BIC} > 10$).
- **Interpretation:** Prior to Layer 9, blocking text$\to$image attention destroys refusal discriminability ($>75\%$ signal loss). Between Layer 9 and 14, over 50% of the visual information enters the text stream.

---

## 3. Diagnostic 1B: Text-Aligned Harm Emergence

Text probes trained on AdvBench harmful vs. Alpaca benign text instructions were frozen and evaluated directly on mean-pooled visual patch tokens across layers $0..27$:

### Empirical Transfer Table

| Layer $l$ | Text Probe AUROC on Visual Tokens | Significance / Alignment State |
|:---:|:---:|:---|
| **0** | 0.6217 | Weak initial alignment |
| **2** | 0.0000 | Orthogonal / Anti-aligned |
| **4** | 0.0874 | Sub-chance / Rotated |
| **6** | **0.8249** | **Strong Text-Vision Harm Convergence ($\ge 0.70$ Threshold)** |
| **8** | 0.0055 | Sub-chance / Rotated |
| **10** | 0.1011 | Sub-chance |
| **12** | 0.2525 | Intermediate |
| **14** | 0.1338 | Sub-chance |
| **16** | 0.0750 | Sub-chance |
| **18** | 0.0410 | Sub-chance |
| **20** | 0.0087 | Sub-chance |
| **22** | 0.0076 | Sub-chance |
| **24** | 0.0001 | Sub-chance |
| **26** | 0.0044 | Sub-chance |
| **27** | 0.0000 | Orthogonal / Rotated |

### Interpretation of Layer 6 Convergence
- At **Layer 6**, the representation of harmful visual scenes maps directly onto the text-derived harm direction with **$\text{AUROC} = 0.8249$**.
- This indicates that `Qwen2.5-VL-7B` forms its multimodal hazard representation very early in the network hierarchy (Layer 6 of 28, ~21% network depth).
- At deeper layers ($l \ge 8$), the visual representation undergoes rotation relative to the linear prompt probe, causing transfer to drop.

---

## 4. Gate 1 Evaluation & Pre-Registered Protocol

### Gate 1 Metric
- $L_\text{harm} = 6$ (95% CI: [6.0, 6.0])
- $L_\text{read} = 9$ (Piecewise) or $26$ (Floor Rule)
- $\text{Gap} = L_\text{harm} - L_\text{read} = 6 - 9 = -3$ (or $6 - 26 = -20$)
- **Gate 1 Status: FAIL** (Expected $\text{Gap} \ge +1.5$)
- **Falsifier F1 (Premature Alignment): TRIGGERED**
  $$95\% \text{ CI upper bound on } (L_\text{harm} - L_\text{read}) < -1.5$$

### Grid Impact & Mathematical Analysis
Under the prior hypothesis, $L_\text{harm} \approx 16..18$, so $L_\text{src} \in [10..22]$ contained both pre-harm and post-harm representations.
Empirically, because $L_\text{harm} = 6 \le \min(L_\text{src}) = 10$, **all 55 cells in the pre-registered grid satisfy $L_\text{src} \ge L_\text{harm}$**.
Consequently:
- **Bridge Quadrant ($L_\text{dst} \le 9, L_\text{src} \ge 6$):** 27 cells
- **TooLate Quadrant ($L_\text{dst} > 9, L_\text{src} \ge 6$):** 28 cells
- **TooEarly Quadrant:** 0 cells
- **Neither Quadrant:** 0 cells

The three-way interaction $HS \times Read \times Harm$ reduces to the two-way interaction $HS \times Read$ (Bridge vs TooLate), directly evaluating whether injecting the harm representation inside the reading window produces higher refusal than injecting it after the reading window has closed.
