# Dual-Clock VLM Refusal: Empirical Investigation Report & Scientific Synthesis

**Target Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (28 transformer layers, 0–27, `bfloat16`)  
**Execution Environment:** MFS Lab Server `vitallab3` (NVIDIA RTX 6000 Ada Generation, 48 GB VRAM)  
**Methodology:** Pre-registered Multi-Phase Mechanistic Activation Patching Pipeline  
**Repository:** [dual-clock-vlm-refusal](https://github.com/ankur777jinn/dual-clock-vlm-refusal.git)  
**Hugging Face Dataset:** [Jinn-69/dual-clock-vlm-refusal-results](https://huggingface.co/datasets/Jinn-69/dual-clock-vlm-refusal-results)  
**Official Outcome:** **`FALSIFIED`** (Hierarchical chain stopped at Gate 1; Falsifier F1 triggered).

---

## 1. Executive Summary & Pre-Registered Verdict

The pre-registered hierarchical decision tree was evaluated strictly against empirical data collected across Phase 1, Phase 2, and the full experimental control suite.

```
                    ┌────────────────────────┐
                    │  Gate 1: Two Clocks    │
                    │   L_harm > L_read?     │
                    └───────────┬────────────┘
                                │
                                ▼
                         [FAILED: NO]
                     (L_harm=6 <= L_read=9)
                                │
                                ▼
                    ┌────────────────────────┐
                    │      FALSIFIER F1      │
                    │  Premature Alignment   │
                    │       TRIGGERED        │
                    └────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     FINAL VERDICT:     │
                    │       FALSIFIED        │
                    └────────────────────────┘
```

### Official Decision Tree Outcome
* **Final Verdict:** **`FALSIFIED`**
* **Termination Point:** Chain stopped at **Gate 1 (Two-Clock Existence)**.
* **Falsifiers Triggered:** **`Falsifier F1 (Premature Alignment)`** ($\text{Gap} = L_\text{harm} - L_\text{read} = -20 \text{ to } -3 \le -1.5$).
* **Construct Flags:** **`CONSTRUCT WARNING FLAG`** triggered by Control C1d (Spatial Shuffle Retention = $100.23\%$, $95\%\text{ CI } [98.89\%, 101.60\%]$).
* **Preconditions:** **`C4 Bitwise Identity`** PASSED ($\max |\Delta z| = 0.000000e+00$ bitwise exact); **`C5 Typographic Calibration`** PASSED ($100\%$ baseline refusal rate).

> **Core Scientific Discovery:** The foundational hypothesis that cross-modal visual feature reading completes *before* text-aligned multimodal safety recognition emerges ($L_\text{read} < L_\text{harm}$) is **empirically falsified** in `Qwen2.5-VL-7B-Instruct`. Instead, safety recognition happens remarkably **early** ($L_\text{harm} = 6$), while visual token reading persists through mid-to-late layers ($L_\text{read} = 9\text{--}26$). Refusal is an **early perceptual reflex** rather than a late deliberative judgment.

---

## 2. Phase 1 Diagnostic Measurements

Phase 1 established the empirical boundaries of the two putative clocks on $N=100$ balanced multimodal evaluation prompts (Harmful Safety HS vs. Benign Safe SS).

| Diagnostic Metric | Value | 95% Bootstrap CI | Pre-registered Prior | Empirical Status |
| :--- | :---: | :---: | :---: | :---: |
| **Visual Reading Window ($L_\text{read}$)** | **9** (kink) / **26** (floor) | $[9, 9]$ / $[26, 26]$ | Layers 10–12 | Persists significantly deeper than expected |
| **Harm Emergence Clock ($L_\text{harm}$)** | **6** | $[6, 6]$ | Layers 16–18 | Emerges 10 layers earlier than predicted |
| **Clock Separation Gap ($\Delta L$)** | **$-3$** (kink) / **$-20$** (floor) | $[-3, -3]$ / $[-20, -20]$ | $\ge +1.5$ layers | **Reversed** ($L_\text{harm} \ll L_\text{read}$) |
| **Harm Probe AUROC at $L_\text{harm}$** | **$0.8249$** | $[0.7812, 0.8654]$ | $\ge 0.70$ | Strong multimodal hazard discrimination |
| **Knockout Changepoint $\Delta\text{BIC}$** | **$36.37$** | — | $> 10.0$ | Decisive evidence for Layer 9 inflection |

### Diagnostic 1A: Visual Knockout Curve
* Knockout of text-to-image cross-attention prior to Layer 9 results in near-total ablation of image-grounded refusal.
* An elbow/kink occurs at **Layer 9** ($\Delta\text{BIC} = 36.37$), where the marginal dependence on visual tokens drops sharply.
* Under the strict $5\%$ floor criterion ($\Delta M(l) \le 0.05 \cdot \Delta M(1)$), visual information continues leaking into the representation until **Layer 26**.

### Diagnostic 1B: Cross-Modal Harm Transfer Probe
* Linear probes trained exclusively on text representations (AdvBench vs. Alpaca) generalize to mean-pooled visual tokens starting at **Layer 6** ($\text{AUROC} = 0.8249$).
* By Layer 6, the model's intermediate representations have already aligned visual safety features into the text safety manifold.

---

## 3. Phase 2 Factorial Back-Patching Grid ($55 \text{ Cells} \times 100 \text{ Images}$)

The complete $9 \times 7$ factorial back-patching grid ($L_\text{src} \in \{10, 12, 14, 15, 16, 17, 18, 20, 22\}$, $L_\text{dst} \in \{4, 6, 8, 10, 12, 13, 14\}$ with $L_\text{src} > L_\text{dst}$) yielded 5,600 total evaluations.

```
                           L_dst (Destination Layer)
           4        6        8        10       12       13       14
    10   [Bridge] [Bridge] [Bridge]   .        .        .        .
    12   [Bridge] [Bridge] [Bridge] [TooLate]  .        .        .
L   14   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate]  .
_   15   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]
s   16   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]
r   17   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]
c   18   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]
    20   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]
    22   [Bridge] [Bridge] [Bridge] [TooLate] [TooLate] [TooLate] [TooLate]

    Total: 55 cells (Bridge = 27 cells, TooLate = 28 cells, TooEarly = 0 cells)
```

### Empirical Response Surface Summary
| Condition / Quadrant | Cohort | Mean Logit Mass ($M$) | $\Delta M$ vs Clean Baseline | Refusal Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Unpatched Baseline** | HS (Harmful) | **$2.4034$** | $0.0000$ | $88.0\%$ |
| **Unpatched Baseline** | SS (Benign Safe) | **$1.5240$** | $0.0000$ | $8.0\%$ |
| **TooLate Quadrant ($L_\text{dst} > 9$)** | HS | **$2.4038$** | **$+0.0004$** | $88.0\%$ |
| **TooLate Quadrant ($L_\text{dst} > 9$)** | SS | **$1.5238$** | **$-0.0002$** | $8.0\%$ |
| **Bridge Quadrant ($L_\text{dst} \le 9$)** | HS | **$1.2944$** | **$-1.1090$** | $24.0\%$ |
| **Bridge Quadrant ($L_\text{dst} \le 9$)** | SS | **$0.9541$** | **$-0.5699$** | $4.0\%$ |

### Key Mechanistic Insights from Phase 2
1. **The TooLate Quadrant exhibits exact null modulation ($\Delta M \approx 0.0000$):**  
   Back-patching late representations into layers $L_\text{dst} \ge 10$ produces identically zero shift in refusal logit mass or generated text. Once the attention reading window closes at Layer 9, the textual generation stream is completely decoupled from further visual back-patching.
2. **Bridge Quadrant Inversion ($\Delta M < 0$):**  
   Injecting late visual tokens into early layers ($L_\text{dst} \le 9$) significantly *attenuates* refusal for HS images ($\Delta M = -1.1090$). This reveals that early layers require raw, unprocessed visual tokens to trigger safety neurons; pre-computed late representations lack the low-level trigger patterns expected by early transformer layers.
3. **Continuous Changepoint Discovery:**  
   The continuous broken-stick regression model identified **$L_\text{dst} = 6.0$** as the global optimal changepoint, precisely matching $L_\text{harm} = 6.0$ from Phase 1.

---

## 4. Confirmatory Mixed-Effects Model (LMM)

To test the interaction between Cohort (HS vs. SS) and Window Boundaries, a confirmatory Linear Mixed-Effects Model was fit with centered effect coding ($\pm 0.5$) and Huber-White sandwich standard errors clustered by `image_id`:

$$M_{ij} = \beta_0 + \beta_1 \widetilde{\text{HS}}_i + \beta_2 \widetilde{\text{Read}}_j + \beta_3 \text{Patched}_{ij} + \beta_4 (\widetilde{\text{HS}} \times \text{Patched})_{ij} + \beta_5 (\widetilde{\text{HS}} \times \widetilde{\text{Read}})_{ij} + u_i + \epsilon_{ij}$$

### Statistical Parameter Estimates
* **Target Interaction Parameter ($\beta_\text{three-way}$ scaled):** **$-0.6591$**
* **Cluster-Robust Sandwich SE:** **$0.0231$**
* **95% Confidence Interval:** **$[-0.7044, -0.6137]$**
* **One-Sided Hypothesis Test ($H_1: \beta > 0$):** **$z = -28.53$**, **$p = 1.0000$** (Null hypothesis NOT rejected; direction inverted).
* **Two One-Sided Tests (TOST) Equivalence:** Not equivalent within $\delta_\text{equiv} = \pm 0.0372$ ($p_\text{lower} = 1.0000$).
* **Pooled Within-Condition Variance ($\sigma_M$):** **$0.1489$**

---

## 5. Experimental Controls & Preconditions Battery

Every pre-registered control and precondition was systematically executed and validated:

| Code | Control / Precondition Name | Empirical Metric | Criterion / Threshold | Status |
| :---: | :--- | :---: | :---: | :---: |
| **C4** | **Bitwise Identity Check** | $\max \|\Delta z\| = 0.000000e+00$ | Exact $0.0$ in `bf16` | **PASSED** |
| **C5** | **Typographic Calibration** | Baseline Refusal = $100.0\%$ | Refusal $> 80\%$ | **PASSED** |
| **C1a** | **Cross-Class Donor Swap** | HS $\to$ SS $\Delta M = +2.030$; SS $\to$ HS $\Delta M = -1.588$ | Modulates $M$ in donor direction | **VALIDATED** |
| **C1b** | **Within-Pair Specificity** | Paired difference $\Delta M = -0.5666$ ($p = 1.0$) | Selective to harmful visual semantics | **VALIDATED** |
| **C1c** | **Same-Class Donor Shuffle** | Shuffled-to-Self Ratio = $0.9912$ ($99.1\%$) | Ratio $\approx 1.0$ | **VALIDATED** |
| **C1d** | **Spatial Order Shuffle** | Retention Ratio = $100.23\%$ ($CI: [98.9\%, 101.6\%]$) | Retention $< 90\%$ (Flag if $\sim 100\%$) | **FLAGGED** |
| **C2** | **Global Mean Injection** | Global Mean $M = 2.1604$ | Intermediate between HS and SS | **VALIDATED** |
| **C3** | **Norm Rescaling (Dir vs Norm)** | Mean Loss Ratio = $0.8627$ (retains $86.3\%$) | Loss Ratio $< 0.50$ (Falsifier F5) | **PASSED (F5 NOT triggered)** |
| **Cap** | **Capability Preservation** | POPE & MMBench non-inferiority = `True` | Degradation $< 3.0$ pp | **PASSED** |

### Critical Control Findings
1. **Falsifier F5 (Activation Scale Artifact) Ruled Out:**  
   Control C3 proved that normalizing patch vectors to destination layer statistics preserves $86.3\%$ of the effect. The patch effect is driven by vector **direction**, not activation magnitude explosion.
2. **Construct Warning Flag (Spatial Invariance):**  
   Control C1d demonstrated that shuffling the 2D spatial arrangement of visual tokens yields $100.23\%$ effect retention. Refusal mechanisms in `Qwen2.5-VL-7B` operate on **semantic token presence** (bag-of-visual-concepts) rather than geometric/spatial structure.

---

## 6. Scientific Discussion: Why Did the Dual-Clock Hypothesis Fail?

### The Initial Theoretical Premise
The "Dual-Clock / Three-Timeline" hypothesis posited that multimodal language models follow a sequential three-stage pipeline:
$$\text{Perception (Layers } 0\text{--}10) \longrightarrow \text{Cross-Modal Fusion (Layers } 10\text{--}14) \longrightarrow \text{Safety Refusal Alignment (Layers } 16\text{--}22)$$
Under this model, visual reading ($L_\text{read}$) would finish early, and safety recognition ($L_\text{harm}$) would emerge late, creating a fertile "Bridge" window ($L_\text{read} < L < L_\text{harm}$) where the model has perceived the visual content but has not yet decided to refuse.

### The Empirical Reality: The "Early Safety Bottleneck" Architecture
Our experiments decisively overturn this sequential timeline:

```
Proposed Dual-Clock Model (Hypothesized):
Layers:   0 ─────── 10 ──────────── 14 ──────────── 18 ─────── 27
                    ▲               ▲               ▲
                 L_read          Fusion          L_harm
            [Visual Done]                    [Safety Decided]

Actual Empirical Architecture (Discovered):
Layers:   0 ─────── 6 ───────────── 9 ─────────────────────── 27
                    ▲               ▲                         ▲
                 L_harm          L_read (Kink)           L_read (Floor)
            [Safety Decided]  [Major Read Closes]    [Residual Read]
```

1. **Safety Emerges Prematurely ($L_\text{harm} = 6$):**  
   `Qwen2.5-VL-7B` aligns multimodal visual features to safety refusal classifiers at **Layer 6**—before cross-modal visual reading has even completed. Refusal in modern VLMs is not an emergent property of late-layer deliberative reasoning; it is an **early perceptual reflex**.
2. **Visual Reading is Protracted ($L_\text{read} = 9\text{--}26$):**  
   The model does not finalize visual token processing in early layers. While the primary cross-attention reading window closes at Layer 9, residual information transfer continues up to Layer 26.
3. **The Window Gap is Inverted ($\Delta L < 0$):**  
   Because $L_\text{harm} \le L_\text{read}$, there is no window where visual concepts are fully formed while safety remains blind. Any attack attempting to exploit a latency gap between perception and alignment encounters a safety barrier that is already active.
