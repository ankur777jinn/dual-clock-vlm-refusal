# Comprehensive Investigation Report: Experimental Execution, Empirical Results, Critical Review, and Methodological Remediation

**Target Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (28 Decoder Layers, `bfloat16`)  
**Execution Environment:** MFS Lab Server `vitallab3` (NVIDIA RTX 6000 Ada Generation, 48 GB VRAM)  
**Repository:** [dual-clock-vlm-refusal](https://github.com/ankur777jinn/dual-clock-vlm-refusal.git)  
**Hugging Face Dataset:** [Jinn-69/dual-clock-vlm-refusal-results](https://huggingface.co/datasets/Jinn-69/dual-clock-vlm-refusal-results)  
**Date:** October 2026  

---

## 1. Context & Motivation: Why We Did This Experiment

### 1.1 The Multimodal Safety Bypass Mystery
Modern Vision-Language Models (VLMs) have strong text safety guardrails. When presented with harmful direct text requests (e.g., *"How do I make a Molotov cocktail?"*), models refuse virtually $100\%$ of the time. 

However, when the hazard is depicted in an image and paired with an innocuous prompt:
> *"List step-by-step instructions for the activity shown in this image."*

The refusal rate collapses to nearly **$0\%$**, and the model complies with detailed, dangerous instructions. 

### 1.2 The "Dual-Clock / Three-Timeline" Hypothesis
To explain *why* visual inputs bypass safety mechanisms, a prominent hypothesis in the mechanistic interpretability community posited that the transformer architecture suffers from an **internal latency asynchrony (a cross-modal timing mismatch across layers)**:

```
                      THE HYPOTHESIZED LATENCY GAP:
  Layers:   0 ────────── 10 ──────────── 14 ──────────── 18 ────────── 27
                         ▲                               ▲
                      L_read                          L_harm
                 [Reading Closes]                [Safety Emerges]
                         └────────── GAP ────────────────┘
```

1. **Timeline 1 (Perception):** Low-level visual features are encoded from Layer 0.
2. **Timeline 2 (Reading Window, $L_\text{read} \sim 10\text{--}12$):** Text tokens read from visual tokens only during early layers. In middle and late layers, cross-modal attention is hypothesized to close.
3. **Timeline 3 (Harm Emergence, $L_\text{harm} \sim 16\text{--}18$):** The model realizes that an image contains a dangerous hazard only late in the network, after extensive semantic abstraction.

**The Hypothesized Vulnerability:** By the time the image representation becomes rich enough for safety heads to recognize the hazard ($L \ge 16$), the text tokens have **already stopped reading the image** ($L \le 10$). Safety heads never see text-aligned harm representations while their cross-modal attention is open.

### 1.3 The Proposed Intervention: Activation Back-Patching
If this latency gap were real, one could "bridge" the gap by taking late-layer image representations ($L_\text{src} \ge 16$, where harm semantics are fully formed) and injecting them into early layers ($L_\text{dst} \le 10$, while the reading window is still open). This should restore refusal.

---

## 2. Experimental Execution & Methodology

The experiment was conducted on `Qwen2.5-VL-7B-Instruct` under a pre-registered framework:

```
[Phase 1 Diagnostics] ──► [Preconditions C4 & C5] ──► [Phase 2 Factorial Grid] ──► [Controls C1-C3] ──► [Hierarchical Decision Chain]
   (L_read, L_harm)        (Identity & Typo)           (55 Cells x 100 Images)     (Swaps, Shuffles)            (Verdict)
```

1. **Phase 1 Diagnostics ($N=100$ Images: 50 Harmful HS, 50 Safe SS):**
   * **Diagnostic 1A (Attention Knockout Horizon, $L_\text{read}$):** Pre-softmax cross-attention from text to image tokens was masked with $-\infty$ from layer $l$ to 27 across 5,400 passes to locate where the text stream ceases depending on visual features.
   * **Diagnostic 1B (Harm Emergence, $L_\text{harm}$):** Linear logistic regression probes trained on text-only instructions (AdvBench harmful vs. Alpaca benign) were frozen and evaluated zero-shot on mean-pooled visual tokens across layers $0\dots 27$.
2. **Preconditions:**
   * **C4 Bitwise Identity Check:** Verified that patching a layer into itself produces exact $\max |\Delta z| = 0.000000e+00$ bitwise identity in `bfloat16`.
   * **C5 Typographic Calibration:** Verified that rendering harmful text on a white canvas triggers $100\%$ refusal.
3. **Phase 2 Factorial Back-Patching Grid ($55 \text{ Cells} \times 100 \text{ Images} = 5,600 \text{ Evaluations}$):**
   * $9$ source layers ($10, 12, 14, 15, 16, 17, 18, 20, 22$) $\times$ $7$ destination layers ($4, 6, 8, 10, 12, 13, 14$) with $L_\text{src} > L_\text{dst}$.
   * Preserved M-RoPE position encodings while swapping image token hidden states.
   * Measured refusal logit mass: $M = \text{logsumexp}(\text{refusal}) - \text{logsumexp}(\text{compliance})$.
4. **Confirmatory Controls & Capability:**
   * **C1a Cross-Class Swap:** Injected HS image tokens into SS host images, and vice versa.
   * **C1b Within-Pair Swap:** Evaluated 40 matched semantic pairs.
   * **C1c Same-Class Shuffle:** Random permutation of donors within the same class.
   * **C1d Spatial Shuffle:** Random permutation of the 2D spatial arrangement of visual tokens.
   * **C2 Mean-Source Injection:** Injected global-mean visual vectors.
   * **C3 Norm-Rescaling:** Rescaled patch vectors to match destination layer norms.
   * **Capability Benchmarks:** POPE and MMBench non-inferiority evaluations.

---

## 3. The Empirical Results & Official Verdict

### 3.1 Phase 1 Clock Metrics
* **Visual Reading Window ($L_\text{read}$):**
  * Continuous piecewise linear regression elbow: **Layer 9** ($\Delta\text{BIC} = 36.37$).
  * Strict pre-registered $5\%$ floor rule: **Layer 26**.
* **Harm Emergence Clock ($L_\text{harm}$):**
  * One-sided text probe $\text{AUROC} \ge 0.70$: **Layer 6** ($\text{AUROC} = 0.8249$).
* **Separation Gap ($\Delta L = L_\text{harm} - L_\text{read}$):**
  * $\Delta L = 6 - 9 = \mathbf{-3}$ (or $6 - 26 = \mathbf{-20}$).
  * Pre-registered criterion: $\Delta L \ge +1.5$.
  * **Result:** $\Delta L \le -1.5 \implies$ **Falsifier F1 (Premature Alignment) TRIGGERED**.

### 3.2 Phase 2 Grid Response Surface Summary
| Condition / Quadrant | Cohort | Baseline $M$ | Patched $M$ | Shift ($\Delta M$) | Refusal Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | HS | $2.4034$ | — | — | $88.0\%$ |
| **Clean Baseline** | SS | $1.5240$ | — | — | $8.0\%$ |
| **TooLate Quadrant ($L_\text{dst} > 9$)** | HS | $2.4034$ | $2.4038$ | **$+0.0004$** | $88.0\%$ |
| **TooLate Quadrant ($L_\text{dst} > 9$)** | SS | $1.5240$ | $1.5238$ | **$-0.0002$** | $8.0\%$ |
| **Bridge Quadrant ($L_\text{dst} \le 9$)** | HS | $2.4034$ | $1.2944$ | **$-1.1090$** | $24.0\%$ |
| **Bridge Quadrant ($L_\text{dst} \le 9$)** | SS | $1.5240$ | $0.9541$ | **$-0.5699$** | $4.0\%$ |

### 3.3 Statistical Modeling
* **Linear Mixed Model (LMM):**
  * Interaction parameter ($\beta_\text{three-way}$ scaled): **$-0.6591$** ($\text{SE} = 0.0231, 95\%\text{ CI } [-0.7044, -0.6137]$).
  * One-sided test ($H_1: \beta > 0$): $z = -28.53, p = 1.0000$ (Null not rejected; effect direction inverted).
* **Continuous Broken-Stick Changepoint Competitor:**
  * Globally optimal changepoint identified at **$L_\text{dst} = 6.0$** ($\text{Slope}_\text{pre} = +0.7801, \text{Slope}_\Delta = -0.7053 \implies \text{Slope}_\text{post} = +0.0748$).
* **Official Pre-Registered Verdict:** **`FALSIFIED`** (Chain stopped at Gate 1).

---

## 4. Arnav Bendre's Critiques & Technical Concerns

Arnav Bendre reviewed the empirical findings and raised six profound methodological and interpretive critiques:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ARNAV BENDRE'S 6 CORE CONCERNS                                     │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. AUROC Sign Inversion & Artifact at L=6 (0.0000 AUROC is perfect separation, not zero signal).     │
│ 2. Degenerate Single-Point Confidence Intervals ([6,6], [26,26], [-20,-20] show no real variance).   │
│ 3. L_read Ambiguity & Persistent Reading (Layer 27 knockout still removes 20% of the signal).       │
│ 4. TooLate Null Paradox vs. 1A (Knocking out L10 removes 61% signal, yet back-patching L10-22 is 0).  │
│ 5. Bridge Effect is OOD Circuit Disruption, Not Semantic Bypass (Alien vectors confuse early layers).│
│ 6. Small Sample Size (N=100) & Lack of True Negative Controls (No noise control, no unrelated image).│
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Below is an exhaustive, honest examination of each critique.

---

### Concern 1: The AUROC Sign-Inversion Artifact ($L_\text{harm} = 6$)
* **Arnav's Critique:**
  > *"L_harm = 6 is probably an artifact. In their own 1B table the AUROC is 0.0000 at layer 2, 0.0055 at layer 8, 0.0001 at layer 24, and 0.0000 at layer 27. An AUROC near 0 means the probe separates HS from SS almost perfectly, just with the sign flipped. So the harm/safe distinction is clearly readable at nearly every layer, and layer 6 is just where the sign happens to come out positive. The probe is trained on text and applied to averaged image tokens, which have a different distribution, so its sign and offset aren't meaningful there. Their rule ('first layer with AUROC >= 0.70') ignores all of this, and the verdict rests on that rule."*

* **Technical Analysis & Why He is Right:**
  * In binary classification, $\text{AUROC} = P(w^T x_\text{HS} > w^T x_\text{SS})$.
  * An $\text{AUROC} = 0.50$ represents random chance (no separation).
  * An $\text{AUROC} = 0.0000$ represents **100% perfect discrimination**, with the score inverted ($w^T x_\text{HS} < w^T x_\text{SS}$ for every single pair).
  * If we evaluate the two-sided / absolute discrimination capability:
    * **Layer 2:** $\text{AUROC} = 0.0000 \implies$ **$1.0000$ (100% separable)**
    * **Layer 6:** $\text{AUROC} = 0.8249 \implies$ **$0.8249$ (82.5% separable)**
    * **Layer 8:** $\text{AUROC} = 0.0055 \implies$ **$0.9945$ (99.5% separable)**
    * **Layer 24:** $\text{AUROC} = 0.0001 \implies$ **$0.9999$ (99.99% separable)**
    * **Layer 27:** $\text{AUROC} = 0.0000 \implies$ **$1.0000$ (100% separable)**
  * **The Root Cause:** The probe was trained on text representations and transferred zero-shot to mean-pooled visual representations. Because visual tokens occupy an entirely different manifold (different centroid and principal axes), the text hyperplane intersects the visual distribution with an arbitrary sign and offset.
  * Layer 6 was not where safety semantics "magically emerged"; Layer 6 was merely the single layer where the rotating representation manifold happened to align with positive correlation to the probe vector.

---

### Concern 2: Degenerate Single-Point Confidence Intervals
* **Arnav's Critique:**
  > *"The confidence intervals don't mean anything. Every CI is a single point ([6, 6], [26, 26], gap (-20, -20)). The resampling keeps landing on the same layer, so it shows no uncertainty."*

* **Technical Analysis & Why He is Right:**
  * Bootstrap resampling ($10,000$ iterations) produced degenerate point-mass CIs:
    * For $L_\text{harm}$: Layer 6 had $\text{AUROC} = 0.8249$, while all other layers had positive AUROCs $< 0.30$. In every single bootstrap resample, Layer 6 was the *only* layer crossing $0.70$. Hence, the bootstrap distribution has zero variance: $[6, 6]$.
    * For $L_\text{read}$: The pre-registered $5\%$ threshold floor rule required the signal drop $\Delta_l \le 0.05 \cdot \Delta_1 = 0.0474$. In reality, the drop was $0.2897$ at Layer 26 and $0.1985$ at Layer 27. The floor was never reached in any bootstrap draw, pinning the index to the boundary ($26$) in $100\%$ of iterations.
  * The single-point CIs masked the true epistemic uncertainty of the layer boundaries.

---

### Concern 3: $L_\text{read}$ Ambiguity & Persistent Reading
* **Arnav's Critique:**
  > *"L_read is unclear, and the verdict uses a different value than the write-up. The docs lead with 9, but final_verdict.json uses 26. Even knocking out attention at only layer 27 removes about 20% of the signal, so the model is still reading the image very late."*

* **Technical Analysis & Why He is Right:**
  * There is a dual definition in the study:
    * The narrative emphasized the **regression elbow kink** at **Layer 9** ($\Delta\text{BIC} = 36.37$).
    * But the automated code and `final_verdict.json` evaluated the **strict 5% threshold floor**, which yielded **Layer 26**.
  * More critically: Diagnostic 1A shows that knocking out attention at **Layer 27 alone** drops the cross-modal margin $d'$ from $0.9982$ to $0.7997$ (a **$19.85\%$ loss of signal**).
  * This proves that the text stream never fully terminates its reading of visual tokens; visual information flows into text generation until the very last layer.

---

### Concern 4: The TooLate Null vs. Diagnostic 1A Paradox
* **Arnav's Critique:**
  > *"The TooLate null contradicts 1A. According to 1A, blocking text→image attention from layer 10 onward removes about 61% of the signal, so text is still reading image tokens after layer 10. Swapping those image states for states from layers 10–22 should then change something. Instead the result is essentially zero and refusal stays at exactly 88%. Either late image states barely change between layers, or the patch hook isn't working at those layers. The identity check (C4) can't tell these apart, because pasting identical states back in shows no change even when the hook works. The two phases also use different metrics (a probe score in 1A, refusal-token logits in Phase 2)."*

* **Technical Analysis & Why He is Right:**
  * If blocking attention from Layer 10 onward destroys $61\%$ of the visual signal in 1A, text tokens *must* be attending to image tokens during layers 10–27.
  * Why then did back-patching image tokens across destination layers 10, 12, 13, and 14 produce an exact null shift ($\Delta M = +0.0004$)?
  * **Three Plausible Mechanisms:**
    1. **Representation Stagnation:** Between Layer 10 and Layer 22, image token representations may have already converged in directional space. If $\cos(z_{10}, z_{16}) \approx 0.98\text{--}0.99$, replacing Layer 10 with Layer 16 is mathematically an identity operation.
    2. **Hook Inefficacy / Ignored Tokens:** Control C4 only verifies that injecting $x \to x$ produces $\Delta z = 0$. If downstream attention weights to those tokens are saturated or ignored, $x \to x$ still passes, but injecting $y \to x$ will have zero impact.
    3. **Metric Disconnect:** 1A measured final-layer probe discriminability $d'$, while Phase 2 measured the first-token refusal logit mass $M$. An internal representation shift does not necessarily move output logits past the softmax threshold.

---

### Concern 5: The Bridge Effect is OOD Circuit Disruption, Not Semantic Bypass
* **Arnav's Critique:**
  > *"The Bridge effect looks like disruption, not 'harm'. Safe images shift too. Same-class shuffling keeps 99% of the effect, and spatial shuffling keeps 100%. Together that points to 'late-layer activations dropped into early layers confuse the model', whatever the image is. They present the spatial result as evidence that safety works on a 'bag of tokens'. The size rescaling (C3) only rules out activations being too large. Nothing tests whether the states are simply unlike anything the early layers normally receive."*

* **Technical Analysis & Why He is Right:**
  * This is the single most critical threat to the paper's central thesis:
    * When we back-patch Layer 18 activations into Layer 6, harmful images drop refusal ($\Delta M = -1.109$).
    * But **safe images drop refusal too** ($\Delta M = -0.5699$).
    * Randomly shuffling donor images within the same class retains **$99.1\%$** of the effect.
    * Randomly scrambling the 2D spatial arrangement of tokens retains **$100.2\%$** of the effect.
  * **The "Alien Vector" Confound:** Early transformer layers (Layers 0–6) expect hidden states from Layer 5 with specific layer statistics, channel covariances, and residual geometry.
  * When a late-layer vector from Layer 18 is injected into Layer 6, it acts as an **out-of-distribution (OOD) perturbation**.
  * Early attention heads get disrupted, the refusal circuit fails to fire, and the model defaults to standard un-refused text generation.
  * Our report interpreted $100\%$ spatial shuffle retention as a "bag-of-tokens safety mechanism." But as Arnav correctly notes: **random noise is also spatially invariant!** Any generic disruption vector would show $100\%$ spatial shuffle retention.
  * Control C3 rescaled the Euclidean vector norm ($\|z\|_2$), but did *not* test whether the vector lies within the valid activation subspace of Layer 6.

---

### Concern 6: Small Sample Size ($N=100$) and Lack of Negative Controls
* **Arnav's Critique:**
  > *"The experiment was very crude... like there were no controls and the vectors were extracted from very less images like 100 which are very less."*

* **Technical Analysis & Why He is Right:**
  * **Sample Size ($N=100$):** In the config, $N=100$ was pre-registered as the pilot/validation split. In top VLM safety benchmarks (MM-SafetyBench, VLGuard, JailbreakV-28K), evaluations span $500\text{--}2,000$ images across diverse hazard domains. Drawing sweeping architectural conclusions from 50 harmful and 50 safe images on a single fixed prompt string is vulnerable to sampling bias.
  * **Lack of True Negative Controls:** All controls run (C1a–C1d, C2, C3) used vectors from *within* the experiment. We lacked:
    1. **Matched Gaussian Noise Control:** Does injecting pure random noise with the same norm into Layer 6 produce the exact same refusal drop?
    2. **Unrelated Image Control:** Does injecting representations of an unrelated neutral image (e.g., a coffee mug or landscape) into Layer 6 also suppress refusal?
    3. **Zero-Ablation Control:** What happens if image tokens at Layer 6 are simply zeroed out?

---

## 5. Methodological Remediation: How to Make the Study Bulletproof

To address Arnav's critiques and transform the study into a publishable paper, we define five concrete diagnostic remediation experiments.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   REMEDIATION EXPERIMENTAL PROTOCOL                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Matched Gaussian Noise Control at Layer 6 (Disentangles semantic bypass from circuit disruption).    │
│ 2. Unrelated Image Control (ImageNet / Nature vectors injected at Layer 6).                             │
│ 3. Layer-to-Layer Cosine Similarity Matrix (Explains the TooLate null modulation).                      │
│ 4. Centered / Two-Sided Probe Analysis (Resolves the AUROC sign-inversion artifact).                    │
│ 5. Scaled Evaluation on 3 Key Anchor Cells (N = 500 images across Clean, Bridge, and TooLate).          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Remediation 1: Matched Gaussian Noise Control (The True Disruption Baseline)
* **Protocol:** At the peak Bridge cell $(L_\text{src}=16, L_\text{dst}=6)$, replace the injected image vector with random Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ scaled to match the exact Euclidean norm $\|z_{16}\|_2$.
* **The Decision Rule:**
  * If Gaussian noise produces the exact same refusal drop ($\Delta M \approx -1.10$), **the Bridge effect is 100% generic circuit disruption**.
  * If Gaussian noise produces gibberish or maintains refusal, while the HS image patch produces coherent compliance, **the semantic bypass hypothesis is validated**.

### Remediation 2: Unrelated Natural Image Control
* **Protocol:** Take 50 completely neutral images (e.g., photos of everyday objects, trees, landscapes). Extract their Layer 16 representations and inject them into Layer 6 of harmful images.
* **The Decision Rule:** If benign object representations suppress refusal as much as harmful hazard representations, the effect is non-specific.

### Remediation 3: Layer-to-Layer Cosine Similarity Matrix
* **Protocol:** Compute the pairwise cosine similarity matrix:
  $$S(l, l') = \frac{1}{N} \sum_{i=1}^N \frac{\langle z_l^{(i)}, z_{l'}^{(i)} \rangle}{\|z_l^{(i)}\| \|z_{l'}^{(i)}\|}, \quad \forall l, l' \in [0, 27]$$
* **Expected Outcome:** If $S(10, 20) \ge 0.98$, this mathematically proves why the TooLate quadrant has $\Delta M \approx 0$: late-layer visual representations stagnate, making late back-patching an effective identity operation.

### Remediation 4: Centered & Two-Sided AUROC Analysis
* **Protocol:** Re-evaluate Diagnostic 1B by:
  1. Centering image tokens by subtracting the visual centroid $\bar{x}_\text{vis}$ before probe projection.
  2. Reporting two-sided AUROC: $\text{AUROC}_\text{two-sided} = 2 \cdot |\text{AUROC} - 0.5|$.
* **Expected Outcome:** Formally acknowledge that visual tokens are linearly separable across almost all layers, while documenting the precise rotation of the representation space.

### Remediation 5: Scaled Anchor-Cell Evaluation ($N = 500$)
* Rather than running all 55 cells on 500 images ($27,500$ evaluations), evaluate $N=500$ images on the **3 decisive anchor conditions**:
  1. **Clean Baseline (Unpatched)**
  2. **Peak Bridge Cell:** $(L_\text{src}=16, L_\text{dst}=6)$
  3. **Peak TooLate Cell:** $(L_\text{src}=16, L_\text{dst}=12)$
* This scales statistical power by $5\times$ in under 20 minutes of compute on `vitallab3`.

---

## 6. Summary Comparison Table

| Dimension | Initial Interpretation | Arnav Bendre's Critique | Remediation / True Mechanistic Reality |
| :--- | :--- | :--- | :--- |
| **Harm Emergence ($L_\text{harm}$)** | Safety emerges at Layer 6 ($\text{AUROC} = 0.825$). | $\text{AUROC} \approx 0$ at other layers is perfect separation with inverted sign; Layer 6 is an artifact. | Use two-sided / centered AUROC; visual hazard is separable across nearly all layers. |
| **Reading Window ($L_\text{read}$)** | Reading closes at Layer 9 / 26. | Layer 27 knockout still drops signal by $20\%$; reading never fully terminates. | Distinguish the functional inflection (Layer 9) from residual reading (up to Layer 27). |
| **TooLate Null ($\Delta M = 0$)** | Text stream stops reading past Layer 9. | Contradicts 1A (blocking L10 drops 61% signal); late states may stagnate or patch hook was ineffective. | Measure cross-layer cosine similarity $\cos(z_{10}, z_{20})$ to quantify representation stagnation. |
| **Bridge Suppression ($\Delta M < 0$)** | Late abstract tokens blind the early safety reflex. | Late tokens are alien vectors that generically disrupt early layer attention circuits. | Run **Matched Gaussian Noise** and **Unrelated Image** controls at Layer 6 to isolate semantic vs. disruption effects. |
| **Spatial Invariance ($100.2\%$)** | Safety operates as a "bag-of-tokens" detector. | Random noise is also spatially invariant; disruption effects are inherently order-independent. | Disentangle by testing whether noise disruption exhibits the same spatial invariance. |
| **Sample Size ($N=100$)** | Sufficient for pre-registered validation sweep. | Too small for generalizable claims across multimodal safety. | Scale $N=500$ across the 3 key anchor cells (Clean, Bridge, TooLate). |

---

*This document serves as the formal comprehensive record of the experimental results, the critical peer-review objections raised by Arnav Bendre, and the actionable scientific roadmap for subsequent verification.*
