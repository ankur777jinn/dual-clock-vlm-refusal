"""
Phase 1: Diagnostic Clocks.

Diagnostic 1A: Visual Attention Reading Window (L_read)
  - Cumulative pre-softmax attention knockout from layer l onward
  - Primary readout: change in first-token logit mass M (on typographic images)
  - Secondary readout: decision-token probe margin d' (on natural images)
  - Rule: l* = first layer where Δ_l ≤ 0.05·Δ_1; L_read = l* - 1

Diagnostic 1B: Text-Aligned Harm Emergence (L_harm)
  - Train text probes on AdvBench vs. Alpaca at each layer
  - Evaluate cross-modal transfer on image tokens (mean-pooled)
  - L_harm = first layer where transfer AUROC crosses 0.70
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.config import Config
from src.patching_engine import (
    AttentionKnockoutHook,
    PrefillRunner,
    extract_hidden_states,
    get_image_token_indices,
)
from src.utils import bootstrap_ci, compute_logit_mass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic 1A: Reading Window
# ---------------------------------------------------------------------------

class ReadingWindowDiagnostic:
    """Measures L_read via cumulative attention knockout.

    Primary readout (M-based): evaluated on typographic images where M has
    range. Secondary readout (probe-based): evaluated on natural images using
    a decision-token probe at the final layer.
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        device: torch.device,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self._runner = PrefillRunner(model, processor, device)
        self._image_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )

    def measure_knockout_curve_M(
        self,
        images: list,
        prompt: str,
        refusal_ids: list[int],
        compliance_ids: list[int],
    ) -> pd.DataFrame:
        """Measure Δ_l^M (change in logit mass) under knockout from layer l.

        Evaluates on provided images (typically typographic for dynamic range).

        Returns:
            DataFrame with columns: [layer, image_id, M_clean, M_knockout, delta_M].
        """
        records = []
        n_layers = Config.NUM_LAYERS

        for img_idx, image in enumerate(tqdm(images, desc="Knockout curve (M)")):
            inputs = self._runner._prepare_inputs(image, prompt)
            img_mask = torch.zeros(
                inputs["input_ids"].shape[1], dtype=torch.bool, device=self.device
            )
            img_positions = get_image_token_indices(
                inputs["input_ids"], self._image_token_id
            )
            img_mask[img_positions] = True

            # Clean M
            with torch.no_grad():
                clean_out = self.model(**inputs)
            M_clean = compute_logit_mass(
                clean_out.logits[0, -1, :], refusal_ids, compliance_ids
            )

            # Knockout from each layer l
            for l in range(1, n_layers):
                with AttentionKnockoutHook(self.model, l, img_mask):
                    with torch.no_grad():
                        ko_out = self.model(**inputs)
                M_ko = compute_logit_mass(
                    ko_out.logits[0, -1, :], refusal_ids, compliance_ids
                )
                records.append({
                    "layer": l,
                    "image_id": img_idx,
                    "M_clean": M_clean,
                    "M_knockout": M_ko,
                    "delta_M": abs(M_clean - M_ko),
                })

        return pd.DataFrame(records)

    def measure_knockout_curve_probe(
        self,
        images: list,
        labels: list[int],
        prompt: str,
        probe_layer: int = 27,
    ) -> pd.DataFrame:
        """Measure Δ_l^{d'} (probe margin drop) under knockout.

        Trains a ridge probe on the decision-token at ``probe_layer`` using
        5-fold cross-fitting, then evaluates d' under each knockout condition.

        Args:
            images: List of PIL images.
            labels: Binary labels (1=HS, 0=SS).
            prompt: The prompt string.
            probe_layer: Layer for the decision-token probe (default: 27, final).

        Returns:
            DataFrame with columns: [layer, image_id, d_prime_clean, d_prime_knockout, delta_d_prime].
        """
        n_layers = Config.NUM_LAYERS

        # Step 1: Extract clean decision-token representations at probe_layer
        logger.info(
            "Extracting clean decision-token activations at layer %d...", probe_layer
        )
        clean_reps = []
        for image in tqdm(images, desc="Clean extraction"):
            inputs = self._runner._prepare_inputs(image, prompt)
            states = extract_hidden_states(self.model, inputs, [probe_layer])
            # Decision token is the last token
            decision_rep = states[probe_layer][0, -1, :].float().cpu().numpy()
            clean_reps.append(decision_rep)
        clean_reps = np.array(clean_reps)
        labels_arr = np.array(labels)

        # Step 2: 5-fold cross-fitted probe
        kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)
        d_prime_clean = np.zeros(len(images))

        for train_idx, test_idx in kf.split(clean_reps):
            probe = RidgeClassifier(alpha=1.0)
            probe.fit(clean_reps[train_idx], labels_arr[train_idx])
            scores = probe.decision_function(clean_reps[test_idx])
            d_prime_clean[test_idx] = scores * (2 * labels_arr[test_idx] - 1)

        # Step 3: Under each knockout, re-extract and evaluate
        records = []
        for l in tqdm(range(1, n_layers), desc="Knockout curve (probe)"):
            ko_reps = []
            for image in images:
                inputs = self._runner._prepare_inputs(image, prompt)
                img_mask = torch.zeros(
                    inputs["input_ids"].shape[1], dtype=torch.bool, device=self.device
                )
                img_positions = get_image_token_indices(
                    inputs["input_ids"], self._image_token_id
                )
                img_mask[img_positions] = True

                with AttentionKnockoutHook(self.model, l, img_mask):
                    states = extract_hidden_states(self.model, inputs, [probe_layer])
                ko_reps.append(states[probe_layer][0, -1, :].float().cpu().numpy())
            ko_reps = np.array(ko_reps)

            # Evaluate with the same cross-fitted probes
            d_prime_ko = np.zeros(len(images))
            for train_idx, test_idx in kf.split(ko_reps):
                probe = RidgeClassifier(alpha=1.0)
                probe.fit(clean_reps[train_idx], labels_arr[train_idx])
                scores = probe.decision_function(ko_reps[test_idx])
                d_prime_ko[test_idx] = scores * (2 * labels_arr[test_idx] - 1)

            for i in range(len(images)):
                records.append({
                    "layer": l,
                    "image_id": i,
                    "d_prime_clean": d_prime_clean[i],
                    "d_prime_knockout": d_prime_ko[i],
                    "delta_d_prime": d_prime_clean[i] - d_prime_ko[i],
                })

        return pd.DataFrame(records)

    def find_L_read(
        self,
        knockout_curve: pd.DataFrame,
        readout_col: str = "delta_M",
        threshold_frac: float = 0.05,
    ) -> tuple[int, tuple[int, int]]:
        """Find L_read from knockout curve.

        Rule: l* = first layer where mean Δ_l ≤ threshold_frac × Δ_1.
        L_read = l* - 1.

        Args:
            knockout_curve: DataFrame with [layer, image_id, <readout_col>].
            readout_col: Column name for the attenuation metric.
            threshold_frac: Fraction of Δ_1 that defines the floor (default 0.05).

        Returns:
            (L_read, (ci_low, ci_high)) with 95% bootstrap CI.
        """
        mean_by_layer = (
            knockout_curve.groupby("layer")[readout_col].mean().sort_index()
        )
        delta_1 = mean_by_layer.iloc[0]  # Layer 1 (largest)
        threshold = threshold_frac * delta_1

        l_star = None
        for layer, delta in mean_by_layer.items():
            if delta <= threshold:
                l_star = int(layer)
                break

        if l_star is None:
            logger.warning(
                "Knockout curve never reaches floor; using last layer as l*."
            )
            l_star = int(mean_by_layer.index[-1])

        L_read = l_star - 1

        # Bootstrap CI
        def _boot_L_read(data):
            grouped = data.groupby("layer")[readout_col].mean().sort_index()
            d1 = grouped.iloc[0]
            thresh = threshold_frac * d1
            for lay, d in grouped.items():
                if d <= thresh:
                    return int(lay) - 1
            return int(grouped.index[-1]) - 1

        ci = bootstrap_ci(knockout_curve, _boot_L_read, n_boot=2000)

        logger.info("L_read = %d (95%% CI: %s), l* = %d", L_read, ci, l_star)
        return L_read, ci

    def measure_attention_mass(
        self,
        images: list,
        prompt: str,
    ) -> pd.DataFrame:
        """Measure cumulative text→image attention mass per layer.

        Returns:
            DataFrame with columns: [layer, image_id, attention_mass].
        """
        records = []
        for img_idx, image in enumerate(tqdm(images, desc="Attention mass")):
            inputs = self._runner._prepare_inputs(image, prompt)
            img_positions = get_image_token_indices(
                inputs["input_ids"], self._image_token_id
            )

            # Extract attention weights layer by layer
            with torch.no_grad():
                outputs = self.model(
                    **inputs, output_attentions=True
                )

            for l, attn in enumerate(outputs.attentions):
                # attn shape: (batch, num_heads, seq_len, seq_len)
                # Decision token is last position
                decision_to_image = attn[0, :, -1, img_positions].mean().item()
                records.append({
                    "layer": l,
                    "image_id": img_idx,
                    "attention_mass": decision_to_image,
                })

        df = pd.DataFrame(records)
        # Normalize by layer-1 mass
        if 1 in df["layer"].values:
            layer1_mass = df[df["layer"] == 1]["attention_mass"].mean()
            if layer1_mass > 0:
                df["attention_mass_normalized"] = df["attention_mass"] / layer1_mass
        return df

    def fit_piecewise_sensitivity(
        self,
        knockout_curve: pd.DataFrame,
        readout_col: str = "delta_M",
    ) -> dict:
        """Fit continuous piecewise-linear regression as sensitivity check.

        Returns:
            {"changepoint": int, "bic_piecewise": float, "bic_linear": float,
             "delta_bic": float}
        """
        from scipy.optimize import minimize_scalar
        from scipy.stats import linregress

        mean_by_layer = (
            knockout_curve.groupby("layer")[readout_col].mean().sort_index()
        )
        x = mean_by_layer.index.values.astype(float)
        y = mean_by_layer.values

        # Linear fit
        slope, intercept, _, _, _ = linregress(x, y)
        y_pred_lin = slope * x + intercept
        rss_lin = np.sum((y - y_pred_lin) ** 2)
        n = len(x)
        bic_lin = n * np.log(rss_lin / n) + 2 * np.log(n)  # 2 params

        # Piecewise fit: try each interior point as changepoint
        best_bic_pw = np.inf
        best_cp = None
        for cp_idx in range(2, n - 2):
            cp = x[cp_idx]
            # Fit two segments
            mask1 = x <= cp
            mask2 = x >= cp
            if mask1.sum() < 2 or mask2.sum() < 2:
                continue
            s1, i1, _, _, _ = linregress(x[mask1], y[mask1])
            s2, i2, _, _, _ = linregress(x[mask2], y[mask2])
            y_pred = np.where(x <= cp, s1 * x + i1, s2 * x + i2)
            rss = np.sum((y - y_pred) ** 2)
            bic = n * np.log(rss / n) + 4 * np.log(n)  # 4 params
            if bic < best_bic_pw:
                best_bic_pw = bic
                best_cp = int(cp)

        return {
            "changepoint": best_cp,
            "bic_piecewise": best_bic_pw,
            "bic_linear": bic_lin,
            "delta_bic": bic_lin - best_bic_pw,
        }


# ---------------------------------------------------------------------------
# Diagnostic 1B: Harm Emergence
# ---------------------------------------------------------------------------

class HarmEmergenceDiagnostic:
    """Measures L_harm via cross-modal text probe transfer.

    Trains text probes (L2 logistic regression) on AdvBench vs. Alpaca at each
    layer, then evaluates on mean-pooled image patch tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        device: torch.device,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self._image_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )

    def train_text_probes(
        self,
        harmful_texts: list[str],
        benign_texts: list[str],
        layers: list[int],
    ) -> dict[int, LogisticRegression]:
        """Train L2 logistic regression probes on pure text at each layer.

        Mean-pools across all prompt tokens for the text representation.

        Returns:
            Mapping from layer index to fitted probe.
        """
        probes = {}

        # Tokenize all texts and extract representations
        for l in tqdm(layers, desc="Training text probes"):
            reps = []
            labels = []

            for text in harmful_texts:
                inputs = self.processor(
                    text=[text], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                states = extract_hidden_states(self.model, inputs, [l])
                # Mean-pool across all tokens
                rep = states[l][0].mean(dim=0).float().cpu().numpy()
                reps.append(rep)
                labels.append(1)

            for text in benign_texts:
                inputs = self.processor(
                    text=[text], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                states = extract_hidden_states(self.model, inputs, [l])
                rep = states[l][0].mean(dim=0).float().cpu().numpy()
                reps.append(rep)
                labels.append(0)

            X = np.array(reps)
            y = np.array(labels)

            probe = LogisticRegression(
                penalty="l2", C=1.0, max_iter=1000, solver="lbfgs"
            )
            probe.fit(X, y)
            probes[l] = probe
            logger.debug(
                "Layer %d text probe: train AUROC = %.3f",
                l,
                roc_auc_score(y, probe.predict_proba(X)[:, 1]),
            )

        return probes

    def evaluate_cross_modal_transfer(
        self,
        probes: dict[int, LogisticRegression],
        images: list,
        labels: list[int],
        prompt: str,
        layers: list[int],
        trim_percentile: float = 2.0,
    ) -> pd.DataFrame:
        """Evaluate frozen text probes on mean-pooled image patch tokens.

        Args:
            probes: Text probes from :meth:`train_text_probes`.
            images: List of PIL images.
            labels: Binary labels (1=HS, 0=SS).
            prompt: Prompt string.
            layers: Layers to evaluate.
            trim_percentile: Top % of channels to trim by norm (default 2%).

        Returns:
            DataFrame with [layer, auroc, transfer_score_per_image, ...].
        """
        records = []

        for l in tqdm(layers, desc="Cross-modal transfer"):
            img_reps = []
            for image in images:
                runner = PrefillRunner(self.model, self.processor, self.device)
                inputs = runner._prepare_inputs(image, prompt)
                img_positions = get_image_token_indices(
                    inputs["input_ids"], self._image_token_id
                )

                states = extract_hidden_states(self.model, inputs, [l])
                img_hs = states[l][0, img_positions, :]  # (n_img, d)

                # Zero out top percentile outlier channels by norm (preserving dimension)
                if trim_percentile > 0:
                    norms = img_hs.norm(dim=0)
                    threshold = np.percentile(
                        norms.float().cpu().numpy(), 100 - trim_percentile
                    )
                    outliers = norms > threshold
                    img_hs = img_hs.clone()
                    img_hs[:, outliers] = 0.0

                rep = img_hs.mean(dim=0).float().cpu().numpy()
                img_reps.append(rep)

            X_img = np.array(img_reps)
            y = np.array(labels)

            # NOTE: Trimmed dimensions must match probe input.
            # For now, use the probe trained on full dimensions and
            # handle dimension mismatch by retraining with trimming.
            probe = probes[l]
            try:
                scores = probe.predict_proba(X_img)[:, 1]
                auroc = roc_auc_score(y, scores)
            except ValueError:
                # Dimension mismatch — skip trimming for this evaluation
                logger.warning(
                    "Dimension mismatch at layer %d; re-extracting without trim", l
                )
                img_reps_full = []
                for image in images:
                    runner = PrefillRunner(self.model, self.processor, self.device)
                    inputs = runner._prepare_inputs(image, prompt)
                    img_positions = get_image_token_indices(
                        inputs["input_ids"], self._image_token_id
                    )
                    states = extract_hidden_states(self.model, inputs, [l])
                    rep = states[l][0, img_positions, :].mean(dim=0).float().cpu().numpy()
                    img_reps_full.append(rep)
                X_img = np.array(img_reps_full)
                scores = probe.predict_proba(X_img)[:, 1]
                auroc = roc_auc_score(y, scores)

            for i in range(len(images)):
                records.append({
                    "layer": l,
                    "image_id": i,
                    "auroc": auroc,
                    "transfer_score": scores[i],
                })

        return pd.DataFrame(records)

    def find_L_harm(
        self,
        transfer_curve: pd.DataFrame,
        threshold: float = 0.70,
    ) -> tuple[int, tuple[int, int]]:
        """Find L_harm: first layer where cross-modal AUROC crosses threshold.

        Returns:
            (L_harm, (ci_low, ci_high)) with 95% bootstrap CI.
        """
        auroc_by_layer = (
            transfer_curve.groupby("layer")["auroc"].first().sort_index()
        )

        L_harm = None
        for layer, auroc in auroc_by_layer.items():
            if auroc >= threshold:
                L_harm = int(layer)
                break

        if L_harm is None:
            logger.warning("Transfer AUROC never reaches %.2f; using last layer.", threshold)
            L_harm = int(auroc_by_layer.index[-1])

        # Bootstrap CI over images
        def _boot_L_harm(data):
            by_layer = data.groupby("layer")["auroc"].first().sort_index()
            for lay, a in by_layer.items():
                if a >= threshold:
                    return int(lay)
            return int(by_layer.index[-1])

        ci = bootstrap_ci(transfer_curve, _boot_L_harm, n_boot=2000)

        logger.info("L_harm = %d (95%% CI: %s)", L_harm, ci)
        return L_harm, ci

    def compute_ceiling_normalized_control(
        self,
        harm_aurocs: pd.Series,
        harm_ceilings: pd.Series,
        control_aurocs: pd.Series,
        control_ceilings: pd.Series,
    ) -> pd.DataFrame:
        """Compute ceiling-normalized transfer difference δ_norm(l).

        δ_norm(l) = (AUROC_harm - 0.5) / (ceiling_harm - 0.5)
                   - (AUROC_control - 0.5) / (ceiling_control - 0.5)
        """
        layers = harm_aurocs.index
        records = []
        for l in layers:
            h = harm_aurocs.get(l, 0.5)
            hc = harm_ceilings.get(l, 0.5)
            c = control_aurocs.get(l, 0.5)
            cc = control_ceilings.get(l, 0.5)

            norm_harm = (h - 0.5) / max(hc - 0.5, 1e-6)
            norm_ctrl = (c - 0.5) / max(cc - 0.5, 1e-6)
            records.append({"layer": l, "delta_norm": norm_harm - norm_ctrl})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phase1(
    model: nn.Module,
    processor,
    dataset_manager,
    text_partitioner,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Orchestrate full Phase 1 diagnostics.

    Returns:
        Dictionary with L_read, L_harm, CIs, gap, gate1 evaluation,
        and paths to saved plots/data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare data ---
    pool = dataset_manager.get_phase1_pool()
    hs_images = [s for s in pool if s.cohort.name == "HS"]
    ss_images = [s for s in pool if s.cohort.name == "SS"]
    all_images = hs_images + ss_images
    all_labels = [1] * len(hs_images) + [0] * len(ss_images)

    probe_texts, _ = text_partitioner.cluster_and_split()

    # Separate AdvBench (harmful) and Alpaca (benign) from probe partition
    harmful_candidates = set(getattr(text_partitioner, "_harmful_strings", getattr(text_partitioner, "_advbench", [])))
    benign_candidates = set(getattr(text_partitioner, "_benign_strings", getattr(text_partitioner, "_alpaca", [])))
    harmful_texts = [t for t in probe_texts if t in harmful_candidates]
    benign_texts = [t for t in probe_texts if t in benign_candidates]

    # --- 1A: Reading Window ---
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC 1A: Reading Window (L_read)")
    logger.info("=" * 60)

    rw_diag = ReadingWindowDiagnostic(model, processor, device)

    # Primary: M-based on typographic images (TODO: load typographic images)
    # For now, use natural images with probe-based readout
    logger.info("Running probe-based knockout curve on natural images...")
    # Load actual images from ImageSample paths
    from PIL import Image
    pil_images = [Image.open(s.path).convert("RGB") for s in all_images]

    ko_curve = rw_diag.measure_knockout_curve_probe(
        pil_images, all_labels, Config.PROMPT
    )
    ko_curve.to_parquet(output_dir / "knockout_curve_probe.parquet")

    L_read, L_read_ci = rw_diag.find_L_read(ko_curve, readout_col="delta_d_prime")

    # Sensitivity: piecewise regression
    pw = rw_diag.fit_piecewise_sensitivity(ko_curve, readout_col="delta_d_prime")
    logger.info("Piecewise sensitivity: changepoint=%s, ΔBIC=%.2f", pw["changepoint"], pw["delta_bic"])

    # --- 1B: Harm Emergence ---
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC 1B: Harm Emergence (L_harm)")
    logger.info("=" * 60)

    he_diag = HarmEmergenceDiagnostic(model, processor, device)

    layers = list(range(0, Config.NUM_LAYERS, 2)) + [Config.NUM_LAYERS - 1]
    layers = sorted(set(layers))

    probes = he_diag.train_text_probes(harmful_texts, benign_texts, layers)

    transfer_curve = he_diag.evaluate_cross_modal_transfer(
        probes, pil_images, all_labels, Config.PROMPT, layers
    )
    transfer_curve.to_parquet(output_dir / "transfer_curve.parquet")

    L_harm, L_harm_ci = he_diag.find_L_harm(transfer_curve)

    # --- Gate 1 Evaluation ---
    gap = L_harm - L_read
    gap_ci_low = L_harm_ci[0] - L_read_ci[1]
    gap_ci_high = L_harm_ci[1] - L_read_ci[0]

    ci_width = L_harm_ci[1] - L_harm_ci[0] + L_read_ci[1] - L_read_ci[0]
    gate1_pass = (
        gap_ci_low >= Config.THRESHOLDS.DELTA_LAYER_MEI
        and ci_width <= Config.THRESHOLDS.CI_WIDTH_MAX_LAYERS
    )

    results = {
        "L_read": int(L_read),
        "L_read_ci": [int(x) for x in L_read_ci],
        "L_harm": int(L_harm),
        "L_harm_ci": [int(x) for x in L_harm_ci],
        "gap": int(gap),
        "gap_ci": (float(gap_ci_low), float(gap_ci_high)),
        "gate1_pass": bool(gate1_pass),
        "piecewise_sensitivity": pw,
        "plots_dir": str(output_dir),
    }

    logger.info("=" * 60)
    logger.info("PHASE 1 SUMMARY")
    logger.info("L_read = %d (CI: %s)", L_read, L_read_ci)
    logger.info("L_harm = %d (CI: %s)", L_harm, L_harm_ci)
    logger.info("Gap = %d layers (CI: %s)", gap, (gap_ci_low, gap_ci_high))
    logger.info("Gate 1: %s", "PASS" if gate1_pass else "FAIL")
    logger.info("=" * 60)

    return results
