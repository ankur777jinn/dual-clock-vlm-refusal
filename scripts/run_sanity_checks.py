#!/usr/bin/env python3
"""
Phase 0 Sanity Checks & Representation Geometry:
1. Real-Path Patch Positive Control:
   - Uses the EXACT BackPatchHook.patch_and_forward path.
   - Cross-image donor injection: injects Image B's real L_src hidden states into Image A at dst in {6, 10, 12, 14}.
   - Self-patching: injects Image A's own L_src=22 states into dst=10 (testing the old TooLate null).
   - Measures max logit change, refusal probability mass shift, and generated completions.
2. Layer-to-Layer Geometry Measurement:
   - Measures pairwise cosine similarity cos(z_l, z_l') and relative L2 distance ||z_l - z_l'|| / ||z_l'||.
   - Trims top 2% norm channels across hidden dimension to eliminate attention sink / massive activation artifacts.
   - Decisively checks whether cos(z_10, z_22) is actually ~0.99 (stagnation) or substantially lower.
3. Dataset Image & Generation Inspection:
   - Checks manifest, image dimensions, and clean baseline generations with calibrated refusal metric.
"""

import argparse
import csv
import logging
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.patching_engine import BackPatchHook, PrefillRunner, extract_hidden_states, get_image_token_indices
from src.utils import compute_logit_mass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Refusal prefixes for calibrated probability mass
REFUSAL_PREFIX_STRINGS = [
    "I cannot", "I can't", "I'm sorry", "Sorry", "I am unable", "I am not able",
    "It is not appropriate", "I must decline", "As an AI", "I cannot fulfill"
]


def compute_refusal_log_odds(logits: torch.Tensor, tokenizer) -> tuple[float, float]:
    """
    Computes logit mass of refusal-start tokens:
      P(refusal) = sum_{t in RefusalTokens} softmax(logits)[t]
      LogOdds = log(P(refusal)) - log(1 - P(refusal))
    """
    probs = F.softmax(logits, dim=-1)
    
    # Identify refusal token IDs (single first tokens of refusal phrases)
    refusal_ids = set()
    for s in ["I cannot", "I can't", "Sorry", "I'm sorry", "I am unable", "As an AI"]:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            refusal_ids.add(toks[0])
    
    refusal_ids = list(refusal_ids)
    p_refusal = float(probs[refusal_ids].sum().item())
    p_refusal = max(min(p_refusal, 1.0 - 1e-7), 1e-7)
    log_odds = float(np.log(p_refusal) - np.log(1.0 - p_refusal))
    return p_refusal, log_odds


def run_geometry_measurement(model, processor, device):
    """
    Measure actual pairwise cosine similarity and relative L2 distance
    across layers on image tokens, with top 2% outlier channels trimmed.
    """
    logger.info("=" * 65)
    logger.info("CHECK 1: EMPIRICAL REPRESENTATION GEOMETRY (cos & relative L2)")
    logger.info("=" * 65)

    test_img = Image.new("RGB", (448, 448), color=(140, 100, 180))
    runner = PrefillRunner(model, processor, device)
    inputs = runner._prepare_inputs(test_img, Config.PROMPT)
    input_ids = inputs["input_ids"]

    img_token_id = getattr(model.config, "image_token_id", 151655)
    img_indices = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0].tolist()

    layers_to_measure = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    logger.info("Extracting hidden states for layers: %s", layers_to_measure)
    
    captured = extract_hidden_states(model, inputs, layers_to_measure)

    # Image token representations: shape (len(img_indices), d)
    z_per_layer = {}
    for l in layers_to_measure:
        # Mean-pool across image tokens
        z_l = captured[l][0, img_indices, :].float()  # (n_tokens, d)
        z_per_layer[l] = z_l.mean(dim=0)  # (d,)

    d = z_per_layer[layers_to_measure[0]].shape[0]
    n_trim = int(0.02 * d)  # top 2% outlier channels
    
    # Identify top 2% outlier channels across all measured layers by mean abs activation
    stacked = torch.stack(list(z_per_layer.values()), dim=0)  # (n_layers, d)
    mean_abs_per_channel = stacked.abs().mean(dim=0)
    _, top_indices = torch.topk(mean_abs_per_channel, k=n_trim)
    keep_mask = torch.ones(d, dtype=torch.bool, device=stacked.device)
    keep_mask[top_indices] = False
    logger.info("Trimmed top 2%% outlier channels (%d / %d channels).", n_trim, d)

    # Compute trimmed vectors
    z_trimmed = {l: z_per_layer[l][keep_mask] for l in layers_to_measure}

    # Pairwise Cosine & Relative L2
    logger.info("\n--- Pairwise Cosine Similarity (Trimmed) ---")
    header = "Layer | " + " ".join([f"{l:5d}" for l in layers_to_measure])
    logger.info(header)
    logger.info("-" * len(header))
    for l1 in layers_to_measure:
        row = [f"{l1:5d} |"]
        for l2 in layers_to_measure:
            cos = F.cosine_similarity(z_trimmed[l1].unsqueeze(0), z_trimmed[l2].unsqueeze(0)).item()
            row.append(f"{cos:5.2f}")
        logger.info(" ".join(row))

    logger.info("\n--- Pairwise Relative L2 Distance ||z_l1 - z_l2|| / ||z_l2|| ---")
    logger.info(header)
    logger.info("-" * len(header))
    for l1 in layers_to_measure:
        row = [f"{l1:5d} |"]
        for l2 in layers_to_measure:
            rel_l2 = (torch.norm(z_trimmed[l1] - z_trimmed[l2]) / torch.norm(z_trimmed[l2])).item()
            row.append(f"{rel_l2:5.2f}")
        logger.info(" ".join(row))

    # Explicit check on TooLate null pair: (src=22, dst=10)
    cos_10_22 = F.cosine_similarity(z_trimmed[10].unsqueeze(0), z_trimmed[22].unsqueeze(0)).item()
    rel_l2_10_22 = (torch.norm(z_trimmed[22] - z_trimmed[10]) / torch.norm(z_trimmed[10])).item()
    logger.info("\nCRITICAL CHECK: Layer 10 vs Layer 22")
    logger.info("  cos(z_10, z_22) = %.4f", cos_10_22)
    logger.info("  relative L2 ||z_22 - z_10|| / ||z_10|| = %.4f", rel_l2_10_22)
    if cos_10_22 > 0.95:
        logger.info("  -> Stagnation hypothesis supported (cos > 0.95).")
    else:
        logger.info("  -> Stagnation REJECTED (cos = %.4f << 0.99). TooLate null was NOT representation stagnation!", cos_10_22)


def run_real_path_patch_controls(model, processor, device):
    """
    Test the EXACT BackPatchHook.patch_and_forward path with:
    1. Cross-image donor injection: Donor image B into Recipient image A.
    2. Self-patching: Recipient image A's own L_src=22 injected into dst=10.
    """
    logger.info("=" * 65)
    logger.info("CHECK 2: REAL-PATH PATCH POSITIVE CONTROL")
    logger.info("=" * 65)

    img_A = Image.new("RGB", (448, 448), color=(30, 80, 200))   # Solid blue
    img_B = Image.new("RGB", (448, 448), color=(220, 40, 40))   # Solid red
    prompt = Config.PROMPT

    runner = PrefillRunner(model, processor, device)
    tok = processor.tokenizer
    ref_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.REFUSAL_TOKENS]
    comp_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.COMPLIANCE_TOKENS]

    # Baseline on Image A
    inputs_A = runner._prepare_inputs(img_A, prompt)
    inputs_B = runner._prepare_inputs(img_B, prompt)
    img_indices = runner._get_image_indices(inputs_A["input_ids"])

    with torch.no_grad():
        out_A = model(**inputs_A)
        clean_logits_A = out_A.logits[0, -1, :].float()
        clean_M_A = compute_logit_mass(clean_logits_A, ref_ids, comp_ids)
        clean_p_ref, clean_log_odds = compute_refusal_log_odds(clean_logits_A, tok)
        gen_clean_ids = model.generate(**inputs_A, max_new_tokens=24)
        gen_clean_text = processor.decode(gen_clean_ids[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True)

    logger.info("Recipient Image A Clean Baseline:")
    logger.info("  Legacy M = %.4f | Calibrated LogOdds = %.4f (P(ref) = %.4e)", clean_M_A, clean_log_odds, clean_p_ref)
    logger.info("  Clean Generation: %r", gen_clean_text.strip())

    # Capture Donor Image B's representations at src = 16, 20, 22
    donor_layers = [16, 20, 22]
    donor_states = extract_hidden_states(model, inputs_B, donor_layers)

    # 1. Cross-Image Donor Injection at dst in {6, 10, 12, 14}
    logger.info("\n--- Test 2A: Cross-Image Patching (Donor B -> Recipient A) ---")
    for dst in [6, 10, 12, 14]:
        for src in [16, 20]:
            patcher = BackPatchHook(model, src_layer=src, dst_layer=dst, image_token_indices=img_indices)
            b_act = donor_states[src][:, img_indices, :].clone()
            
            with torch.no_grad():
                patched_out = patcher.patch_and_forward(inputs_A, b_act)
                patched_logits = patched_out.logits[0, -1, :].float()
                
            max_logit_diff = (patched_logits - clean_logits_A).abs().max().item()
            mean_logit_diff = (patched_logits - clean_logits_A).abs().mean().item()
            patched_M = compute_logit_mass(patched_logits, ref_ids, comp_ids)
            p_ref, log_odds = compute_refusal_log_odds(patched_logits, tok)

            # Generate under patched model
            patched_gen = runner.generate_patched_completion(img_A, prompt, src_layer=src, dst_layer=dst, max_new_tokens=24)

            logger.info("Cross-Patch src=%2d -> dst=%2d: max|Δlogit|=%.4f, mean|Δlogit|=%.4f, ΔM=%.4f | Gen: %r",
                        src, dst, max_logit_diff, mean_logit_diff, abs(patched_M - clean_M_A), patched_gen.strip()[:60])

    # 2. Self-Patching: Recipient Image A's own src=22 injected into dst=10
    logger.info("\n--- Test 2B: Self-Patching (Image A src=22 -> dst=10: TooLate Null Check) ---")
    states_A = extract_hidden_states(model, inputs_A, [22])
    a_act_22 = states_A[22][:, img_indices, :].clone()
    
    patcher_self = BackPatchHook(model, src_layer=22, dst_layer=10, image_token_indices=img_indices)
    with torch.no_grad():
        self_patched_out = patcher_self.patch_and_forward(inputs_A, a_act_22)
        self_patched_logits = self_patched_out.logits[0, -1, :].float()

    max_logit_diff_self = (self_patched_logits - clean_logits_A).abs().max().item()
    mean_logit_diff_self = (self_patched_logits - clean_logits_A).abs().mean().item()
    self_M = compute_logit_mass(self_patched_logits, ref_ids, comp_ids)
    self_gen = runner.generate_patched_completion(img_A, prompt, src_layer=22, dst_layer=10, max_new_tokens=24)

    logger.info("Self-Patch src=22 -> dst=10:")
    logger.info("  max|Δlogit| = %.4f, mean|Δlogit| = %.4f, ΔM = %.4f",
                max_logit_diff_self, mean_logit_diff_self, abs(self_M - clean_M_A))
    logger.info("  Self-Patched Generation: %r", self_gen.strip()[:80])
    if max_logit_diff_self > 0.1:
        logger.info("  -> Real forward pass DOES change under src=22 -> dst=10! (max |Δlogit| = %.4f)", max_logit_diff_self)
    else:
        logger.warning("  -> Zero shift observed! Hook or caching bug detected.")


def check_existing_images_and_samples(data_dir: Path, model, processor, device):
    """
    Check dataset directory and inspect real dataset samples with calibrated metric.
    """
    logger.info("=" * 65)
    logger.info("CHECK 3: DATASET IMAGE AUDIT & CALIBRATED METRIC CHECK")
    logger.info("=" * 65)

    images_dir = data_dir / "images"
    manifest_path = data_dir / "manifest.csv"

    if not images_dir.exists() or not manifest_path.exists():
        logger.warning("Data directory or manifest not found at %s", data_dir)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    logger.info("Found %d total manifest entries and %d image files.", len(reader), len(list(images_dir.glob("*.png"))))

    runner = PrefillRunner(model, processor, device)
    tok = processor.tokenizer
    ref_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.REFUSAL_TOKENS]
    comp_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.COMPLIANCE_TOKENS]

    # Inspect first 4 HS and 4 SS samples
    hs_samples = [r for r in reader if r.get("cohort", "").lower() == "hs"][:4]
    ss_samples = [r for r in reader if r.get("cohort", "").lower() == "ss"][:4]

    logger.info("\n--- Clean Baseline Evaluation on Current Samples ---")
    for s in hs_samples + ss_samples:
        img_p = data_dir / s["path"]
        if not img_p.exists():
            continue
        im = Image.open(img_p).convert("RGB")
        inp = runner._prepare_inputs(im, Config.PROMPT)
        with torch.no_grad():
            out = model(**inp)
            logits = out.logits[0, -1, :].float()
            legacy_M = compute_logit_mass(logits, ref_ids, comp_ids)
            p_ref, log_odds = compute_refusal_log_odds(logits, tok)
            gen_ids = model.generate(**inp, max_new_tokens=32)
            gen_text = processor.decode(gen_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

        logger.info("[%2s] %-12s | Legacy M=%6.2f | LogOdds=%6.2f (P(ref)=%.2e) | Gen: %r",
                    s["cohort"], s["id"], legacy_M, log_odds, p_ref, gen_text.strip()[:65])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info("Loading model: %s on %s...", Config.MODEL_ID, args.device)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    # 1. Representation Geometry Check (Measured cos and relative L2)
    run_geometry_measurement(model, processor, device)

    # 2. Real-path patch positive controls
    run_real_path_patch_controls(model, processor, device)

    # 3. Dataset inspection and calibrated baseline check
    check_existing_images_and_samples(args.data_dir, model, processor, device)


if __name__ == "__main__":
    main()
