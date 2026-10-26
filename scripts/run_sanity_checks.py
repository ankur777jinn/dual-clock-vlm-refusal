#!/usr/bin/env python3
"""
Phase 0 Sanity Checks:
1. Hook Positive Control at dst in {6, 10, 12, 14}:
   Injects high-variance noise / perturbation at dst to confirm that BackPatchHook
   actually modifies activations and changes output logit mass M at deeper layers.
2. Image Inspection:
   Checks the dataset manifest and images to confirm whether test images are
   synthetic dummy images or natural photographs.
"""

import argparse
import logging
import sys
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.patching_engine import BackPatchHook, PrefillRunner
from src.utils import compute_logit_mass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_hook_positive_control(model, processor, device):
    logger.info("=" * 60)
    logger.info("CHECK 1: HOOK POSITIVE CONTROL (dst = 6, 10, 12, 14)")
    logger.info("=" * 60)

    # Create dummy image and input
    test_img = Image.new("RGB", (448, 448), color=(100, 150, 200))
    prompt = Config.PROMPT

    runner = PrefillRunner(model, processor, device)
    tok = processor.tokenizer
    ref_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.REFUSAL_TOKENS]
    comp_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.COMPLIANCE_TOKENS]

    # Clean forward baseline
    inputs = runner._prepare_inputs(test_img, prompt)
    input_ids = inputs["input_ids"]
    clean_M = runner.compute_logit_mass(test_img, prompt, ref_ids, comp_ids)
    
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=32)
        gen_text = processor.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    logger.info("Clean baseline M = %.4f | Generation: %r", clean_M, gen_text)

    dst_layers = [6, 10, 12, 14]
    results = {}

    img_token_id = getattr(model.config, "image_token_id", 151655)
    img_indices = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0].tolist()
    hidden_dim = getattr(
        model.config,
        "hidden_size",
        getattr(getattr(model.config, "text_config", None), "hidden_size", 3584),
    )
    logger.info("Image tokens count: %d, Hidden dim: %d", len(img_indices), hidden_dim)

    for dst in dst_layers:
        # Generate large random perturbation (positive control)
        noise_activations = torch.randn(1, len(img_indices), hidden_dim, dtype=torch.bfloat16, device=device) * 5.0

        # Inject noise at dst using exact BackPatchHook implementation
        hook = BackPatchHook(model, src_layer=dst + 2, dst_layer=dst, image_token_indices=img_indices)
        
        with torch.no_grad():
            out = hook.patch_and_forward(inputs, noise_activations)
            logits = out.logits[0, -1, :].float()
            noisy_M = compute_logit_mass(logits, ref_ids, comp_ids)

        delta_M = abs(noisy_M - clean_M)
        logger.info("dst = %2d: clean M = %.4f, noisy M = %.4f, |ΔM| = %.4f", dst, clean_M, noisy_M, delta_M)
        results[dst] = delta_M

    logger.info("-" * 60)
    for dst, dM in results.items():
        status = "PASSED (Hook is active)" if dM > 0.05 else "FAILED (HOOK IS DEAD/NO-OP!)"
        logger.info("Layer %2d: |ΔM| = %.4f -> %s", dst, dM, status)
    logger.info("-" * 60)


def check_existing_images(data_dir: Path):
    logger.info("=" * 60)
    logger.info("CHECK 2: DATASET IMAGE INSPECTION")
    logger.info("=" * 60)

    images_dir = data_dir / "images"
    if not images_dir.exists():
        logger.warning("Images dir %s does not exist!", images_dir)
        return

    sample_images = list(images_dir.glob("*.png"))[:5]
    logger.info("Found %d total images. Inspecting first %d:", len(list(images_dir.glob("*.png"))), len(sample_images))
    for p in sample_images:
        im = Image.open(p)
        logger.info("  %s: size=%s, mode=%s", p.name, im.size, im.mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    check_existing_images(args.data_dir)

    logger.info("Loading model: %s", Config.MODEL_ID)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    run_hook_positive_control(model, processor, torch.device(args.device))

    # Real Dataset Samples Inspection
    logger.info("=" * 60)
    logger.info("CHECK 3: REAL DATASET GENERATION & BASELINE CHECK")
    logger.info("=" * 60)
    manifest_path = args.data_dir / "manifest.csv"
    if manifest_path.exists():
        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        logger.info("Manifest has %d total samples", len(reader))
        runner = PrefillRunner(model, processor, torch.device(args.device))
        tok = processor.tokenizer
        ref_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.REFUSAL_TOKENS]
        comp_ids = [tok.encode(t, add_special_tokens=False)[0] for t in Config.COMPLIANCE_TOKENS]
        
        hs_samples = [r for r in reader if r.get("cohort", "").lower() == "hs"][:3]
        ss_samples = [r for r in reader if r.get("cohort", "").lower() == "ss"][:2]
        
        for s in hs_samples + ss_samples:
            img_p = args.data_dir / s["path"]
            if img_p.exists():
                im = Image.open(img_p).convert("RGB")
                m_val = runner.compute_logit_mass(im, Config.PROMPT, ref_ids, comp_ids)
                inp = runner._prepare_inputs(im, Config.PROMPT)
                with torch.no_grad():
                    g_ids = model.generate(**inp, max_new_tokens=32)
                    g_text = processor.decode(g_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                logger.info("Sample %s (%s): M = %.4f | Gen: %r", s["id"], s["cohort"], m_val, g_text.strip())
    else:
        logger.warning("Manifest %s not found", manifest_path)


if __name__ == "__main__":
    main()
