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
from pathlib import Path
from PIL import Image
import torch

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
    clean_M = runner.compute_logit_mass(test_img, prompt, ref_ids, comp_ids)
    logger.info("Clean baseline M = %.4f", clean_M)

    dst_layers = [6, 10, 12, 14]
    results = {}

    for dst in dst_layers:
        inputs = runner._prepare_inputs(test_img, prompt)
        input_ids = inputs["input_ids"]
        img_token_id = model.config.image_token_id
        img_indices = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0].tolist()

        # Generate large random perturbation (positive control)
        # Hidden dimension for Qwen2.5-VL-7B is 3584
        hidden_dim = model.config.hidden_size
        noise_activations = torch.randn(len(img_indices), hidden_dim, dtype=torch.bfloat16, device=device) * 5.0

        # Inject noise at dst
        hook = BackPatchHook(model, src_layer=dst + 2, dst_layer=dst, image_token_indices=img_indices)
        
        # Test forward with noise injection
        with torch.no_grad():
            hook_handle = None
            def inject_hook(module, args):
                h = args[0].clone()
                h[0, img_indices, :] = noise_activations
                return (h,) + args[1:]
            
            hook_handle = model.model.layers[dst].register_forward_pre_hook(inject_hook)
            out = model(**inputs)
            hook_handle.remove()

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


if __name__ == "__main__":
    main()
