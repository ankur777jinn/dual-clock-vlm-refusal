#!/usr/bin/env python3
"""
Phase 2 Grid Execution Script.

Runs the 55-cell factorial back-patching grid + unpatched reference on the
specified split (Validation for pilot sweeps, Test for confirmatory).

Usage:
    python scripts/run_phase2.py --data-dir /path/to/data --output-dir /path/to/output \
        --split validation --L-read 12 --L-harm 18
"""

import argparse
import logging
import json
from pathlib import Path

import pandas as pd
import torch

from src.config import Config, build_grid, count_quadrants
from src.data_loader import DatasetManager, DataSplit
from src.patching_engine import PrefillRunner
from src.phase2_grid import GridRunner, JudgeEvaluator
from src.utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Factorial Grid")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["validation", "test"], required=True)
    parser.add_argument("--L-read", type=int, required=True, help="L_read from Phase 1")
    parser.add_argument("--L-harm", type=int, required=True, help="L_harm from Phase 1")
    parser.add_argument("--run-nonsafety", action="store_true", help="Also run non-safety grid")
    parser.add_argument("--judge", action="store_true", help="Run judge evaluation on completions")
    parser.add_argument("--judge-model", type=str, default=None, help="Local HF judge model (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct or Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    quadrants = count_quadrants(args.L_read, args.L_harm, grid)
    logger.info("Grid: %d cells. Quadrants at L_read=%d, L_harm=%d:", len(grid), args.L_read, args.L_harm)
    for name, count in quadrants.items():
        logger.info("  %s: %d cells", name, count)

    split = DataSplit.VALIDATION if args.split == "validation" else DataSplit.TEST

    logger.info("Loading model: %s", Config.MODEL_ID)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    dataset_manager = DatasetManager(args.data_dir)
    prefill_runner = PrefillRunner(model, processor, torch.device(args.device))
    grid_runner = GridRunner(model, processor, prefill_runner, Config, torch.device(args.device))

    # Get images for the split
    images = dataset_manager.get_split(split)

    grid_parquet = args.output_dir / f"grid_results_{args.split}.parquet"
    backup_parquet = args.output_dir / "grid_results.parquet"

    if grid_parquet.exists() and grid_parquet.stat().st_size > 0:
        logger.info("Found existing grid results at %s (%d bytes). Loading to avoid rerun...",
                    grid_parquet, grid_parquet.stat().st_size)
        results_df = pd.read_parquet(grid_parquet)
        logger.info("Loaded grid results: %d rows", len(results_df))
    elif backup_parquet.exists() and backup_parquet.stat().st_size > 0:
        logger.info("Found existing full grid at %s (%d bytes). Copying to %s...",
                    backup_parquet, backup_parquet.stat().st_size, grid_parquet)
        results_df = pd.read_parquet(backup_parquet)
        results_df.to_parquet(grid_parquet)
        logger.info("Saved grid results: %d rows", len(results_df))
    else:
        logger.info("Running grid on %d images (%s split)...", len(images), args.split)
        results_df = grid_runner.run_full_grid(images, grid, args.output_dir)
        results_df.to_parquet(grid_parquet)
        logger.info("Grid results saved: %d rows", len(results_df))

    if args.run_nonsafety:
        nonsafety_parquet = args.output_dir / f"nonsafety_grid_{args.split}.parquet"
        backup_nonsafety = args.output_dir / "nonsafety_grid_results.parquet"

        if nonsafety_parquet.exists() and nonsafety_parquet.stat().st_size > 0:
            logger.info("Found existing non-safety grid at %s. Skipping rerun...", nonsafety_parquet)
            nonsafety_df = pd.read_parquet(nonsafety_parquet)
        elif backup_nonsafety.exists() and backup_nonsafety.stat().st_size > 0:
            logger.info("Found existing non-safety grid at %s. Copying to %s...", backup_nonsafety, nonsafety_parquet)
            nonsafety_df = pd.read_parquet(backup_nonsafety)
            nonsafety_df.to_parquet(nonsafety_parquet)
        else:
            from src.data_loader import ImageCohort
            control_images = (
                dataset_manager.get_split(split, ImageCohort.CONTROL_PLUMBING)
                + dataset_manager.get_split(split, ImageCohort.CONTROL_GARDENING)
            )
            if not control_images:
                # Fall back to test split control images if current split does not contain them
                control_images = (
                    dataset_manager.get_split(DataSplit.TEST, ImageCohort.CONTROL_PLUMBING)
                    + dataset_manager.get_split(DataSplit.TEST, ImageCohort.CONTROL_GARDENING)
                )
            if control_images:
                logger.info("Running non-safety grid on %d control images...", len(control_images))
                nonsafety_df = grid_runner.run_nonsafety_grid(control_images, grid, {}, args.output_dir)
                nonsafety_df.to_parquet(nonsafety_parquet)
            else:
                logger.warning("No non-safety control images found; skipping non-safety grid.")

    if args.judge:
        logger.info("Running judge evaluation on generated completions...")
        local_judge = None
        local_tok = None
        if args.judge_model:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading local judge model: %s", args.judge_model)
            local_tok = AutoTokenizer.from_pretrained(args.judge_model)
            local_judge = AutoModelForCausalLM.from_pretrained(
                args.judge_model, torch_dtype=torch.bfloat16, device_map=args.device
            )

        judge = JudgeEvaluator(
            primary_judge=Config.PRIMARY_JUDGE,
            backup_judge=Config.BACKUP_JUDGE,
            local_model=local_judge,
            local_tokenizer=local_tok,
        )

        labels = []
        for _, row in results_df.iterrows():
            prompt_text = Config.PROMPT
            gen_text = row.get("generated_text", "")
            lbl = judge.judge_single(prompt_text, gen_text)
            labels.append(lbl)

        results_df["judge_label"] = labels
        results_df.to_parquet(args.output_dir / f"grid_results_{args.split}.parquet")
        logger.info("Saved judge evaluations to grid results parquet: %s",
                    args.output_dir / f"grid_results_{args.split}.parquet")

    logger.info("Phase 2 grid execution complete.")


if __name__ == "__main__":
    main()
