#!/usr/bin/env python3
"""
Phase 1 Diagnostic Execution Script.

Runs on the pooled diagnostic pool (Tuning + Validation, n=200):
1. Diagnostic 1A: L_read via cumulative attention knockout
2. Diagnostic 1B: L_harm via cross-modal probe transfer
3. Typographic calibration
4. Gate 1 evaluation

Usage:
    python scripts/run_phase1.py --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import logging
import json
from pathlib import Path

import torch

from src.config import Config, build_grid
from src.data_loader import DatasetManager, DataSplit, TextPartitioner
from src.phase1_diagnostics import ReadingWindowDiagnostic, HarmEmergenceDiagnostic, run_phase1
from src.utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Diagnostic Clocks")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--advbench-path", type=Path, required=True, help="AdvBench strings file")
    parser.add_argument("--alpaca-path", type=Path, required=True, help="Alpaca strings file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", Config.MODEL_ID)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    logger.info("Loading dataset from %s", args.data_dir)
    dataset_manager = DatasetManager(args.data_dir)
    text_partitioner = TextPartitioner(args.advbench_path, args.alpaca_path)

    logger.info("Running Phase 1 diagnostics...")
    results = run_phase1(
        model=model,
        processor=processor,
        dataset_manager=dataset_manager,
        text_partitioner=text_partitioner,
        device=torch.device(args.device),
        output_dir=args.output_dir,
    )

    # Save results
    results_path = args.output_dir / "phase1_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if not k.startswith("_")},
            f,
            indent=2,
            default=str,
        )
    logger.info("Phase 1 results saved to %s", results_path)

    # Report
    logger.info("=" * 60)
    logger.info("PHASE 1 RESULTS")
    logger.info("=" * 60)
    logger.info("L_read = %d (95%% CI: %s)", results["L_read"], results["L_read_ci"])
    logger.info("L_harm = %d (95%% CI: %s)", results["L_harm"], results["L_harm_ci"])
    logger.info("Gap = %d layers (95%% CI: %s)", results["gap"], results["gap_ci"])
    logger.info("Gate 1 pass: %s", results["gate1_pass"])

    if results["gate1_pass"]:
        logger.info("✓ Gate 1 PASSED — proceeding to Phase 2 is warranted.")
    else:
        logger.warning("✗ Gate 1 FAILED — review Phase 1 diagnostics before proceeding.")


if __name__ == "__main__":
    main()
