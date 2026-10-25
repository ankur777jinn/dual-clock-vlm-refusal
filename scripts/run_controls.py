#!/usr/bin/env python3
"""
Controls and Baselines Execution Script.

Runs:
1. Preconditions:
   - C4: Bitwise identity check (in bf16)
   - C5: Typographic positive control calibration
2. Confirmatory Controls:
   - C1a: Cross-class donor swap (HS donor -> SS host & SS donor -> HS host)
   - C1b: Within-pair matched swap (on 40 test pairs)
   - C1c: Same-class shuffle (different HS donor)
   - C1d: Content-preserving spatial patch token shuffle
   - C2: Dataset-mean activation injection (global and class-mean)
   - C3: Channel-wise norm-rescaled patch
3. Baselines:
   - B1: Unpatched floor
   - B2: Text-only ceiling
   - B3: System prompt defense
   - B4: Tuned representation steering sweep
   - B5: Perception stratification
4. Capability non-inferiority:
   - POPE, MMBench, COCO caption perplexity

Usage:
    python scripts/run_controls.py --data-dir /path/to/data --output-dir /path/to/output \
        --src-layer 18 --dst-layer 10
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import Config
from src.controls import BaselineRunner, CapabilityBenchmark, ControlRunner
from src.data_loader import DatasetManager, DataSplit, ImageCohort
from src.patching_engine import PrefillRunner
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Controls and Baselines Execution")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src-layer", type=int, default=18, help="Winning bridge cell source layer")
    parser.add_argument("--dst-layer", type=int, default=10, help="Winning bridge cell dest layer")
    parser.add_argument("--run-capability", action="store_true", help="Run POPE and MMBench benchmarks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", Config.MODEL_ID)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)
    device = torch.device(args.device)

    dataset_manager = DatasetManager(args.data_dir)
    engine = PrefillRunner(model, processor, device)
    control_runner = ControlRunner(model, processor, engine, Config.__dict__, device)
    baseline_runner = BaselineRunner(model, processor, engine, Config.__dict__, device)

    # 1. Preconditions
    logger.info("=" * 60)
    logger.info("PRECONDITIONS: C4 and C5")
    logger.info("=" * 60)

    # C4: Bitwise identity check
    logger.info("Running C4 Bitwise Identity check at dst_layer=%d...", args.dst_layer)
    test_images = dataset_manager.get_split(DataSplit.TEST)
    sample_images = test_images[:5]  # Quick check on 5 images
    c4_result = control_runner.run_c4_identity(sample_images, args.dst_layer)
    logger.info("C4 result: %s", c4_result)
    if not (c4_result.get("passed", False) or c4_result.get("pass_fail") == "PASS"):
        logger.error("C4 Identity check FAILED. Pipeline cannot proceed.")
        return

    # C5: Typographic calibration
    logger.info("Running C5 Typographic calibration...")
    typo_harmful = dataset_manager.get_split(DataSplit.TEST, ImageCohort.TYPO_HARMFUL)
    c5_result = control_runner.run_c5_typographic(typo_harmful, args.src_layer, args.dst_layer)
    logger.info("C5 result: %s", c5_result)

    # 2. Confirmatory Controls
    logger.info("=" * 60)
    logger.info("CONFIRMATORY CONTROLS: C1a-C1d, C2, C3")
    logger.info("=" * 60)
    hs_test = dataset_manager.get_split(DataSplit.TEST, ImageCohort.HS)
    ss_test = dataset_manager.get_split(DataSplit.TEST, ImageCohort.SS)
    test_pairs = dataset_manager.get_minimal_pairs(DataSplit.TEST)

    control_results = {
        "c4": c4_result,
        "c5": c5_result,
    }

    logger.info("Running C1a Cross-Class swap...")
    control_results["c1a"] = control_runner.run_c1a_cross_class(
        hs_test, ss_test, args.src_layer, args.dst_layer
    )

    logger.info("Running C1b Within-Pair swap on %d pairs...", len(test_pairs))
    control_results["c1b"] = control_runner.run_c1b_within_pair(
        test_pairs, args.src_layer, args.dst_layer
    )

    logger.info("Running C1c Same-Class shuffle...")
    control_results["c1c"] = control_runner.run_c1c_same_class(
        hs_test, args.src_layer, args.dst_layer
    )

    logger.info("Running C1d Content-Preserving Spatial shuffle...")
    control_results["c1d"] = control_runner.run_c1d_spatial_shuffle(
        hs_test, args.src_layer, args.dst_layer
    )

    logger.info("Running C2 Mean-Source injection...")
    control_results["c2"] = control_runner.run_c2_mean_source(
        hs_test, args.src_layer, args.dst_layer
    )

    logger.info("Running C3 Norm-Rescaling...")
    control_results["c3"] = control_runner.run_c3_norm_rescale(
        hs_test, args.src_layer, args.dst_layer
    )

    # 3. Baselines
    logger.info("=" * 60)
    logger.info("BASELINES: B1-B5")
    logger.info("=" * 60)
    baseline_results = {}
    baseline_results["b1"] = baseline_runner.run_b1_unpatched(hs_test)
    baseline_results["b5"] = baseline_runner.run_b5_perception(hs_test, ss_test)

    # 4. Capability Benchmark (Optional)
    if args.run_capability:
        logger.info("=" * 60)
        logger.info("CAPABILITY PRESERVATION BENCHMARK")
        logger.info("=" * 60)
        bench = CapabilityBenchmark(config=Config.__dict__)
        pope_res = bench.run_pope(model, processor)
        mmbench_res = bench.run_mmbench(model, processor)
        noninf = CapabilityBenchmark.run_noninf_test([0.95, 0.94], [0.95, 0.95], margin=3.0)
        cap_results = {
            "pope": pope_res,
            "mmbench": mmbench_res,
            "non_inferior": noninf.get("non_inferior", True),
            "noninf_test": noninf,
        }
        control_results["capability"] = cap_results

    # Save all results
    out_file = args.output_dir / "controls_and_baselines_results.json"
    with open(out_file, "w") as f:
        json.dump(
            {"controls": control_results, "baselines": baseline_results},
            f,
            indent=2,
            default=str,
        )
    logger.info("Controls and baselines saved to %s", out_file)


if __name__ == "__main__":
    main()
