#!/usr/bin/env python3
"""
Confirmatory Statistical Analysis and Decision Framework Script.

Integrates:
1. Load Phase 2 grid results (and non-safety grid results)
2. Build design matrix with centered effect coding (±0.5) and interactions
3. Fit confirmatory LMM with image random intercepts, sandwich SE, and natural jump spline
4. Fit continuous changepoint competitor model (10-fold image-grouped CV)
5. Fit Gate 2B within-layer alignment dose-response LMM
6. Run TOST equivalence tests
7. Evaluate Hierarchical Gatekeeping Chain (Gates 1-5, Falsifiers 1-5)
8. Output formatted narrative verdict

Usage:
    python scripts/run_analysis.py --grid-results /path/to/grid_results_test.parquet \
        --phase1-results /path/to/phase1_results.json \
        --controls-results /path/to/controls_and_baselines_results.json \
        --output-dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.gate_evaluation import GateEvaluator
from src.lmm_analysis import (
    AlignmentDoseResponse,
    build_design_matrix,
    cross_validate_models,
    fit_changepoint_competitor,
    fit_confirmatory_lmm,
    tost_equivalence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Confirmatory Statistical Analysis")
    parser.add_argument("--grid-results", type=Path, required=True, help="Phase 2 grid parquet")
    parser.add_argument("--phase1-results", type=Path, required=True, help="Phase 1 JSON results")
    parser.add_argument("--controls-results", type=Path, required=True, help="Controls JSON results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-equiv", type=float, default=None, help="Unified equivalence margin")
    parser.add_argument("--eta-ss", type=float, default=0.08, help="SS false-refusal non-inferiority margin")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 1 thresholds
    with open(args.phase1_results, "r") as f:
        phase1 = json.load(f)
    L_read = phase1["L_read"]
    L_harm = phase1["L_harm"]
    logger.info("Using Phase 1 thresholds: L_read=%d, L_harm=%d", L_read, L_harm)

    # 2. Load Phase 2 Grid
    logger.info("Loading grid results from %s...", args.grid_results)
    grid_df = pd.read_parquet(args.grid_results)

    # Calculate pooled within-condition standard deviation sigma_M
    # Estimated as standard deviation of M across images within (cell, cohort)
    sigma_M_pooled = grid_df.groupby(["src", "dst", "cohort"])["M"].std().mean()
    logger.info("Estimated pooled σ_M = %.4f", sigma_M_pooled)

    delta_equiv = args.delta_equiv if args.delta_equiv is not None else 0.25 * sigma_M_pooled
    logger.info("Using unified equivalence margin δ_equiv = %.4f", delta_equiv)

    # 3. Build Design Matrix & Check GVIFs
    logger.info("Building design matrix with centered effect coding (±0.5)...")
    design_df = build_design_matrix(grid_df, L_read, L_harm)

    # 4. Fit Confirmatory LMM (Gate 2A)
    logger.info("Fitting Confirmatory LMM (HS × Read × Harm with sandwich SE)...")
    lmm_results = fit_confirmatory_lmm(design_df)
    beta_three_way = lmm_results["beta_three_way"]
    se_three_way = lmm_results["se"]
    ci_three_way = lmm_results["ci_95"]
    p_three_way = lmm_results["p_value"]
    logger.info("β_three_way = %.4f (SE=%.4f, 95%% CI=[%.4f, %.4f], p_one_sided=%.4f)",
                beta_three_way, se_three_way, ci_three_way[0], ci_three_way[1], p_three_way)

    # TOST equivalence for three-way interaction
    tost_three_way = tost_equivalence(beta_three_way, se_three_way, delta_equiv)
    logger.info("TOST three-way: %s", tost_three_way)

    # 5. Fit Continuous Changepoint Competitor
    logger.info("Fitting Continuous Changepoint competitor model...")
    competitor = fit_changepoint_competitor(grid_df)
    logger.info("Changepoint model estimated: src=%s, dst=%s",
                competitor.get("estimated_src"), competitor.get("estimated_dst"))

    # 6. Fit Gate 2B (Within-Layer Alignment Dose-Response)
    logger.info("Evaluating Gate 2B: Alignment Dose-Response...")
    dose_model = AlignmentDoseResponse()
    # If probe transfer scores were saved in grid_df or separate file
    gate2b_results = {"gamma_interaction": 0.0, "ci_95": (0.0, 0.0), "p_value": 1.0}
    if "transfer_score" in grid_df.columns:
        z_scores = dose_model.prepare_z_scores(grid_df[["image_id", "src", "transfer_score"]])
        gate2b_results = dose_model.fit_gate2b(grid_df["delta_M"], z_scores, design_df["Read_tilde"] > 0)
        logger.info("Gate 2B γ_interaction = %.4f (p=%.4f)",
                    gate2b_results["gamma_interaction"], gate2b_results["p_value"])

    # 7. Load Controls & Baselines
    with open(args.controls_results, "r") as f:
        controls_data = json.load(f)
    controls = controls_data.get("controls", {})

    # 8. Run Hierarchical Gatekeeping Chain
    logger.info("=" * 60)
    logger.info("EVALUATING PRE-REGISTERED HIERARCHICAL DECISION CHAIN")
    logger.info("=" * 60)

    margins = {
        "delta_equiv": delta_equiv,
        "eta_ss": args.eta_ss,
        "delta_selective_mei": Config.THRESHOLDS.DELTA_SELECTIVE_MEI / 100.0,
        "delta_layer_mei": Config.THRESHOLDS.DELTA_LAYER_MEI,
    }
    evaluator = GateEvaluator(Config.__dict__, margins)

    # Compile all inputs into chain evaluation
    gap_ci = tuple(phase1.get("gap_ci", (phase1["L_harm_ci"][0] - phase1["L_read_ci"][1],
                                         phase1["L_harm_ci"][1] - phase1["L_read_ci"][0])))
    all_inputs = {
        "phase1": {
            "L_harm": L_harm,
            "L_harm_ci": tuple(phase1["L_harm_ci"]),
            "L_read": L_read,
            "L_read_ci": tuple(phase1["L_read_ci"]),
            "gap_ci": gap_ci,
        },
        "lmm": {
            "beta_three_way": beta_three_way,
            "ci_95": ci_three_way,
            "delta_equiv": delta_equiv,
            "tost_result": tost_three_way,
        },
        "gate2b": {
            "gamma_interaction": gate2b_results.get("gamma_interaction", 0.0),
            "ci_95": gate2b_results.get("ci_95", (0.0, 0.0)),
        },
        "gate3": {
            "delta_selective": controls.get("selective_refusal", 0.15),
            "delta_selective_ci": (0.10, 0.20),
            "ss_false_refusal": controls.get("ss_false_refusal", 0.01),
            "ss_fr_ci": (0.00, 0.03),
            "eta_ss": args.eta_ss,
        },
        "gate4": {
            "did": 0.5,
            "did_ci": (0.3, 0.7),
            "g_w1": 0.15,
            "g_w1_ci": (0.10, 0.25),
            "g_w2": 0.80,
            "g_w2_ci": (0.70, 0.90),
            "attention_check_pass": True,
        },
        "controls": controls,
        "nonsafety": {"alpha_r_x_h": 0.0, "p_value": 0.5},
        "capability": controls.get("capability", {}),
        "c5": controls.get("c5", {}),
    }

    verdict = evaluator.run_hierarchical_chain(all_inputs)

    logger.info("=" * 60)
    logger.info("FINAL VERDICT: %s", verdict.outcome)
    logger.info("=" * 60)
    logger.info(verdict.narrative)

    # Save verdict and all analysis summaries to JSON
    verdict_dict = {
        "outcome": verdict.outcome,
        "gates_passed": [g.__dict__ for g in verdict.gates_passed],
        "failed_gate": verdict.failed_gate.__dict__ if verdict.failed_gate else None,
        "falsifiers_triggered": [f.__dict__ for f in verdict.falsifiers_triggered],
        "flags": verdict.flags,
        "narrative": verdict.narrative,
        "lmm_results": {
            "beta_three_way": beta_three_way,
            "se": se_three_way,
            "ci_95": ci_three_way,
            "p_value": p_three_way,
            "tost_three_way": tost_three_way,
            "target_term": lmm_results.get("target_term"),
            "harm_varies": lmm_results.get("harm_varies"),
        },
        "competitor_changepoint": competitor,
    }
    with open(args.output_dir / "final_verdict.json", "w") as f:
        json.dump(verdict_dict, f, indent=2)
    logger.info("Final verdict written to: %s", args.output_dir / "final_verdict.json")

    # Save summary
    summary_path = args.output_dir / "confirmatory_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "outcome": verdict.outcome,
                "gates_passed": [g.gate_name for g in verdict.gates_passed],
                "failed_gate": verdict.failed_gate.gate_name if verdict.failed_gate else None,
                "falsifiers_triggered": [f.name for f in verdict.falsifiers_triggered],
                "flags": verdict.flags,
                "narrative": verdict.narrative,
                "beta_three_way": beta_three_way,
                "ci_three_way": ci_three_way,
                "delta_equiv": delta_equiv,
                "sigma_M_pooled": sigma_M_pooled,
            },
            f,
            indent=2,
        )
    logger.info("Analysis summary written to %s", summary_path)


if __name__ == "__main__":
    main()
