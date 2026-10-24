#!/usr/bin/env python3
"""
Power simulation script for preregistration.

Estimates statistical power for each gate using pilot variance estimates
from the Tuning/Validation splits.

Usage:
    python scripts/power_simulation.py --sigma-M-hs 1.2 --sigma-M-ss 0.8 \
        --sd-pair 1.0 --sd-contrast 0.9 --output-dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path

from src.lmm_analysis import PowerSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Power Simulation for Preregistration")
    parser.add_argument("--sigma-M-hs", type=float, required=True, help="Within-image SD of M for HS")
    parser.add_argument("--sigma-M-ss", type=float, required=True, help="Within-image SD of M for SS")
    parser.add_argument("--sd-pair", type=float, required=True, help="SD of paired contrasts (C1b)")
    parser.add_argument("--sd-contrast", type=float, required=True, help="SD of within-image Read×Harm contrast")
    parser.add_argument("--delta-equiv", type=float, default=None, help="Equivalence margin (in sigma_M units)")
    parser.add_argument("--n-sim", type=int, default=5000, help="Number of simulations per test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sigma_M_pooled = ((args.sigma_M_hs**2 + args.sigma_M_ss**2) / 2) ** 0.5
    if args.delta_equiv is None:
        delta_equiv = 0.25 * sigma_M_pooled
    else:
        delta_equiv = args.delta_equiv

    logger.info("Pooled σ_M = %.3f", sigma_M_pooled)
    logger.info("δ_equiv = %.3f (%.2f σ_M)", delta_equiv, delta_equiv / sigma_M_pooled)

    sim = PowerSimulator(
        sigma_M_hs=args.sigma_M_hs,
        sigma_M_ss=args.sigma_M_ss,
        sigma_M_pooled=sigma_M_pooled,
        sd_pair=args.sd_pair,
        sd_contrast=args.sd_contrast,
    )

    results = {
        "sigma_M_hs": args.sigma_M_hs,
        "sigma_M_ss": args.sigma_M_ss,
        "sigma_M_pooled": sigma_M_pooled,
        "delta_equiv": delta_equiv,
        "sd_pair": args.sd_pair,
        "sd_contrast": args.sd_contrast,
        "n_sim": args.n_sim,
    }

    # Gate 2A: Three-way interaction
    power_2a = sim.simulate_three_way_power(
        n_hs=100, n_ss=100, n_cells=55, delta_equiv=delta_equiv, n_sim=args.n_sim
    )
    results["gate2a_power"] = power_2a
    logger.info("Gate 2A power (three-way): %.1f%%", power_2a * 100)

    # C1b: Within-pair swap
    power_c1b = sim.simulate_c1b_power(
        n_pairs=40, delta_equiv=delta_equiv, n_sim=args.n_sim
    )
    results["c1b_power"] = power_c1b
    logger.info("C1b power (paired swap): %.1f%%", power_c1b * 100)

    # Gate 3: Judged selective refusal
    for true_effect in [10, 15, 20]:
        power_g3 = sim.simulate_gate3_power(
            n_hs=100, n_ss=100, true_effect_pp=true_effect, n_sim=args.n_sim
        )
        results[f"gate3_power_at_{true_effect}pp"] = power_g3
        logger.info("Gate 3 power (true effect %d pp): %.1f%%", true_effect, power_g3 * 100)

    # C1b designation
    if power_c1b < 0.60:
        results["c1b_designation"] = "descriptive (power < 60%)"
        logger.warning("C1b power < 60%% — designating as descriptive, not confirmatory.")
    else:
        results["c1b_designation"] = "confirmatory"

    # Report
    report = sim.generate_report()
    results["narrative"] = report
    logger.info("\n%s", report)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "power_simulation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Power simulation saved to %s", out_path)


if __name__ == "__main__":
    main()
