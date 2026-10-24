"""
Pre-registered decision framework: hierarchical gatekeeping chain.
Evaluates Gates 1-5, Falsifiers F1-F5, and construct/harness flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Result containers
# ===================================================================

@dataclass
class GateResult:
    """Outcome of evaluating a single gate in the decision chain."""

    gate_name: str
    passed: bool
    metric_value: float
    ci: tuple[float, float]
    threshold: float
    description: str


@dataclass
class FalsifierResult:
    """Outcome of evaluating a falsifier bound to a specific gate."""

    name: str
    bound_gate: str
    triggered: bool
    evidence: dict
    description: str


@dataclass
class Verdict:
    """Final verdict produced by the hierarchical gatekeeping chain.

    Attributes
    ----------
    outcome : str
        One of ``SUPPORTED``, ``FALSIFIED``, or ``INCONCLUSIVE``.
    gates_passed : list[GateResult]
        Gates that passed before the chain terminated.
    failed_gate : GateResult or None
        The first gate that failed (``None`` if all passed).
    falsifiers_triggered : list[FalsifierResult]
        Falsifiers that fired during evaluation.
    flags : list[str]
        Construct or harness warning flags.
    narrative : str
        Human-readable summary of the verdict.
    """

    outcome: str  # "SUPPORTED" | "FALSIFIED" | "INCONCLUSIVE"
    gates_passed: list[GateResult] = field(default_factory=list)
    failed_gate: Optional[GateResult] = None
    falsifiers_triggered: list[FalsifierResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    narrative: str = ""


# ===================================================================
# Gate evaluator
# ===================================================================

class GateEvaluator:
    """Implements the hierarchical gatekeeping decision chain.

    The chain runs gates 1 → 2A → 2B → 3 → 4 → 5 sequentially,
    stopping at the first failure.  For the failed gate the
    corresponding falsifier is evaluated.  Gates beyond the failure
    point are reported as *descriptive* (untested).
    """

    def __init__(
        self,
        config: dict,
        margins: dict,
    ) -> None:
        """
        Parameters
        ----------
        config : dict
            General experiment configuration.
        margins : dict
            Pre-registered decision thresholds.  Expected keys:

            * ``delta_equiv`` – equivalence margin for TOST (Gate 2A).
            * ``eta_ss`` – maximum tolerable SS false-refusal rate (Gate 3).
            * ``delta_selective_mei`` – minimum-effect-of-interest for
              selective refusal (Gate 3).
            * ``delta_layer_mei`` – minimum-effect-of-interest for the
              layer gap between L_harm and L_read (Falsifier 1).
        """
        self.config = config
        self.delta_equiv: float = margins["delta_equiv"]
        self.eta_ss: float = margins["eta_ss"]
        self.delta_selective_mei: float = margins["delta_selective_mei"]
        self.delta_layer_mei: float = margins["delta_layer_mei"]

    # =================================================================
    # Gates
    # =================================================================

    def evaluate_gate1(
        self,
        L_harm: int,
        L_harm_ci: tuple[float, float],
        L_read: int,
        L_read_ci: tuple[float, float],
        typo_results: dict,
    ) -> GateResult:
        """Gate 1: Two-clock existence.

        Passes if L_harm and L_read are identified with non-overlapping CIs
        **and** L_read < L_harm.  Typographic-image results serve as a
        calibration check.

        Parameters
        ----------
        L_harm : int
            Estimated harm-recognition layer.
        L_harm_ci : tuple[float, float]
            95 % CI for L_harm.
        L_read : int
            Estimated image-reading layer.
        L_read_ci : tuple[float, float]
            95 % CI for L_read.
        typo_results : dict
            Results from control C5 (typographic images).

        Returns
        -------
        GateResult
        """
        # L_read must precede L_harm and CIs must not overlap
        cis_separated = L_read_ci[1] < L_harm_ci[0]
        order_correct = L_read < L_harm

        passed = cis_separated and order_correct

        description = (
            f"L_read={L_read} CI={L_read_ci}, L_harm={L_harm} CI={L_harm_ci}. "
            f"CIs separated: {cis_separated}. Order correct: {order_correct}. "
            f"Typo baseline refusal: {typo_results.get('baseline_refusal_rate', 'N/A'):.2%}."
        )

        return GateResult(
            gate_name="Gate 1: Two-clock existence",
            passed=passed,
            metric_value=float(L_harm - L_read),
            ci=(L_harm_ci[0] - L_read_ci[1], L_harm_ci[1] - L_read_ci[0]),
            threshold=0.0,  # gap must be > 0
            description=description,
        )

    def evaluate_gate2a(
        self,
        beta_three_way: float,
        ci_95: tuple[float, float],
        delta_equiv: float,
    ) -> GateResult:
        """Gate 2A: Three-way interaction significance.

        Passes if the one-sided p-value for β_{HS×Read×Harm} > 0 is < 0.05
        (approximated here by CI_low > 0).

        Parameters
        ----------
        beta_three_way : float
            Point estimate of the three-way interaction.
        ci_95 : tuple[float, float]
            95 % confidence interval.
        delta_equiv : float
            Equivalence margin (used only for descriptive reporting).

        Returns
        -------
        GateResult
        """
        passed = ci_95[0] > 0

        return GateResult(
            gate_name="Gate 2A: Three-way interaction",
            passed=passed,
            metric_value=beta_three_way,
            ci=ci_95,
            threshold=0.0,
            description=(
                f"β_three_way = {beta_three_way:.4f}, "
                f"95% CI = ({ci_95[0]:.4f}, {ci_95[1]:.4f}). "
                f"Equivalence margin δ = {delta_equiv:.4f}."
            ),
        )

    def evaluate_gate2b(
        self,
        gamma_interaction: float,
        ci_95: tuple[float, float],
    ) -> GateResult:
        """Gate 2B: Alignment dose-response.

        Passes if γ_interaction > 0 (CI_low > 0).

        Parameters
        ----------
        gamma_interaction : float
            Estimated interaction coefficient from the dose-response LMM.
        ci_95 : tuple[float, float]
            95 % CI.

        Returns
        -------
        GateResult
        """
        passed = ci_95[0] > 0

        return GateResult(
            gate_name="Gate 2B: Alignment dose-response",
            passed=passed,
            metric_value=gamma_interaction,
            ci=ci_95,
            threshold=0.0,
            description=(
                f"γ_interaction = {gamma_interaction:.4f}, "
                f"95% CI = ({ci_95[0]:.4f}, {ci_95[1]:.4f})."
            ),
        )

    def evaluate_gate3(
        self,
        delta_selective: float,
        delta_selective_ci: tuple[float, float],
        ss_false_refusal: float,
        ss_fr_ci: tuple[float, float],
        eta_ss: float,
    ) -> GateResult:
        """Gate 3: Judged selective refusal.

        Passes if:
        1. Δ_selective > 0 (lower CI bound > 0), AND
        2. SS false-refusal rate upper CI bound < η_ss.

        Parameters
        ----------
        delta_selective : float
            Symmetric selective-refusal metric.
        delta_selective_ci : tuple[float, float]
            95 % CI for Δ_selective.
        ss_false_refusal : float
            SS false-refusal rate (patched).
        ss_fr_ci : tuple[float, float]
            95 % CI for the SS false-refusal rate.
        eta_ss : float
            Pre-registered maximum tolerable SS false-refusal rate.

        Returns
        -------
        GateResult
        """
        selective_ok = delta_selective_ci[0] > 0
        ss_ok = ss_fr_ci[1] < eta_ss

        passed = selective_ok and ss_ok

        return GateResult(
            gate_name="Gate 3: Selective refusal",
            passed=passed,
            metric_value=delta_selective,
            ci=delta_selective_ci,
            threshold=0.0,
            description=(
                f"Δ_selective = {delta_selective:.4f}, "
                f"CI = ({delta_selective_ci[0]:.4f}, {delta_selective_ci[1]:.4f}). "
                f"SS false-refusal = {ss_false_refusal:.2%}, "
                f"CI = ({ss_fr_ci[0]:.2%}, {ss_fr_ci[1]:.2%}), η_ss = {eta_ss:.2%}."
            ),
        )

    def evaluate_gate4(
        self,
        did: float,
        did_ci: tuple[float, float],
        g_w1: float,
        g_w1_ci: tuple[float, float],
        g_w2: float,
        g_w2_ci: tuple[float, float],
        attention_check_pass: bool,
    ) -> GateResult:
        """Gate 4: Human evaluation.

        Passes if the difference-in-differences (DiD) is significant, both
        within-condition Hedges' g values are positive, and the attention
        check passes.

        Parameters
        ----------
        did : float
            Difference-in-differences estimate.
        did_ci : tuple[float, float]
            95 % CI for DiD.
        g_w1 : float
            Hedges' g for within-condition comparison 1.
        g_w1_ci : tuple[float, float]
            95 % CI for g_w1.
        g_w2 : float
            Hedges' g for within-condition comparison 2.
        g_w2_ci : tuple[float, float]
            95 % CI for g_w2.
        attention_check_pass : bool
            Whether ≥ 90 % of participants passed attention checks.

        Returns
        -------
        GateResult
        """
        did_sig = did_ci[0] > 0
        g_ok = (g_w1_ci[0] > 0) and (g_w2_ci[0] > 0)

        passed = did_sig and g_ok and attention_check_pass

        return GateResult(
            gate_name="Gate 4: Human evaluation",
            passed=passed,
            metric_value=did,
            ci=did_ci,
            threshold=0.0,
            description=(
                f"DiD = {did:.4f}, CI = ({did_ci[0]:.4f}, {did_ci[1]:.4f}). "
                f"g_w1 = {g_w1:.3f} CI = ({g_w1_ci[0]:.3f}, {g_w1_ci[1]:.3f}). "
                f"g_w2 = {g_w2:.3f} CI = ({g_w2_ci[0]:.3f}, {g_w2_ci[1]:.3f}). "
                f"Attention check: {'PASS' if attention_check_pass else 'FAIL'}."
            ),
        )

    def evaluate_gate5(
        self,
        control_results: dict,
        nonsafety_result: dict,
        capability_result: dict,
    ) -> GateResult:
        """Gate 5: Robustness and specificity.

        Passes if:
        - C4 identity check passes.
        - Non-safety (OpenImages) task scores are not significantly
          degraded.
        - Capability benchmarks pass non-inferiority.

        Parameters
        ----------
        control_results : dict
            Aggregated results from C1a–C1d, C2, C3, C4, C5.
        nonsafety_result : dict
            Results from the non-safety grid.
        capability_result : dict
            Results from CapabilityBenchmark (POPE, MMBench, perplexity,
            non-inferiority test).

        Returns
        -------
        GateResult
        """
        c4_pass = control_results.get("c4", {}).get("pass_fail") == "PASS"
        noninf_pass = capability_result.get("non_inferior", False)

        passed = c4_pass and noninf_pass

        description_parts = [
            f"C4 identity: {'PASS' if c4_pass else 'FAIL'}",
            f"Non-inferiority: {'PASS' if noninf_pass else 'FAIL'}",
        ]

        return GateResult(
            gate_name="Gate 5: Robustness & specificity",
            passed=passed,
            metric_value=1.0 if passed else 0.0,
            ci=(0.0, 1.0),
            threshold=0.5,
            description=". ".join(description_parts) + ".",
        )

    # =================================================================
    # Falsifiers
    # =================================================================

    def evaluate_falsifier1(
        self,
        L_harm_L_read_gap_ci: tuple[float, float],
        delta_layer_mei: float,
    ) -> FalsifierResult:
        """Falsifier F1: Layer-gap insufficiency.

        Triggered if the CI for ``L_harm − L_read`` includes values below
        the minimum-effect-of-interest.

        Parameters
        ----------
        L_harm_L_read_gap_ci : tuple[float, float]
            95 % CI for the gap between L_harm and L_read.
        delta_layer_mei : float
            Pre-registered minimum layer-gap MEI.

        Returns
        -------
        FalsifierResult
        """
        triggered = L_harm_L_read_gap_ci[0] < delta_layer_mei

        return FalsifierResult(
            name="F1: Layer-gap insufficiency",
            bound_gate="Gate 1",
            triggered=triggered,
            evidence={
                "gap_ci": L_harm_L_read_gap_ci,
                "delta_layer_mei": delta_layer_mei,
            },
            description=(
                f"Gap CI = ({L_harm_L_read_gap_ci[0]:.1f}, "
                f"{L_harm_L_read_gap_ci[1]:.1f}), MEI = {delta_layer_mei:.1f}. "
                f"{'TRIGGERED' if triggered else 'Not triggered'}."
            ),
        )

    def evaluate_falsifier2(
        self,
        beta_three_way: float,
        tost_result: dict,
        gate2a_passed: bool,
    ) -> FalsifierResult:
        """Falsifier F2: Equivalence of three-way interaction.

        Triggered if Gate 2A failed AND the TOST test establishes
        equivalence within ±δ.  This would indicate that the three-way
        effect is *practically zero*, not merely under-powered.

        Parameters
        ----------
        beta_three_way : float
            Point estimate.
        tost_result : dict
            Output of ``tost_equivalence``.
        gate2a_passed : bool
            Whether Gate 2A passed.

        Returns
        -------
        FalsifierResult
        """
        triggered = (not gate2a_passed) and tost_result.get("equivalent", False)

        return FalsifierResult(
            name="F2: Three-way equivalence",
            bound_gate="Gate 2A",
            triggered=triggered,
            evidence={
                "beta_three_way": beta_three_way,
                "tost_equivalent": tost_result.get("equivalent"),
                "ci_90": tost_result.get("ci_90"),
            },
            description=(
                f"β = {beta_three_way:.4f}. "
                f"TOST equivalent: {tost_result.get('equivalent', 'N/A')}. "
                f"{'TRIGGERED – effect is practically zero.' if triggered else 'Not triggered.'}"
            ),
        )

    def evaluate_falsifier3(
        self,
        g_w1_ci: tuple[float, float],
        g_w2_ci: tuple[float, float],
    ) -> FalsifierResult:
        """Falsifier F3: Reversed human-evaluation effects.

        Triggered if either within-condition Hedges' g CI is entirely
        *negative* (effect in opposite direction).

        Parameters
        ----------
        g_w1_ci : tuple[float, float]
            95 % CI for Hedges' g, condition 1.
        g_w2_ci : tuple[float, float]
            95 % CI for Hedges' g, condition 2.

        Returns
        -------
        FalsifierResult
        """
        reversed_w1 = g_w1_ci[1] < 0
        reversed_w2 = g_w2_ci[1] < 0
        triggered = reversed_w1 or reversed_w2

        return FalsifierResult(
            name="F3: Reversed human effect",
            bound_gate="Gate 4",
            triggered=triggered,
            evidence={
                "g_w1_ci": g_w1_ci,
                "g_w2_ci": g_w2_ci,
                "reversed_w1": reversed_w1,
                "reversed_w2": reversed_w2,
            },
            description=(
                f"g_w1 CI = ({g_w1_ci[0]:.3f}, {g_w1_ci[1]:.3f}), "
                f"g_w2 CI = ({g_w2_ci[0]:.3f}, {g_w2_ci[1]:.3f}). "
                f"{'TRIGGERED' if triggered else 'Not triggered'}."
            ),
        )

    def evaluate_falsifier4(
        self,
        too_late_bridge_ratio_tost: dict,
    ) -> FalsifierResult:
        """Falsifier F4: Bridge-zone late-injection null.

        Triggered if the TOST test shows the bridge-zone ratio is
        equivalent to 1.0 (no late-injection effect).

        Parameters
        ----------
        too_late_bridge_ratio_tost : dict
            TOST result for the ratio of late-bridge M to early-bridge M.

        Returns
        -------
        FalsifierResult
        """
        triggered = too_late_bridge_ratio_tost.get("equivalent", False)

        return FalsifierResult(
            name="F4: Late-injection null",
            bound_gate="Gate 2B",
            triggered=triggered,
            evidence=too_late_bridge_ratio_tost,
            description=(
                f"Bridge ratio TOST equivalent: "
                f"{too_late_bridge_ratio_tost.get('equivalent', 'N/A')}. "
                f"{'TRIGGERED' if triggered else 'Not triggered'}."
            ),
        )

    def evaluate_falsifier5(
        self,
        c3_loss_ratio: float,
    ) -> FalsifierResult:
        """Falsifier F5: Norm-rescaling accounts for the effect.

        Triggered if the rescaled-to-original loss ratio (from C3) is
        close to 1.0 (meaning rescaling does not diminish the effect,
        implying the direction carries no information beyond scale).

        Parameters
        ----------
        c3_loss_ratio : float
            Mean ratio of rescaled M to original M from control C3.

        Returns
        -------
        FalsifierResult
        """
        # Triggered if rescaling retains ≥ 90 % of the effect
        triggered = c3_loss_ratio >= 0.90

        return FalsifierResult(
            name="F5: Norm-rescaling confound",
            bound_gate="Gate 5",
            triggered=triggered,
            evidence={"c3_loss_ratio": c3_loss_ratio},
            description=(
                f"C3 loss ratio = {c3_loss_ratio:.3f}. "
                f"{'TRIGGERED – rescaling retains effect.' if triggered else 'Not triggered.'}"
            ),
        )

    # =================================================================
    # Flags
    # =================================================================

    def evaluate_construct_flag(
        self,
        c1d_retention_ci: tuple[float, float],
    ) -> str:
        """Construct-validity flag based on spatial-shuffle retention.

        If the retention ratio CI includes 1.0, spatial structure may not
        carry information → flag.

        Parameters
        ----------
        c1d_retention_ci : tuple[float, float]
            95 % CI for the retention ratio from C1d.

        Returns
        -------
        str
            Flag description, or empty string if no flag.
        """
        if c1d_retention_ci[0] <= 1.0 <= c1d_retention_ci[1]:
            return (
                f"CONSTRUCT FLAG: C1d spatial-shuffle retention CI "
                f"({c1d_retention_ci[0]:.3f}, {c1d_retention_ci[1]:.3f}) "
                f"includes 1.0 – spatial structure may not carry "
                f"refusal-relevant information."
            )
        return ""

    def evaluate_harness_flag(
        self,
        c5_result: dict,
        typo_calibration: dict,
    ) -> str:
        """Harness-validity flag based on typographic-image behaviour.

        If typographic images have a very different baseline refusal rate
        from scene-based HS images, the harness may conflate OCR with
        visual scene recognition.

        Parameters
        ----------
        c5_result : dict
            Output from ``ControlRunner.run_c5_typographic``.
        typo_calibration : dict
            Additional calibration data (e.g., expected baseline rate).

        Returns
        -------
        str
            Flag description, or empty string if no flag.
        """
        baseline_rate = c5_result.get("baseline_refusal_rate", 0.0)
        expected_rate = typo_calibration.get("expected_baseline_rate", 0.8)

        # Flag if the baseline differs from expectation by > 20 pp
        if abs(baseline_rate - expected_rate) > 0.20:
            return (
                f"HARNESS FLAG: Typographic baseline refusal rate "
                f"({baseline_rate:.2%}) deviates from expected "
                f"({expected_rate:.2%}) by > 20 pp."
            )
        return ""

    # =================================================================
    # Hierarchical chain
    # =================================================================

    def run_hierarchical_chain(self, all_results: dict) -> Verdict:
        """Execute the full hierarchical gatekeeping chain.

        Runs gates 1 → 2A → 2B → 3 → 4 → 5 in order, stopping at the
        first failure.  For the failed gate, the corresponding falsifier
        is evaluated.  Metrics for untested gates are reported as
        descriptive only.

        Parameters
        ----------
        all_results : dict
            Dictionary containing all intermediate results keyed by
            experiment name.  Expected keys (examples):

            * ``phase1`` – ``{L_harm, L_harm_ci, L_read, L_read_ci}``
            * ``lmm`` – ``{beta_three_way, ci_95, tost_result}``
            * ``gate2b`` – ``{gamma_interaction, ci}``
            * ``gate3`` – ``{delta_selective, delta_selective_ci,
              ss_false_refusal, ss_fr_ci}``
            * ``gate4`` – ``{did, did_ci, g_w1, g_w1_ci, g_w2, g_w2_ci,
              attention_check_pass}``
            * ``controls`` – dict of control results
            * ``nonsafety`` – non-safety grid results
            * ``capability`` – capability benchmark results
            * ``c5`` – typographic-image results
            * ``typo_calibration`` – calibration data

        Returns
        -------
        Verdict
        """
        gates_passed: list[GateResult] = []
        falsifiers_triggered: list[FalsifierResult] = []
        flags: list[str] = []

        # ---- Collect flags (always evaluated) ----
        c1d_ci = all_results.get("controls", {}).get("c1d", {}).get(
            "retention_ci", (0.0, 0.0),
        )
        flag_c = self.evaluate_construct_flag(c1d_ci)
        if flag_c:
            flags.append(flag_c)

        flag_h = self.evaluate_harness_flag(
            all_results.get("c5", {}),
            all_results.get("typo_calibration", {}),
        )
        if flag_h:
            flags.append(flag_h)

        # ---- Define gate sequence ----
        phase1 = all_results.get("phase1", {})
        lmm = all_results.get("lmm", {})
        g2b = all_results.get("gate2b", {})
        g3 = all_results.get("gate3", {})
        g4 = all_results.get("gate4", {})
        controls = all_results.get("controls", {})
        nonsafety = all_results.get("nonsafety", {})
        capability = all_results.get("capability", {})

        gate_sequence = [
            (
                "gate1",
                lambda: self.evaluate_gate1(
                    L_harm=phase1.get("L_harm", 0),
                    L_harm_ci=phase1.get("L_harm_ci", (0, 0)),
                    L_read=phase1.get("L_read", 0),
                    L_read_ci=phase1.get("L_read_ci", (0, 0)),
                    typo_results=all_results.get("c5", {}),
                ),
            ),
            (
                "gate2a",
                lambda: self.evaluate_gate2a(
                    beta_three_way=lmm.get("beta_three_way", 0.0),
                    ci_95=lmm.get("ci_95", (0, 0)),
                    delta_equiv=self.delta_equiv,
                ),
            ),
            (
                "gate2b",
                lambda: self.evaluate_gate2b(
                    gamma_interaction=g2b.get("gamma_interaction", 0.0),
                    ci_95=g2b.get("ci", (0, 0)),
                ),
            ),
            (
                "gate3",
                lambda: self.evaluate_gate3(
                    delta_selective=g3.get("delta_selective", 0.0),
                    delta_selective_ci=g3.get("delta_selective_ci", (0, 0)),
                    ss_false_refusal=g3.get("ss_false_refusal", 0.0),
                    ss_fr_ci=g3.get("ss_fr_ci", (0, 0)),
                    eta_ss=self.eta_ss,
                ),
            ),
            (
                "gate4",
                lambda: self.evaluate_gate4(
                    did=g4.get("did", 0.0),
                    did_ci=g4.get("did_ci", (0, 0)),
                    g_w1=g4.get("g_w1", 0.0),
                    g_w1_ci=g4.get("g_w1_ci", (0, 0)),
                    g_w2=g4.get("g_w2", 0.0),
                    g_w2_ci=g4.get("g_w2_ci", (0, 0)),
                    attention_check_pass=g4.get("attention_check_pass", False),
                ),
            ),
            (
                "gate5",
                lambda: self.evaluate_gate5(
                    control_results=controls,
                    nonsafety_result=nonsafety,
                    capability_result=capability,
                ),
            ),
        ]

        # Mapping from gate names to their falsifier evaluators
        falsifier_map: dict[str, Optional[callable]] = {
            "gate1": lambda: self.evaluate_falsifier1(
                L_harm_L_read_gap_ci=phase1.get("gap_ci", (0, 0)),
                delta_layer_mei=self.delta_layer_mei,
            ),
            "gate2a": lambda: self.evaluate_falsifier2(
                beta_three_way=lmm.get("beta_three_way", 0.0),
                tost_result=lmm.get("tost_result", {}),
                gate2a_passed=False,
            ),
            "gate2b": lambda: self.evaluate_falsifier4(
                too_late_bridge_ratio_tost=g2b.get("bridge_ratio_tost", {}),
            ),
            "gate3": None,  # No specific falsifier for Gate 3
            "gate4": lambda: self.evaluate_falsifier3(
                g_w1_ci=g4.get("g_w1_ci", (0, 0)),
                g_w2_ci=g4.get("g_w2_ci", (0, 0)),
            ),
            "gate5": lambda: self.evaluate_falsifier5(
                c3_loss_ratio=controls.get("c3", {}).get("mean_loss_ratio", 0.0),
            ),
        }

        # ---- Run the chain ----
        failed_gate: Optional[GateResult] = None
        outcome = "SUPPORTED"

        for gate_key, gate_fn in gate_sequence:
            try:
                gate_result = gate_fn()
            except Exception as exc:
                logger.error("Gate %s raised an exception: %s", gate_key, exc)
                gate_result = GateResult(
                    gate_name=f"{gate_key} (ERROR)",
                    passed=False,
                    metric_value=float("nan"),
                    ci=(float("nan"), float("nan")),
                    threshold=float("nan"),
                    description=f"Evaluation error: {exc}",
                )

            if gate_result.passed:
                gates_passed.append(gate_result)
            else:
                failed_gate = gate_result
                outcome = "INCONCLUSIVE"

                # Evaluate falsifier for the failed gate
                falsifier_fn = falsifier_map.get(gate_key)
                if falsifier_fn is not None:
                    try:
                        f_result = falsifier_fn()
                        if f_result.triggered:
                            falsifiers_triggered.append(f_result)
                            outcome = "FALSIFIED"
                    except Exception as exc:
                        logger.error(
                            "Falsifier for %s raised an exception: %s",
                            gate_key, exc,
                        )

                break  # Stop the chain

        # ---- Narrative ----
        narrative_parts: list[str] = []

        if outcome == "SUPPORTED":
            narrative_parts.append(
                "All gates passed. The dual-clock hypothesis is SUPPORTED."
            )
        elif outcome == "FALSIFIED":
            narrative_parts.append(
                f"Chain stopped at {failed_gate.gate_name}. "
                f"Falsifier triggered → hypothesis FALSIFIED."
            )
        else:
            narrative_parts.append(
                f"Chain stopped at {failed_gate.gate_name}. "
                f"No falsifier triggered → INCONCLUSIVE."
            )

        narrative_parts.append(
            f"Gates passed: {len(gates_passed)} / {len(gate_sequence)}."
        )

        for g in gates_passed:
            narrative_parts.append(f"  ✓ {g.gate_name}")

        if failed_gate:
            narrative_parts.append(f"  ✗ {failed_gate.gate_name}: {failed_gate.description}")

        for f in falsifiers_triggered:
            narrative_parts.append(f"  ⚠ {f.name}: {f.description}")

        for fl in flags:
            narrative_parts.append(f"  🚩 {fl}")

        narrative = "\n".join(narrative_parts)

        return Verdict(
            outcome=outcome,
            gates_passed=gates_passed,
            failed_gate=failed_gate,
            falsifiers_triggered=falsifiers_triggered,
            flags=flags,
            narrative=narrative,
        )
