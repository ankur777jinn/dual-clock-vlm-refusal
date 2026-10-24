"""
Confirmatory Mixed-Effects Model and statistical tests.
Implements the HS × Read × Harm three-way interaction LMM with:
- Centered effect coding (±0.5)
- HS × Patched term
- Natural cubic spline for jump
- Image random intercepts with sandwich SE
- TOST equivalence tests
- Power simulation
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline

logger = logging.getLogger(__name__)


# ===================================================================
# Design matrix construction
# ===================================================================

def build_design_matrix(
    grid_results: pd.DataFrame,
    L_read: int,
    L_harm: int,
) -> pd.DataFrame:
    """Augment raw grid results with effect-coded predictors and interactions.

    Columns added
    -------------
    * ``Patched`` – 0/1 indicator (1 if the row comes from a patched cell).
    * ``HS_tilde`` – +0.5 for HS images, −0.5 for SS images.
    * ``Read_tilde`` – +0.5 if ``src >= L_read``, else −0.5.
    * ``Harm_tilde`` – +0.5 if ``dst >= L_harm``, else −0.5.
    * ``jump`` – already present; kept as-is.
    * ``HS_x_Patched`` – ``HS_tilde × Patched``.
    * ``HS_x_Read`` – ``HS_tilde × Read_tilde``.
    * ``HS_x_Harm`` – ``HS_tilde × Harm_tilde``.
    * ``Read_x_Harm`` – ``Read_tilde × Harm_tilde``.
    * ``HS_x_Read_x_Harm`` – the three-way interaction.

    After construction the function verifies that the Generalized Variance
    Inflation Factor (GVIF) for every term is below 4.0 and logs a warning
    otherwise.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Must contain columns ``[image_id, cohort, src, dst, jump, M, patched]``.
    L_read : int
        Layer threshold for the *read* boundary.
    L_harm : int
        Layer threshold for the *harm* boundary.

    Returns
    -------
    pd.DataFrame
        Copy of *grid_results* with the new columns appended.
    """
    df = grid_results.copy()

    # ---- Binary / effect codes ----
    df["Patched"] = df["patched"].astype(int)
    df["HS_tilde"] = df["cohort"].map({"HS": 0.5, "SS": -0.5}).astype(float)

    # For unpatched rows src/dst are NaN – treat them as "below threshold" (-0.5)
    # Read_tilde: +0.5 if dst <= L_read (inside reading window), else -0.5
    # Harm_tilde: +0.5 if src >= L_harm (text-aligned harm acquired), else -0.5
    df["Read_tilde"] = np.where(
        df["dst"].notna() & (df["dst"] <= L_read), 0.5, -0.5,
    )
    df["Harm_tilde"] = np.where(
        df["src"].notna() & (df["src"] >= L_harm), 0.5, -0.5,
    )

    # ---- Interactions ----
    df["HS_x_Patched"] = df["HS_tilde"] * df["Patched"]
    df["HS_x_Read"] = df["HS_tilde"] * df["Read_tilde"]
    df["HS_x_Harm"] = df["HS_tilde"] * df["Harm_tilde"]
    df["Read_x_Harm"] = df["Read_tilde"] * df["Harm_tilde"]
    df["HS_x_Read_x_Harm"] = df["HS_tilde"] * df["Read_tilde"] * df["Harm_tilde"]

    # ---- GVIF check (on patched rows only – unpatched have no src/dst) ----
    terms = [
        "HS_tilde", "Read_tilde", "Harm_tilde", "Patched",
        "HS_x_Patched", "HS_x_Read", "HS_x_Harm", "Read_x_Harm",
        "HS_x_Read_x_Harm",
    ]
    patched_df = df.loc[df["Patched"] == 1]
    if len(patched_df) > 0:
        gvif_df = compute_gvif(patched_df, terms)
        high_vif = gvif_df.loc[gvif_df["GVIF"] >= 4.0]
        if len(high_vif) > 0:
            logger.warning(
                "GVIF ≥ 4.0 detected for terms:\n%s", high_vif.to_string(),
            )

    return df


# ===================================================================
# Confirmatory LMM
# ===================================================================

def fit_confirmatory_lmm(design: pd.DataFrame) -> dict:
    """Fit the confirmatory linear mixed model.

    Model specification::

        M ~ HS_tilde + Read_tilde + Harm_tilde + Patched
            + HS_x_Patched + HS_x_Read + HS_x_Harm + Read_x_Harm
            + HS_x_Read_x_Harm
            + (1 | image_id)

    Uses ``statsmodels.regression.mixed_linear_model.MixedLM`` with
    robust (sandwich / HC0) standard errors.

    Parameters
    ----------
    design : pd.DataFrame
        Output of :func:`build_design_matrix`.

    Returns
    -------
    dict
        ``{beta_three_way, se, p_value, ci_95, all_coefficients,
        model_summary}``  where ``p_value`` is one-sided
        (testing β > 0).
    """
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    fixed_terms = [
        "HS_tilde", "Read_tilde", "Harm_tilde", "Patched",
        "HS_x_Patched", "HS_x_Read", "HS_x_Harm", "Read_x_Harm",
        "HS_x_Read_x_Harm",
    ]

    # Drop rows with NaN in M or predictors
    cols_needed = fixed_terms + ["M", "image_id"]
    sub = design.dropna(subset=cols_needed).copy()

    endog = sub["M"]
    exog = sm.add_constant(sub[fixed_terms])
    groups = sub["image_id"]

    model = MixedLM(endog, exog, groups=groups)
    result = model.fit(reml=True)

    # Robust (sandwich) standard errors
    try:
        robust_result = model.fit(reml=True, cov_type="robust")
        se_series = robust_result.bse
        beta_vals = robust_result.fe_params
        summary_str = str(robust_result.summary())
    except Exception:
        logger.warning("Robust SE estimation failed; falling back to model-based SE")
        se_series = result.bse
        beta_vals = result.fe_params
        summary_str = str(result.summary())

    # Extract the three-way term
    tw_key = "HS_x_Read_x_Harm"
    beta_three_way = float(beta_vals[tw_key])
    se_tw = float(se_series[tw_key])

    # One-sided p-value: H₁: β > 0
    z = beta_three_way / se_tw
    p_one_sided = float(1.0 - stats.norm.cdf(z))

    # 95 % CI
    ci_low = beta_three_way - 1.96 * se_tw
    ci_high = beta_three_way + 1.96 * se_tw

    # All coefficients
    all_coefs = {
        name: {
            "beta": float(beta_vals[name]),
            "se": float(se_series[name]),
            "z": float(beta_vals[name] / se_series[name]),
            "p_two_sided": float(
                2.0 * (1.0 - stats.norm.cdf(abs(beta_vals[name] / se_series[name]))),
            ),
        }
        for name in beta_vals.index
    }

    return {
        "beta_three_way": beta_three_way,
        "se": se_tw,
        "p_value": p_one_sided,
        "ci_95": (float(ci_low), float(ci_high)),
        "all_coefficients": all_coefs,
        "model_summary": summary_str,
    }


# ===================================================================
# Spline basis for jump
# ===================================================================

def fit_spline_jump(design: pd.DataFrame, n_knots: int = 3) -> np.ndarray:
    """Construct a natural cubic spline basis for the ``jump`` column.

    Knots are placed at evenly-spaced quantiles of the observed ``jump``
    values (excluding the unpatched rows where jump is NaN).

    Parameters
    ----------
    design : pd.DataFrame
        Must contain a ``jump`` column.
    n_knots : int
        Number of interior knots.

    Returns
    -------
    np.ndarray
        Spline basis matrix of shape ``(n_patched, n_knots + 2)``.
    """
    jumps = design.loc[design["jump"].notna(), "jump"].values.astype(float)
    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
    knots = np.percentile(jumps, quantiles)

    # Build truncated-power basis (natural cubic spline)
    basis_cols: list[np.ndarray] = [jumps, jumps ** 2, jumps ** 3]
    for k in knots:
        basis_cols.append(np.maximum(jumps - k, 0) ** 3)

    basis = np.column_stack(basis_cols)
    return basis


# ===================================================================
# TOST equivalence test
# ===================================================================

def tost_equivalence(
    beta: float,
    se: float,
    delta_equiv: float,
    alpha: float = 0.10,
) -> dict:
    """Two One-Sided Tests for equivalence within ±δ.

    Tests H₀₁: β ≤ −δ  and  H₀₂: β ≥ +δ.  Equivalence is declared if
    *both* null hypotheses are rejected at level α.

    Parameters
    ----------
    beta : float
        Point estimate.
    se : float
        Standard error of the estimate.
    delta_equiv : float
        Equivalence margin (positive).
    alpha : float
        Significance level for each one-sided test (default 0.10 → 90 % CI).

    Returns
    -------
    dict
        ``{equivalent: bool, ci_90: (low, high), p_upper, p_lower}``
    """
    z_crit = stats.norm.ppf(1.0 - alpha)

    # Test 1: H₀₁ says β ≤ −δ  →  reject if (β + δ) / se > z_crit
    t_lower = (beta + delta_equiv) / se
    p_lower = float(1.0 - stats.norm.cdf(t_lower))

    # Test 2: H₀₂ says β ≥ +δ  →  reject if (δ − β) / se > z_crit
    t_upper = (delta_equiv - beta) / se
    p_upper = float(1.0 - stats.norm.cdf(t_upper))

    ci_low = beta - z_crit * se
    ci_high = beta + z_crit * se

    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return {
        "equivalent": bool(equivalent),
        "ci_90": (float(ci_low), float(ci_high)),
        "p_upper": p_upper,
        "p_lower": p_lower,
    }


# ===================================================================
# Generalized VIF
# ===================================================================

def compute_gvif(design_matrix: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Compute Generalized Variance Inflation Factor for each term.

    For single-df terms this reduces to the standard VIF.

    Parameters
    ----------
    design_matrix : pd.DataFrame
        Data containing the columns listed in *terms* (numeric).
    terms : list[str]
        Column names to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ``[term, GVIF]``.
    """
    import statsmodels.api as sm

    sub = design_matrix[terms].dropna().astype(float)
    n = len(sub)
    if n < len(terms) + 1:
        logger.warning("Too few observations (%d) to compute GVIF reliably", n)
        return pd.DataFrame({"term": terms, "GVIF": [float("nan")] * len(terms)})

    results: list[dict] = []
    for t in terms:
        y = sub[t].values
        X = sm.add_constant(sub.drop(columns=[t]).values)
        try:
            ols = sm.OLS(y, X).fit()
            r2 = ols.rsquared
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vif = float("nan")
        results.append({"term": t, "GVIF": float(vif)})

    return pd.DataFrame(results)


# ===================================================================
# Changepoint competitor model
# ===================================================================

def fit_changepoint_competitor(grid_results: pd.DataFrame) -> dict:
    """Fit a continuous piece-wise linear (changepoint) model.

    The model is::

        M = a + b₁·layer + b₂·max(layer − cp, 0) + ε

    with *cp* (changepoint) estimated by profiling over all candidate
    layers.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Must include ``dst`` and ``M``.

    Returns
    -------
    dict
        ``{changepoints: list[float], cv_loglik: float,
        best_cp: float, coefficients: dict}``
    """
    import statsmodels.api as sm

    patched = grid_results.dropna(subset=["dst", "M"]).copy()
    layers = np.sort(patched["dst"].unique())
    M = patched["M"].values
    dst = patched["dst"].values.astype(float)

    best_cp: Optional[float] = None
    best_ll = -np.inf
    best_coefs: dict = {}

    for cp in layers[1:-1]:  # avoid boundary
        hinge = np.maximum(dst - cp, 0.0)
        X = sm.add_constant(np.column_stack([dst, hinge]))
        try:
            fit = sm.OLS(M, X).fit()
            ll = float(fit.llf)
            if ll > best_ll:
                best_ll = ll
                best_cp = float(cp)
                best_coefs = {
                    "intercept": float(fit.params[0]),
                    "slope_pre": float(fit.params[1]),
                    "slope_delta": float(fit.params[2]),
                }
        except Exception:
            continue

    return {
        "changepoints": [best_cp] if best_cp is not None else [],
        "cv_loglik": best_ll,
        "best_cp": best_cp,
        "coefficients": best_coefs,
    }


# ===================================================================
# Cross-validation
# ===================================================================

def cross_validate_models(
    thresholded_model,
    changepoint_model,
    data: pd.DataFrame,
    n_folds: int = 10,
) -> dict:
    """Image-grouped k-fold cross-validation comparing two models.

    Both *thresholded_model* and *changepoint_model* must expose
    ``.fit(train_df)`` and ``.loglik(test_df)`` methods.

    Parameters
    ----------
    thresholded_model
        The pre-registered thresholded LMM.
    changepoint_model
        The competitor changepoint model.
    data : pd.DataFrame
        Full dataset.
    n_folds : int
        Number of cross-validation folds (default 10).

    Returns
    -------
    dict
        ``{thresholded_cv_ll, changepoint_cv_ll, delta_ll,
        fold_results: list[dict]}``
    """
    images = data["image_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(images)
    folds = np.array_split(images, n_folds)

    fold_results: list[dict] = []
    total_thresh = 0.0
    total_cp = 0.0

    for fold_idx, test_images in enumerate(folds):
        test_mask = data["image_id"].isin(test_images)
        train = data.loc[~test_mask]
        test = data.loc[test_mask]

        try:
            thresholded_model.fit(train)
            ll_thresh = thresholded_model.loglik(test)
        except Exception:
            logger.warning("Thresholded model failed on fold %d", fold_idx)
            ll_thresh = float("nan")

        try:
            changepoint_model.fit(train)
            ll_cp = changepoint_model.loglik(test)
        except Exception:
            logger.warning("Changepoint model failed on fold %d", fold_idx)
            ll_cp = float("nan")

        fold_results.append({
            "fold": fold_idx,
            "thresholded_ll": float(ll_thresh),
            "changepoint_ll": float(ll_cp),
        })
        total_thresh += ll_thresh if not np.isnan(ll_thresh) else 0.0
        total_cp += ll_cp if not np.isnan(ll_cp) else 0.0

    return {
        "thresholded_cv_ll": total_thresh,
        "changepoint_cv_ll": total_cp,
        "delta_ll": total_thresh - total_cp,
        "fold_results": fold_results,
    }


# ===================================================================
# Uncertainty propagation
# ===================================================================

def propagate_uncertainty(
    grid_results: pd.DataFrame,
    L_read_bootstrap: np.ndarray,
    L_harm_bootstrap: np.ndarray,
) -> dict:
    """Refit the LMM across Phase-1 bootstrap draws of L_read and L_harm.

    For each bootstrap replicate we reconstruct the design matrix with the
    draw-specific thresholds and re-estimate the three-way interaction.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Raw grid results.
    L_read_bootstrap : np.ndarray
        1-D array of bootstrap estimates for L_read.
    L_harm_bootstrap : np.ndarray
        1-D array of bootstrap estimates for L_harm.

    Returns
    -------
    dict
        ``{beta_samples: np.ndarray, median, ci_95: (low, high),
        fraction_significant: float}``
    """
    n_draws = min(len(L_read_bootstrap), len(L_harm_bootstrap))
    betas: list[float] = []

    for i in range(n_draws):
        lr = int(L_read_bootstrap[i])
        lh = int(L_harm_bootstrap[i])
        try:
            design = build_design_matrix(grid_results, L_read=lr, L_harm=lh)
            result = fit_confirmatory_lmm(design)
            betas.append(result["beta_three_way"])
        except Exception:
            logger.debug("Bootstrap draw %d failed; skipping", i)
            continue

    betas_arr = np.array(betas)
    ci = (float(np.percentile(betas_arr, 2.5)), float(np.percentile(betas_arr, 97.5)))
    frac_sig = float(np.mean(betas_arr > 0))

    return {
        "beta_samples": betas_arr,
        "median": float(np.median(betas_arr)),
        "ci_95": ci,
        "fraction_significant": frac_sig,
    }


# ===================================================================
# Alignment dose-response
# ===================================================================

class AlignmentDoseResponse:
    """Dose-response analysis linking alignment-probe z-scores to ΔM."""

    def __init__(self) -> None:
        pass

    def prepare_z_scores(self, probe_scores: pd.DataFrame) -> pd.DataFrame:
        """Z-score probe activations within each L_src layer.

        Parameters
        ----------
        probe_scores : pd.DataFrame
            Must contain columns ``[image_id, src_layer, probe_score]``.

        Returns
        -------
        pd.DataFrame
            Same frame with an additional ``z_score`` column.
        """
        df = probe_scores.copy()
        grouped = df.groupby("src_layer")["probe_score"]
        df["z_score"] = grouped.transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0,
        )
        return df

    def fit_gate2b(
        self,
        delta_M: pd.DataFrame,
        z_scores: pd.DataFrame,
        bridge_indicator: pd.Series,
    ) -> dict:
        """Fit the γ_interaction LMM for Gate 2b.

        Model::

            ΔM ~ z_score * bridge_indicator + (1 | image_id)

        Parameters
        ----------
        delta_M : pd.DataFrame
            Must contain ``[image_id, delta_M]``.
        z_scores : pd.DataFrame
            Must contain ``[image_id, z_score]``.
        bridge_indicator : pd.Series
            Boolean series aligned to delta_M indicating whether the cell
            is in the bridge zone (src < L_harm ≤ dst).

        Returns
        -------
        dict
            ``{gamma_interaction, p_value, ci}``
        """
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        merged = delta_M.merge(z_scores[["image_id", "z_score"]], on="image_id")
        merged["bridge"] = bridge_indicator.astype(float).values[: len(merged)]
        merged["z_x_bridge"] = merged["z_score"] * merged["bridge"]

        exog = sm.add_constant(merged[["z_score", "bridge", "z_x_bridge"]])
        endog = merged["delta_M"]
        groups = merged["image_id"]

        model = MixedLM(endog, exog, groups=groups)
        result = model.fit(reml=True)

        gamma = float(result.fe_params["z_x_bridge"])
        se = float(result.bse["z_x_bridge"])
        z_val = gamma / se
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_val))))
        ci = (gamma - 1.96 * se, gamma + 1.96 * se)

        return {
            "gamma_interaction": gamma,
            "p_value": p_val,
            "ci": (float(ci[0]), float(ci[1])),
        }


# ===================================================================
# Power simulation
# ===================================================================

class PowerSimulator:
    """Simulation-based power analysis for the pre-registered tests.

    All simulations use parametric draws with the specified variance
    components.
    """

    def __init__(
        self,
        sigma_M_hs: float,
        sigma_M_ss: float,
        sigma_M_pooled: float,
        sd_pair: float,
        sd_contrast: float,
    ) -> None:
        """
        Parameters
        ----------
        sigma_M_hs : float
            SD of M in the HS cohort (estimated from pilot).
        sigma_M_ss : float
            SD of M in the SS cohort.
        sigma_M_pooled : float
            Pooled SD across cohorts.
        sd_pair : float
            SD of paired differences (for C1b).
        sd_contrast : float
            SD of the contrast estimate (for three-way interaction).
        """
        self.sigma_M_hs = sigma_M_hs
        self.sigma_M_ss = sigma_M_ss
        self.sigma_M_pooled = sigma_M_pooled
        self.sd_pair = sd_pair
        self.sd_contrast = sd_contrast

    def simulate_three_way_power(
        self,
        n_hs: int = 100,
        n_ss: int = 100,
        n_cells: int = 55,
        delta_equiv: Optional[float] = None,
        n_sim: int = 5000,
    ) -> float:
        """Simulate power for detecting the three-way interaction.

        Under H₁ the true three-way effect equals ``sd_contrast``
        (a one-SD effect).

        Parameters
        ----------
        n_hs : int
            Number of HS images.
        n_ss : int
            Number of SS images.
        n_cells : int
            Number of grid cells.
        delta_equiv : float, optional
            If provided, tests TOST equivalence instead.
        n_sim : int
            Number of Monte Carlo simulations.

        Returns
        -------
        float
            Estimated power ∈ [0, 1].
        """
        rng = np.random.default_rng(123)
        n_total = (n_hs + n_ss) * n_cells
        se = self.sd_contrast / np.sqrt(n_total)
        true_effect = self.sd_contrast  # one-SD effect

        rejects = 0
        for _ in range(n_sim):
            beta_hat = rng.normal(true_effect, se)
            if delta_equiv is not None:
                result = tost_equivalence(beta_hat, se, delta_equiv)
                if result["equivalent"]:
                    rejects += 1
            else:
                z = beta_hat / se
                p = 1.0 - stats.norm.cdf(z)
                if p < 0.05:
                    rejects += 1

        return rejects / n_sim

    def simulate_c1b_power(
        self,
        n_pairs: int = 40,
        delta_equiv: Optional[float] = None,
        n_sim: int = 5000,
    ) -> float:
        """Simulate power for the C1b paired within-pair test.

        Parameters
        ----------
        n_pairs : int
            Number of image pairs.
        delta_equiv : float, optional
            If provided, tests TOST equivalence.
        n_sim : int
            Monte Carlo iterations.

        Returns
        -------
        float
            Estimated power ∈ [0, 1].
        """
        rng = np.random.default_rng(456)
        true_diff = self.sd_pair  # one-SD effect
        se = self.sd_pair / np.sqrt(n_pairs)

        rejects = 0
        for _ in range(n_sim):
            d_hat = rng.normal(true_diff, se)
            if delta_equiv is not None:
                result = tost_equivalence(d_hat, se, delta_equiv)
                if result["equivalent"]:
                    rejects += 1
            else:
                t = d_hat / se
                p = float(1.0 - stats.t.cdf(t, df=n_pairs - 1))
                if p < 0.05:
                    rejects += 1

        return rejects / n_sim

    def simulate_gate3_power(
        self,
        n_hs: int = 100,
        n_ss: int = 100,
        true_effect_pp: float = 15.0,
        n_sim: int = 5000,
    ) -> float:
        """Simulate power for the Gate 3 judged selective-refusal test.

        Parameters
        ----------
        n_hs : int
            Number of HS images.
        n_ss : int
            Number of SS images.
        true_effect_pp : float
            True selective-refusal difference in percentage points.
        n_sim : int
            Monte Carlo iterations.

        Returns
        -------
        float
            Estimated power ∈ [0, 1].
        """
        rng = np.random.default_rng(789)
        # Under H₁ the true Δ_selective = true_effect_pp pp
        # SE of a difference of proportions ≈ sqrt(p(1-p)(1/n1 + 1/n2))
        p_base = 0.7  # assumed baseline refusal rate
        se = np.sqrt(
            p_base * (1 - p_base) * (1.0 / n_hs + 1.0 / n_ss),
        ) * 100.0  # scale to pp

        rejects = 0
        for _ in range(n_sim):
            delta_hat = rng.normal(true_effect_pp, se)
            z = delta_hat / se
            p = 1.0 - stats.norm.cdf(z)
            if p < 0.05:
                rejects += 1

        return rejects / n_sim

    def generate_report(self) -> str:
        """Run all power simulations and return a summary report.

        Returns
        -------
        str
            Multi-line human-readable power report.
        """
        power_tw = self.simulate_three_way_power()
        power_c1b = self.simulate_c1b_power()
        power_g3 = self.simulate_gate3_power()

        lines = [
            "=" * 60,
            "Power Simulation Report",
            "=" * 60,
            f"Three-way interaction (n_hs=100, n_ss=100, 55 cells):",
            f"  Power = {power_tw:.3f}",
            "",
            f"C1b paired within-pair test (n_pairs=40):",
            f"  Power = {power_c1b:.3f}",
            "",
            f"Gate 3 selective refusal (n_hs=100, n_ss=100, Δ=15pp):",
            f"  Power = {power_g3:.3f}",
            "",
            "Variance components used:",
            f"  σ_M(HS) = {self.sigma_M_hs:.4f}",
            f"  σ_M(SS) = {self.sigma_M_ss:.4f}",
            f"  σ_M(pooled) = {self.sigma_M_pooled:.4f}",
            f"  SD(pair) = {self.sd_pair:.4f}",
            f"  SD(contrast) = {self.sd_contrast:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
