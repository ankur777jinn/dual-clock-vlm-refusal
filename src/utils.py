"""Common utilities for the dual-clock VLM refusal experiment.

Provides seeding, device selection, hashing, logit-mass computation,
calibration, result containers, bootstrapping, and caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import CACHE_DIR, SEED


# ---------------------------------------------------------------------------
# Reproducibility & hardware
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for ``torch``, ``numpy``, and ``random``.

    Also enables deterministic cuDNN behaviour when possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False       # type: ignore[attr-defined]


def get_device() -> torch.device:
    """Return the best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_hash(text: str) -> str:
    """Return the hex SHA-256 digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Logit-mass metric  M = logsumexp(refusal) − logsumexp(compliance)
# ---------------------------------------------------------------------------

def compute_logit_mass(
    logits: torch.Tensor,
    refusal_ids: List[int],
    compliance_ids: List[int],
) -> float:
    """Compute the refusal logit-mass differential *M*.

    .. math::

        M = \\log\\sum_{r \\in \\text{refusal}} e^{z_r}
          - \\log\\sum_{c \\in \\text{compliance}} e^{z_c}

    Parameters
    ----------
    logits : torch.Tensor
        1-D tensor of logits (vocabulary dimension) for a single
        next-token prediction.
    refusal_ids : list[int]
        Token IDs that signal refusal.
    compliance_ids : list[int]
        Token IDs that signal compliance.

    Returns
    -------
    float
        The scalar *M* value.  Positive → model leans toward refusal.
    """
    logits = logits.float()  # promote for numerical stability

    ref_logits = logits[refusal_ids]
    comp_logits = logits[compliance_ids]

    lse_ref = torch.logsumexp(ref_logits, dim=0)
    lse_comp = torch.logsumexp(comp_logits, dim=0)

    return (lse_ref - lse_comp).item()


# ---------------------------------------------------------------------------
# Calibration: M → pp via logistic regression
# ---------------------------------------------------------------------------

def calibrate_M_to_pp(
    M_values: Sequence[float],
    refusal_labels: Sequence[int],
) -> Tuple[float, float]:
    """Fit a logistic mapping from raw *M* to probability of refusal.

    Solves  P(refusal | M) = σ(a · M + b)  via maximum-likelihood.

    Parameters
    ----------
    M_values : sequence of float
        Observed logit-mass values.
    refusal_labels : sequence of int
        Binary labels (1 = refusal, 0 = compliance).

    Returns
    -------
    tuple[float, float]
        Coefficients ``(a, b)`` of the fitted logistic model.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(M_values, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(refusal_labels, dtype=np.int32)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X, y)

    a: float = float(clf.coef_[0, 0])
    b: float = float(clf.intercept_[0])
    return a, b


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    """Container for one activation-patching trial.

    Attributes
    ----------
    cell : tuple[int, int]
        ``(src_layer, dst_layer)`` grid coordinates.
    image_id : str
        Identifier of the input image.
    M : float
        Logit-mass differential.
    generated_text : str
        Decoded output from the model.
    judge_label : str
        Label assigned by the external judge (e.g. ``"refusal"``,
        ``"compliance"``, ``"incoherent"``).
    metadata : dict
        Any additional fields (alpha, cohort, etc.).
    """
    cell: Tuple[int, int]
    image_id: str
    M: float
    generated_text: str
    judge_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,  # type: ignore[arg-type]
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = SEED,
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data : sequence of float
        Observed data points.
    stat_fn : callable
        Statistic to compute on each bootstrap sample (default: mean).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (CI is ``100·(1 − α)%``).
    rng_seed : int
        Seed for the bootstrap RNG.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds of the CI.
    """
    rng = np.random.RandomState(rng_seed)
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


# ---------------------------------------------------------------------------
# Results cache with SHA-256 integrity
# ---------------------------------------------------------------------------

class ResultsCache:
    """Persist and reload :class:`PatchResult` lists with integrity checks.

    Files are stored as JSON under :data:`CACHE_DIR` and accompanied by
    a ``.sha256`` sidecar containing the content hash.

    Parameters
    ----------
    cache_dir : Path | None
        Override the default cache directory.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _data_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _hash_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.sha256"

    def save(self, key: str, results: List[PatchResult]) -> Path:
        """Serialise *results* to disk and write a SHA-256 sidecar.

        Parameters
        ----------
        key : str
            A short, filesystem-safe identifier (e.g. ``"sweep_tuning"``).
        results : list[PatchResult]
            Results to persist.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        payload = json.dumps(
            [asdict(r) for r in results],
            indent=2,
            ensure_ascii=False,
        )
        data_path = self._data_path(key)
        data_path.write_text(payload, encoding="utf-8")

        digest = sha256_hash(payload)
        self._hash_path(key).write_text(digest, encoding="utf-8")

        return data_path

    def load(self, key: str, verify: bool = True) -> List[PatchResult]:
        """Load results from disk, optionally verifying integrity.

        Parameters
        ----------
        key : str
            Cache key used during :meth:`save`.
        verify : bool
            If *True*, recompute SHA-256 and compare to the stored hash.

        Returns
        -------
        list[PatchResult]
            Deserialised result objects.

        Raises
        ------
        FileNotFoundError
            If the cache file does not exist.
        ValueError
            If *verify* is True and the hash does not match.
        """
        data_path = self._data_path(key)
        if not data_path.exists():
            raise FileNotFoundError(f"No cached results for key '{key}'")

        payload = data_path.read_text(encoding="utf-8")

        if verify:
            hash_path = self._hash_path(key)
            if hash_path.exists():
                stored = hash_path.read_text(encoding="utf-8").strip()
                computed = sha256_hash(payload)
                if stored != computed:
                    raise ValueError(
                        f"Integrity check failed for '{key}': "
                        f"stored={stored!r}, computed={computed!r}"
                    )

        raw: List[Dict[str, Any]] = json.loads(payload)
        results: List[PatchResult] = []
        for entry in raw:
            # Reconstruct tuple for 'cell'
            entry["cell"] = tuple(entry["cell"])
            results.append(PatchResult(**entry))
        return results

    def exists(self, key: str) -> bool:
        """Check whether a cache entry exists for *key*."""
        return self._data_path(key).exists()
