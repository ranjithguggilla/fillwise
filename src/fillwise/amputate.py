"""Controlled amputation: punch holes into a complete table.

Three textbook mechanisms, all seeded:

* MCAR — every cell of the target columns is masked with equal probability.
* MAR — the probability a cell goes missing depends on a *different, fully
  observed* driver column (logistic in the driver's z-score). Rows with high
  driver values lose more cells.
* MNAR — the probability depends on the masked value itself: larger values
  are more likely to vanish, which is exactly the case naive imputers get
  wrong (they shrink the tail toward the mean).

Rates are calibrated per column so the realised missing fraction matches the
requested one in expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrated_probs(score: np.ndarray, rate: float, steepness: float = 1.5) -> np.ndarray:
    """Logistic probabilities in `score` with intercept tuned to hit `rate`."""
    z = (score - score.mean()) / (score.std() + 1e-12)
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = _sigmoid(steepness * z + mid)
        if p.mean() > rate:
            hi = mid
        else:
            lo = mid
    return _sigmoid(steepness * z + (lo + hi) / 2)


def amputate(
    X: pd.DataFrame,
    mechanism: str,
    rate: float,
    columns: list[str] | None = None,
    driver: str | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (amputated copy, boolean mask of removed cells)."""
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    if mechanism not in {"mcar", "mar", "mnar"}:
        raise ValueError(f"unknown mechanism: {mechanism}")
    rng = np.random.default_rng(seed)
    cols = columns or list(X.columns)
    if mechanism == "mar":
        if driver is None:
            candidates = [c for c in X.columns if c not in cols]
            if not candidates:
                raise ValueError("MAR needs a fully observed driver column outside `columns`")
            driver = candidates[0]
        if driver in cols:
            raise ValueError("driver must stay fully observed")
    out = X.copy()
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for col in cols:
        if mechanism == "mcar":
            p = np.full(len(X), rate)
        elif mechanism == "mar":
            p = _calibrated_probs(X[driver].to_numpy(dtype=float), rate)
        else:  # mnar: the value itself drives its own disappearance
            p = _calibrated_probs(X[col].to_numpy(dtype=float), rate)
        hit = rng.random(len(X)) < p
        out.loc[hit, col] = np.nan
        mask.loc[hit, col] = True
    return out, mask
