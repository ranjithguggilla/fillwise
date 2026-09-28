"""Amputation study: known ground truth, controlled damage, measured recovery.

The breast cancer table (569 rows, 30 numeric columns, zero missing values)
serves as ground truth. The ten mean_* columns get amputated under each
mechanism at 10/30/50% rates; worst_area stays observed and drives the MAR
pattern, so sicker-looking cases lose more of their measurements — the
pattern clinical data actually shows.

Two readouts per (mechanism, rate, strategy, seed):

* reconstruction — per-column NRMSE on the masked cells (imputer fitted on
  a train split only, evaluated on held-out rows' masked cells);
* downstream — 5-fold CV ROC-AUC of a logistic regression trained on the
  imputed table, against the 0.995 AUC ceiling of the intact table.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .amputate import amputate
from .data import Dataset, load_breast_cancer_complete
from .strategies import SEED, _numeric_imputer, build_logreg

MASK_COLUMNS_N = 10
MECHANISMS = ("mcar", "mar", "mnar")
RATES = (0.1, 0.3, 0.5)
SIM_STRATEGIES = ("mean", "median", "median_flag", "knn", "iterative")
DRIVER = "worst_area"


def _mask_columns(ds: Dataset) -> list[str]:
    return list(ds.X.columns[:MASK_COLUMNS_N])


def reconstruction_nrmse(
    X_true: pd.DataFrame, X_amp: pd.DataFrame, mask: pd.DataFrame, strategy: str, seed: int
) -> float:
    """Fit imputer on train rows of the amputated table, score masked test cells."""
    if strategy == "median_flag":
        strategy = "median"  # indicators do not change the filled values
    idx_tr, idx_te = train_test_split(X_true.index, test_size=0.35, random_state=seed)
    imp = _numeric_imputer(strategy, seed)
    imp.fit(X_amp.loc[idx_tr])
    filled = pd.DataFrame(
        np.asarray(imp.transform(X_amp.loc[idx_te]))[:, : X_true.shape[1]],
        index=idx_te,
        columns=X_true.columns,
    )
    errs = []
    for col in X_true.columns:
        if not mask[col].any():
            continue
        m = mask.loc[idx_te, col]
        if m.sum() < 3:
            continue
        true_v = X_true.loc[idx_te, col][m]
        pred_v = filled[col][m]
        sd = X_true[col].std()
        errs.append(np.sqrt(np.mean((true_v - pred_v) ** 2)) / (sd + 1e-12))
    return float(np.mean(errs)) if errs else float("nan")


def downstream_auc(ds_name: str, X: pd.DataFrame, y: np.ndarray, strategy: str) -> float:
    pipe = build_logreg(strategy, list(X.columns), [])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def run_amputation_study(seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    ds = load_breast_cancer_complete()
    cols = _mask_columns(ds)
    ceiling = downstream_auc(ds.name, ds.X, ds.y, "median")
    rows = []
    for mech, rate, seed in product(MECHANISMS, RATES, seeds):
        X_amp, mask = amputate(ds.X, mech, rate, columns=cols, driver=DRIVER, seed=seed)
        realised = float(mask[cols].to_numpy().mean())
        for strat in SIM_STRATEGIES:
            nrmse = reconstruction_nrmse(ds.X, X_amp, mask, strat, seed)
            auc = downstream_auc(ds.name, X_amp, ds.y, strat)
            rows.append(
                {
                    "mechanism": mech,
                    "rate": rate,
                    "realised_rate": realised,
                    "seed": seed,
                    "strategy": strat,
                    "nrmse": nrmse,
                    "roc_auc": auc,
                    "auc_ceiling": ceiling,
                }
            )
    return pd.DataFrame(rows)
