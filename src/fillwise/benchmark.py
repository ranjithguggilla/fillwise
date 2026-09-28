"""Strategy benchmark on real datasets with repeated stratified CV.

Complete-case is implemented the way it actually happens in the wild:
incomplete rows are discarded from the *training* fold, and any residual
NaN at scoring time falls back to training-fold medians (a deployed model
still has to return a prediction for every incoming row). Row retention is
reported so the cost of deletion is visible next to its score. If deletion
leaves fewer than 30 rows or a single class, the config is recorded as
infeasible instead of silently fitting garbage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import Dataset
from .strategies import SEED, build_lgbm, build_logreg, drop_incomplete

MODELS = ("logreg", "lgbm")


@dataclass
class FoldResult:
    dataset: str
    model: str
    strategy: str
    fold: int
    roc_auc: float
    pr_auc: float
    f1: float
    brier: float
    train_rows: int
    fit_seconds: float


def _build(model: str, strategy: str, ds: Dataset):
    if model == "logreg":
        return build_logreg(strategy, ds.numeric, ds.categorical)
    return build_lgbm(strategy, ds.numeric, ds.categorical)


def _median_backfill(train_X: pd.DataFrame, test_X: pd.DataFrame) -> pd.DataFrame:
    """Fill residual NaNs in test rows with training medians/modes (complete-case)."""
    out = test_X.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            fill = train_X[col].median()
        else:
            mode = train_X[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "missing"
        out[col] = out[col].fillna(fill)
    return out


def run_strategy(
    ds: Dataset,
    model: str,
    strategy: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED,
) -> list[FoldResult]:
    if model == "logreg" and strategy == "native_nan":
        return []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results: list[FoldResult] = []
    for fold, (tr, te) in enumerate(cv.split(ds.X, ds.y)):
        X_tr, y_tr = ds.X.iloc[tr], ds.y[tr]
        X_te, y_te = ds.X.iloc[te], ds.y[te]
        if strategy == "complete_case":
            X_tr, y_tr = drop_incomplete(X_tr, y_tr)
            if len(y_tr) < 30 or len(np.unique(y_tr)) < 2:
                results.append(
                    FoldResult(
                        ds.name, model, strategy, fold,
                        np.nan, np.nan, np.nan, np.nan, len(y_tr), 0.0,
                    )
                )
                continue
            X_te = _median_backfill(X_tr, X_te)
        t0 = time.perf_counter()
        pipe = _build(model, strategy, ds)
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        proba = pipe.predict_proba(X_te)[:, 1]
        results.append(
            FoldResult(
                dataset=ds.name,
                model=model,
                strategy=strategy,
                fold=fold,
                roc_auc=roc_auc_score(y_te, proba),
                pr_auc=average_precision_score(y_te, proba),
                f1=f1_score(y_te, (proba >= 0.5).astype(int)),
                brier=brier_score_loss(y_te, proba),
                train_rows=len(y_tr),
                fit_seconds=fit_s,
            )
        )
    return results


def paired_delta(
    df: pd.DataFrame, dataset: str, model: str, strategy: str, baseline: str = "median"
) -> dict:
    """Mean fold-paired ROC-AUC delta vs baseline, bootstrap CI and t-test p."""
    from scipy import stats

    a = df.query(
        "dataset == @dataset and model == @model and strategy == @strategy"
    ).sort_values("fold")["roc_auc"].to_numpy()
    b = df.query(
        "dataset == @dataset and model == @model and strategy == @baseline"
    ).sort_values("fold")["roc_auc"].to_numpy()
    nan = float("nan")
    if len(a) != len(b) or len(a) == 0 or np.isnan(a).any() or np.isnan(b).any():
        return {"delta": nan, "ci_lo": nan, "ci_hi": nan, "p": nan}
    d = a - b
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(2000)])
    t = stats.ttest_rel(a, b)
    return {
        "delta": float(d.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p": float(t.pvalue),
    }


def results_frame(results: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
