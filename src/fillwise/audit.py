"""Missingness auditing.

Three questions a table should answer before anyone trains on it:

1. How much is missing, where, and how correlated are the gaps?
2. Are there disguised missing values (sentinel zeros in columns where zero
   is impossible)?
3. Is the missingness plausibly random (MCAR), or does it depend on other
   observed columns (MAR-like)? True MNAR cannot be confirmed from observed
   data alone, but a predictable gap pattern is enough to rule MCAR out.

The MCAR probe treats each column's missingness indicator as a label and
asks a cross-validated logistic regression to predict it from the other
columns (median/mode-filled inside the fold). AUC near 0.5 is consistent
with MCAR; well above it means the gaps carry signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Physiologically impossible zeros: column-name fragments worth flagging when
# a zero shows up. Deliberately conservative; the audit reports, humans decide.
SUSPECT_ZERO_HINTS = (
    "glucose",
    "pressure",
    "bmi",
    "insulin",
    "thickness",
    "temp",
    "pulse",
    "height",
    "weight",
    "age",
)


@dataclass
class ColumnAudit:
    column: str
    missing_rate: float
    suspect_zero_rate: float
    mcar_probe_auc: float | None


def missing_summary(X: pd.DataFrame) -> pd.DataFrame:
    rates = X.isna().mean()
    out = pd.DataFrame({"column": rates.index, "missing_rate": rates.to_numpy()})
    return out.sort_values("missing_rate", ascending=False).reset_index(drop=True)


def suspect_zeros(X: pd.DataFrame) -> dict[str, float]:
    """Rate of exact zeros in numeric columns whose name suggests zero is invalid."""
    flags: dict[str, float] = {}
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        name = col.lower()
        if any(h in name for h in SUSPECT_ZERO_HINTS):
            nonnull = s.dropna()
            if len(nonnull) == 0:
                continue
            rate = float((nonnull == 0).mean())
            if rate > 0:
                flags[col] = rate
    return flags


def mcar_probe(
    X: pd.DataFrame, column: str, seed: int = 0, min_missing: int = 15
) -> float | None:
    """Cross-validated AUC of predicting `column is missing` from the rest."""
    mask = X[column].isna().to_numpy().astype(int)
    if mask.sum() < min_missing or mask.sum() > len(mask) - min_missing:
        return None
    rest = X.drop(columns=[column])
    num = [c for c in rest.columns if pd.api.types.is_numeric_dtype(rest[c])]
    cat = [c for c in rest.columns if c not in num]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, rest, mask, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def audit_frame(X: pd.DataFrame, seed: int = 0) -> list[ColumnAudit]:
    zero_flags = suspect_zeros(X)
    audits: list[ColumnAudit] = []
    for col in X.columns:
        rate = float(X[col].isna().mean())
        probe = mcar_probe(X, col, seed=seed) if rate > 0 else None
        audits.append(
            ColumnAudit(
                column=col,
                missing_rate=rate,
                suspect_zero_rate=zero_flags.get(col, 0.0),
                mcar_probe_auc=probe,
            )
        )
    return audits


def audit_report(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rows = [
        {
            "column": a.column,
            "missing_rate": round(a.missing_rate, 4),
            "suspect_zero_rate": round(a.suspect_zero_rate, 4),
            "mcar_probe_auc": None if a.mcar_probe_auc is None else round(a.mcar_probe_auc, 3),
        }
        for a in audit_frame(X, seed=seed)
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["missing_rate", "suspect_zero_rate"], ascending=False
    ).reset_index(drop=True)
