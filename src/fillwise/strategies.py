"""Imputation strategies as leakage-safe scikit-learn pipelines.

Every strategy is a full Pipeline whose imputers are fitted inside the CV
fold, on training rows only. Categorical columns are handled identically
across strategies (mode fill + one-hot for linear models, ordinal codes with
NaN passthrough for LightGBM) so the comparison isolates the numeric
imputation choice.

Strategies
----------
complete_case   drop any row with a missing value (the "clean data only" myth)
mean            SimpleImputer(mean)
median          SimpleImputer(median)
median_flag     median + missingness indicator columns
knn             KNNImputer(k=5) on standardised numerics
iterative       IterativeImputer (BayesianRidge round-robin, MICE-style)
native_nan      no numeric imputation at all; LightGBM routes NaNs per split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer

# IterativeImputer still lives behind the experimental flag.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

STRATEGIES = ["complete_case", "mean", "median", "median_flag", "knn", "iterative", "native_nan"]

SEED = 42


def _numeric_imputer(strategy: str, seed: int):
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "median_flag":
        return SimpleImputer(strategy="median", add_indicator=True)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    if strategy == "iterative":
        return IterativeImputer(max_iter=15, random_state=seed, sample_posterior=False)
    raise ValueError(f"no numeric imputer for strategy {strategy}")


def build_logreg(
    strategy: str, numeric: list[str], categorical: list[str], seed: int = SEED
) -> Pipeline:
    """Linear model. Needs complete inputs, so native_nan does not apply."""
    if strategy == "native_nan":
        raise ValueError("native_nan only applies to models that accept NaN")
    if strategy == "complete_case":
        num_steps = [("sc", StandardScaler())]
    else:
        num_steps = [("imp", _numeric_imputer(strategy, seed)), ("sc", StandardScaler())]
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_lgbm(
    strategy: str, numeric: list[str], categorical: list[str], seed: int = SEED
) -> Pipeline:
    if strategy in {"complete_case", "native_nan"}:
        num: list[tuple] = [("num", "passthrough", numeric)]
    else:
        num = [("num", _numeric_imputer(strategy, seed), numeric)]
    transformers = list(num)
    if categorical:
        # Ordinal codes; unseen/missing become NaN which LightGBM handles natively.
        cat_pipe = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
        )
        transformers.append(("cat", cat_pipe, categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def drop_incomplete(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Listwise deletion for the complete_case strategy."""
    keep = X.notna().all(axis=1).to_numpy()
    return X.loc[keep], y[keep]
