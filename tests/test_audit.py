import numpy as np
import pandas as pd

from fillwise.audit import audit_report, mcar_probe, suspect_zeros
from fillwise.data import load_pima

RNG = np.random.default_rng(0)


def test_suspect_zeros_found_in_raw_pima():
    ds = load_pima(repair_zeros=False)
    flags = suspect_zeros(ds.X)
    assert flags["insulin"] > 0.4
    assert flags["skin_thickness"] > 0.25
    assert "pedigree" not in flags


def test_suspect_zeros_ignores_plausible_zero_columns():
    df = pd.DataFrame({"pregnancies": [0, 1, 2, 0], "glucose": [0.0, 90.0, 100.0, 110.0]})
    flags = suspect_zeros(df)
    assert "pregnancies" not in flags
    assert flags["glucose"] == 0.25


def test_mcar_probe_near_half_for_random_gaps():
    n = 600
    df = pd.DataFrame(
        {
            "a": RNG.normal(size=n),
            "b": RNG.normal(size=n),
            "c": RNG.normal(size=n),
        }
    )
    df.loc[RNG.random(n) < 0.25, "a"] = np.nan
    auc = mcar_probe(df, "a", seed=0)
    assert auc is not None
    assert 0.38 < auc < 0.62


def test_mcar_probe_detects_mar_gaps():
    n = 600
    b = RNG.normal(size=n)
    df = pd.DataFrame({"a": RNG.normal(size=n), "b": b, "c": RNG.normal(size=n)})
    df.loc[b > np.quantile(b, 0.7), "a"] = np.nan  # gaps driven by observed b
    auc = mcar_probe(df, "a", seed=0)
    assert auc is not None and auc > 0.9


def test_probe_skips_columns_with_too_few_gaps():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0] * 3, "b": range(12)})
    assert mcar_probe(df, "a", min_missing=15) is None


def test_audit_report_shape_and_ordering():
    ds = load_pima(repair_zeros=True)
    rep = audit_report(ds.X, seed=0)
    assert list(rep.columns) == [
        "column",
        "missing_rate",
        "suspect_zero_rate",
        "mcar_probe_auc",
    ]
    assert len(rep) == ds.X.shape[1]
    assert rep.missing_rate.iloc[0] == rep.missing_rate.max()
