import numpy as np
import pandas as pd

from fillwise.data import (
    PIMA_DISGUISED_ZERO,
    load_breast_cancer_complete,
    load_colic,
    load_pima,
)


def test_colic_shape_and_target():
    ds = load_colic()
    assert ds.X.shape == (299, 21)
    assert set(np.unique(ds.y)) == {0, 1}
    assert 0.55 < ds.y.mean() < 0.65  # survival is the majority class


def test_colic_missingness_is_substantial():
    ds = load_colic()
    rate = ds.X.isna().mean().mean()
    assert 0.2 < rate < 0.35
    assert ds.complete_rows < 20  # listwise deletion is not viable here


def test_colic_leaky_columns_removed():
    ds = load_colic()
    for col in ("hospital_number", "surgical_lesion", "lesion_1", "cp_data", "outcome"):
        assert col not in ds.X.columns


def test_pima_zero_repair():
    raw = load_pima(repair_zeros=False)
    fixed = load_pima(repair_zeros=True)
    assert raw.X.isna().sum().sum() == 0
    for col in PIMA_DISGUISED_ZERO:
        n_zero = int((raw.X[col] == 0).sum())
        assert int(fixed.X[col].isna().sum()) == n_zero
        if n_zero:
            assert (fixed.X[col].dropna() > 0).all()


def test_pima_target_untouched_by_repair():
    raw = load_pima(repair_zeros=False)
    fixed = load_pima(repair_zeros=True)
    assert np.array_equal(raw.y, fixed.y)


def test_breast_cancer_is_complete():
    ds = load_breast_cancer_complete()
    assert ds.X.isna().sum().sum() == 0
    assert ds.X.shape == (569, 30)
    assert all(pd.api.types.is_numeric_dtype(ds.X[c]) for c in ds.X.columns)
