import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fillwise.data import load_colic, load_pima
from fillwise.strategies import STRATEGIES, build_lgbm, build_logreg, drop_incomplete

COLIC = load_colic()
PIMA = load_pima()


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "complete_case"])
def test_lgbm_pipelines_fit_and_score(strategy):
    pipe = build_lgbm(strategy, COLIC.numeric, COLIC.categorical)
    X_tr, X_te, y_tr, _ = train_test_split(
        COLIC.X, COLIC.y, test_size=0.3, random_state=0, stratify=COLIC.y
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert np.isfinite(proba).all()
    assert proba.min() >= 0 and proba.max() <= 1


@pytest.mark.parametrize(
    "strategy", [s for s in STRATEGIES if s not in ("complete_case", "native_nan")]
)
def test_logreg_pipelines_fit_and_score(strategy):
    pipe = build_logreg(strategy, PIMA.numeric, PIMA.categorical)
    X_tr, X_te, y_tr, _ = train_test_split(
        PIMA.X, PIMA.y, test_size=0.3, random_state=0, stratify=PIMA.y
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert np.isfinite(proba).all()


def test_logreg_rejects_native_nan():
    with pytest.raises(ValueError):
        build_logreg("native_nan", PIMA.numeric, [])


def test_imputer_statistics_come_from_training_fold_only():
    """The median the pipeline uses must be the train median, not the full-data one."""
    X_tr, X_te, y_tr, _ = train_test_split(PIMA.X, PIMA.y, test_size=0.5, random_state=1)
    pipe = build_logreg("median", PIMA.numeric, [])
    pipe.fit(X_tr, y_tr)
    imp = pipe.named_steps["pre"].named_transformers_["num"].named_steps["imp"]
    fitted = np.asarray(imp.statistics_)
    train_medians = X_tr[PIMA.numeric].median().to_numpy()
    full_medians = PIMA.X[PIMA.numeric].median().to_numpy()
    assert np.allclose(fitted, train_medians, equal_nan=True)
    # and they genuinely differ from full-data leakage values somewhere
    assert not np.allclose(train_medians, full_medians)


def test_drop_incomplete():
    X, y = drop_incomplete(COLIC.X, COLIC.y)
    assert len(X) == len(y) == COLIC.complete_rows
    assert X.notna().all().all()


def test_median_flag_adds_indicator_width():
    pipe_plain = build_logreg("median", PIMA.numeric, [])
    pipe_flag = build_logreg("median_flag", PIMA.numeric, [])
    pipe_plain.fit(PIMA.X, PIMA.y)
    pipe_flag.fit(PIMA.X, PIMA.y)
    w_plain = pipe_plain.named_steps["clf"].coef_.shape[1]
    w_flag = pipe_flag.named_steps["clf"].coef_.shape[1]
    assert w_flag > w_plain
