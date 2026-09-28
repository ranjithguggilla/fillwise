import numpy as np

from fillwise.benchmark import results_frame, run_strategy
from fillwise.data import load_colic, load_pima
from fillwise.simulate import run_amputation_study


def test_run_strategy_returns_valid_folds():
    ds = load_pima()
    res = run_strategy(ds, "logreg", "median", n_splits=3, n_repeats=1)
    fr = results_frame(res)
    assert len(fr) == 3
    assert fr.roc_auc.between(0.5, 1.0).all()
    assert fr.pr_auc.between(0.0, 1.0).all()
    assert (fr.train_rows > 400).all()


def test_complete_case_infeasible_on_colic():
    ds = load_colic()
    res = run_strategy(ds, "logreg", "complete_case", n_splits=3, n_repeats=1)
    fr = results_frame(res)
    assert fr.roc_auc.isna().all()  # 6 complete rows cannot support a fold
    assert (fr.train_rows < 30).all()


def test_native_nan_skipped_for_logreg():
    ds = load_pima()
    assert run_strategy(ds, "logreg", "native_nan") == []


def test_amputation_study_smoke():
    df = run_amputation_study(seeds=(0,))
    assert set(df.mechanism) == {"mcar", "mar", "mnar"}
    assert df.nrmse.notna().all()
    assert df.roc_auc.between(0.5, 1.0).all()
    # MNAR at high rates should reconstruct worse than MCAR on average
    mnar = df.query("mechanism == 'mnar' and rate == 0.5").nrmse.mean()
    mcar = df.query("mechanism == 'mcar' and rate == 0.5").nrmse.mean()
    assert mnar > mcar


def test_app_audit_produces_recommendations():
    from fillwise.app import run_audit

    rep, chart = run_audit(None, "pima (raw, zeros kept)")
    assert "recommendation" in rep.columns
    assert (rep.suspect_zero_rate > 0).any()
    assert chart.endswith(".png")
    zero_rows = rep[rep.column == "insulin"]
    assert "sentinel" in zero_rows.recommendation.iloc[0]


def test_seeded_benchmark_reproducibility():
    ds = load_pima()
    a = results_frame(run_strategy(ds, "logreg", "median", n_splits=3, n_repeats=1))
    b = results_frame(run_strategy(ds, "logreg", "median", n_splits=3, n_repeats=1))
    assert np.allclose(a.roc_auc, b.roc_auc)
