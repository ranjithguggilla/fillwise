"""End-to-end experiment runner.

Writes results/benchmark.csv, results/amputation.csv, results/metrics.json
and the four charts under assets/. Every model fit is logged to MLflow
(sqlite backend in the repo root).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fillwise.audit import audit_report  # noqa: E402
from fillwise.benchmark import MODELS, paired_delta, results_frame, run_strategy  # noqa: E402
from fillwise.data import load_colic, load_pima  # noqa: E402
from fillwise.plots import (  # noqa: E402
    amputation_heatmap,
    benchmark_bars,
    missingness_matrix,
    pima_repair_chart,
)
from fillwise.simulate import run_amputation_study  # noqa: E402
from fillwise.strategies import SEED, STRATEGIES  # noqa: E402

RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"


def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fillwise")

    colic = load_colic()
    pima = load_pima(repair_zeros=True)
    pima_raw = load_pima(repair_zeros=False)
    print(f"colic: {colic.n_rows} rows, complete rows: {colic.complete_rows}")
    print(f"pima : {pima.n_rows} rows, complete rows: {pima.complete_rows}")

    # ---- audits -----------------------------------------------------------
    audits = {}
    for ds in (colic, pima):
        rep = audit_report(ds.X, seed=SEED)
        rep.to_csv(RESULTS / f"audit_{ds.name}.csv", index=False)
        audits[ds.name] = rep
        print(f"\naudit {ds.name}:\n{rep.head(8)}")
    raw_rep = audit_report(pima_raw.X, seed=SEED)
    raw_rep.to_csv(RESULTS / "audit_pima_raw.csv", index=False)

    # ---- real-data benchmark ---------------------------------------------
    all_results = []
    for ds in (colic, pima):
        for model in MODELS:
            for strat in STRATEGIES:
                res = run_strategy(ds, model, strat)
                if not res:
                    continue
                all_results.extend(res)
                fr = results_frame(res)
                with mlflow.start_run(run_name=f"{ds.name}-{model}-{strat}"):
                    mlflow.log_params(
                        {"dataset": ds.name, "model": model, "strategy": strat,
                         "cv": "5x3 stratified", "seed": SEED}
                    )
                    for m in ("roc_auc", "pr_auc", "f1", "brier"):
                        if fr[m].notna().any():
                            mlflow.log_metric(m, fr[m].mean())
                    mlflow.log_metric("train_rows_mean", fr.train_rows.mean())
                auc = fr.roc_auc.mean()
                print(f"{ds.name:5s} {model:6s} {strat:13s} AUC={auc:.4f} "
                      f"rows={fr.train_rows.mean():.0f}")
    # zeros-kept pima for the repair comparison (median imputer is a no-op)
    for model in MODELS:
        res = run_strategy(pima_raw, model, "median")
        all_results.extend(res)
        fr = results_frame(res)
        with mlflow.start_run(run_name=f"pima_raw-{model}-asis"):
            mlflow.log_params({"dataset": "pima_raw", "model": model, "strategy": "as-is"})
            mlflow.log_metric("roc_auc", fr.roc_auc.mean())
        print(f"pima_raw {model:6s} as-is AUC={fr.roc_auc.mean():.4f}")

    bench = results_frame(all_results)
    bench.to_csv(RESULTS / "benchmark.csv", index=False)

    # ---- paired comparisons ----------------------------------------------
    deltas = []
    for dsname in ("colic", "pima"):
        for model in MODELS:
            for strat in STRATEGIES:
                if strat == "median" or (model == "logreg" and strat == "native_nan"):
                    continue
                if bench.query(
                    "dataset == @dsname and model == @model and strategy == @strat"
                ).empty:
                    continue
                d = paired_delta(bench, dsname, model, strat)
                d.update({"dataset": dsname, "model": model, "strategy": strat})
                deltas.append(d)
    import pandas as pd

    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(RESULTS / "paired_deltas.csv", index=False)

    # ---- amputation study -------------------------------------------------
    print("\nrunning amputation study…")
    sim = run_amputation_study()
    sim.to_csv(RESULTS / "amputation.csv", index=False)
    with mlflow.start_run(run_name="amputation-study"):
        mlflow.log_param("mechanisms", "mcar,mar,mnar")
        mlflow.log_param("rates", "0.1,0.3,0.5")
        for (mech, strat), g in sim.groupby(["mechanism", "strategy"]):
            mlflow.log_metric(f"nrmse_{mech}_{strat}", g.nrmse.mean())

    # ---- charts -----------------------------------------------------------
    missingness_matrix(colic.X, ASSETS / "missingness_colic.png",
                       "Horse colic: white = observed, black = missing")
    benchmark_bars(bench, ASSETS / "benchmark_auc.png")
    amputation_heatmap(sim, ASSETS / "amputation_heatmap.png")
    pima_repair_chart(bench, ASSETS / "pima_repair.png")

    # ---- headline metrics --------------------------------------------------
    def mean_auc(ds, model, strat):
        sub = bench.query("dataset == @ds and model == @model and strategy == @strat")
        return None if sub.empty else round(float(sub.roc_auc.mean()), 4)

    cc = bench.query("dataset == 'colic' and strategy == 'complete_case'")
    metrics = {
        "colic_rows": colic.n_rows,
        "colic_complete_rows": colic.complete_rows,
        "colic_missing_cell_rate": round(float(colic.X.isna().mean().mean()), 4),
        "pima_disguised_zero_rows": int((load_pima(repair_zeros=False).X[
            ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
        ] == 0).any(axis=1).sum()),
        "best": {},
        "amputation_mnar_penalty_nrmse": round(float(
            sim.query("mechanism=='mnar' and rate==0.5").nrmse.mean()
            - sim.query("mechanism=='mcar' and rate==0.5").nrmse.mean()), 4),
        "complete_case_train_rows_colic": (
            None if cc.empty else round(float(cc.train_rows.mean()), 1)
        ),
        "runtime_seconds": None,
    }
    for ds in ("colic", "pima", "pima_raw"):
        for model in MODELS:
            for strat in (STRATEGIES if ds != "pima_raw" else ["median"]):
                v = mean_auc(ds, model, strat)
                if v is not None:
                    metrics["best"][f"{ds}.{model}.{strat}"] = v
    metrics["runtime_seconds"] = round(time.time() - t0, 1)

    def denan(obj):
        if isinstance(obj, dict):
            return {k: denan(v) for k, v in obj.items()}
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    (RESULTS / "metrics.json").write_text(json.dumps(denan(metrics), indent=2))
    print(f"\ndone in {metrics['runtime_seconds']}s")


if __name__ == "__main__":
    main()
