"""Figures for the README and reports. Matplotlib only, no seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "complete_case": "#b3b3b3",
    "mean": "#8da0cb",
    "median": "#66c2a5",
    "median_flag": "#1b9e77",
    "knn": "#fc8d62",
    "iterative": "#d95f02",
    "native_nan": "#7570b3",
}


def missingness_matrix(X: pd.DataFrame, out: Path, title: str) -> None:
    order = X.isna().mean().sort_values(ascending=False).index
    mat = X[order].isna().to_numpy().T
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(order)), labels=order, fontsize=8)
    ax.set_xlabel("row")
    rates = X[order].isna().mean()
    for i, c in enumerate(order):
        ax.text(mat.shape[1] + 3, i, f"{rates[c]:.0%}", va="center", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def benchmark_bars(df: pd.DataFrame, out: Path) -> None:
    combos = [
        ("colic", "logreg"), ("colic", "lgbm"),
        ("pima", "logreg"), ("pima", "lgbm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, (dsname, model) in zip(axes.ravel(), combos, strict=True):
        sub = df[(df.dataset == dsname) & (df.model == model)]
        stats = sub.groupby("strategy")["roc_auc"].agg(["mean", "std"]).dropna()
        stats = stats.sort_values("mean")
        colors = [PALETTE.get(s, "#333") for s in stats.index]
        ax.barh(stats.index, stats["mean"], xerr=stats["std"], color=colors, height=0.6)
        lo = max(0.5, stats["mean"].min() - 0.05)
        ax.set_xlim(lo, min(1.0, stats["mean"].max() + 0.03))
        ax.set_title(f"{dsname} / {model}")
        ax.set_xlabel("ROC-AUC (15-fold mean, error bar = sd)")
        for i, (m, _s) in enumerate(stats.itertuples(index=False)):
            ax.text(m + 0.002, i, f"{m:.3f}", va="center", fontsize=8)
    fig.suptitle("Imputation strategy vs classifier performance")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def amputation_heatmap(sim: pd.DataFrame, out: Path) -> None:
    metrics = [("nrmse", "Reconstruction NRMSE (lower better)", "viridis_r"),
               ("roc_auc", "Downstream ROC-AUC (higher better)", "viridis")]
    mechs = ["mcar", "mar", "mnar"]
    strategies = list(sim.strategy.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, (metric, label, cmap) in enumerate(metrics):
        vals = sim.groupby(["mechanism", "rate", "strategy"])[metric].mean().reset_index()
        vmin, vmax = vals[metric].min(), vals[metric].max()
        for c, mech in enumerate(mechs):
            ax = axes[r][c]
            sub = vals[vals.mechanism == mech].pivot(
                index="strategy", columns="rate", values=metric
            ).reindex(strategies)
            im = ax.imshow(sub.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(sub.columns)), labels=[f"{x:.0%}" for x in sub.columns])
            ax.set_yticks(range(len(sub.index)), labels=sub.index, fontsize=8)
            ax.set_title(f"{mech.upper()}" + ("" if r else " — missing rate →"))
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub.iloc[i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if (v - vmin) / (vmax - vmin + 1e-9) < 0.5 else "black")
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
        fig.colorbar(im, ax=axes[r], shrink=0.8)
    fig.suptitle("Amputation study on a fully observed table (3-seed means)")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pima_repair_chart(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, means, errs, colors = [], [], [], []
    for model in ("logreg", "lgbm"):
        for dsname, tag, color in (
            ("pima_raw", "zeros kept", "#c44e52"),
            ("pima", "repaired + median_flag", "#1b9e77"),
        ):
            strat = "median" if dsname == "pima_raw" else "median_flag"
            sub = df[(df.dataset == dsname) & (df.model == model) & (df.strategy == strat)]
            labels.append(f"{model}\n{tag}")
            means.append(sub.roc_auc.mean())
            errs.append(sub.roc_auc.std())
            colors.append(color)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=colors, width=0.6)
    ax.set_xticks(x, labels=labels, fontsize=9)
    ax.set_ylim(min(means) - 0.04, max(means) + 0.02)
    for i, m in enumerate(means):
        ax.text(i, m + 0.003, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Disguised zeros: leave them in vs repair-then-impute")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
