"""Plotting helpers with a single consistent visual style.

Every function returns the Matplotlib ``Axes``/``Figure`` so notebooks can adjust
titles, and :func:`save` writes publication-ready PNG files into ``outputs/figures``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import FIGURE_DIR

__all__ = [
    "set_style",
    "save",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_calibration",
    "plot_decision_curve",
    "plot_metrics_by_landmark",
    "plot_feature_importance",
    "plot_ae_grade_distribution",
    "plot_toxicity_over_time",
    "plot_risk_trajectories",
    "plot_missingness",
]

PALETTE = ["#2f6f9f", "#c0504d", "#4f8a5b", "#8064a2", "#e0913a", "#4bacc6", "#7f7f7f"]


def set_style() -> None:
    """Apply the report-wide Matplotlib/Seaborn style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
        "grid.alpha": 0.3,
    })


def save(fig, name: str, directory: Path | None = None) -> Path:
    """Persist a figure as PNG under ``outputs/figures``."""
    directory = Path(directory) if directory is not None else FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------------------
def plot_roc_curves(y, predictions: dict[str, np.ndarray], title: str = "ROC curves",
                    ax=None):
    from sklearn.metrics import roc_auc_score
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.8))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_pr_curves(y, predictions: dict[str, np.ndarray],
                   title: str = "Precision-recall curves", ax=None):
    from sklearn.metrics import average_precision_score
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.8))
    for name, p in predictions.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=1.8,
                label=f"{name} (AP={average_precision_score(y, p):.3f})")
    base = float(np.mean(y))
    ax.axhline(base, ls="--", c="k", lw=1, alpha=0.5, label=f"Prevalence ({base:.2f})")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_calibration(calib_tables: dict[str, pd.DataFrame],
                     title: str = "Calibration", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.8))
    for name, t in calib_tables.items():
        ax.errorbar(t["predicted"], t["observed"], yerr=t["se"], marker="o",
                    ms=4, lw=1.4, capsize=2, label=name)
    lim = max(0.05, float(max(t["predicted"].max() for t in calib_tables.values())) * 1.15)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.set(xlabel="Predicted probability", ylabel="Observed event rate", title=title,
           xlim=(0, lim), ylim=(0, lim))
    ax.legend(fontsize=8)
    return ax


def plot_decision_curve(dc: pd.DataFrame, title: str = "Decision curve analysis", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.6))
    for strategy, sub in dc.groupby("strategy"):
        style = "--" if strategy in ("Treat all", "Treat none") else "-"
        color = "0.45" if strategy in ("Treat all", "Treat none") else None
        ax.plot(sub["threshold"], sub["net_benefit"], style, color=color,
                lw=1.8, label=strategy)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title=title)
    ax.set_ylim(bottom=min(-0.02, dc["net_benefit"].min()))
    ax.legend(fontsize=8)
    return ax


def plot_metrics_by_landmark(tbl: pd.DataFrame, metric: str = "auroc",
                             title: str | None = None, ax=None):
    """Discrimination and risk-set size as the trial progresses."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(tbl["landmark_day"], tbl[metric], marker="o", lw=1.8, color=PALETTE[0])
    ax.axhline(0.5, ls="--", c="k", lw=1, alpha=0.5)
    ax.set(xlabel="Landmark day (study day)", ylabel=metric.upper(),
           title=title or f"{metric.upper()} by landmark")
    ax2 = ax.twinx()
    ax2.bar(tbl["landmark_day"], tbl["n_patients"], width=8, alpha=0.18,
            color=PALETTE[1], label="patients at risk")
    ax2.set_ylabel("Patients at risk")
    ax2.grid(False)
    return ax


def plot_feature_importance(importance: pd.Series, top_n: int = 20,
                            title: str = "Feature importance", ax=None):
    imp = importance.dropna().sort_values(ascending=False).head(top_n).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index.astype(str), imp.to_numpy(), color=PALETTE[0])
    ax.set(xlabel="Importance", title=title)
    ax.tick_params(axis="y", labelsize=8)
    return ax


# --------------------------------------------------------------------------------------
# Clinical / exploratory
# --------------------------------------------------------------------------------------
def plot_ae_grade_distribution(ae: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.8))
    counts = ae["grade"].value_counts().sort_index()
    ax.bar(counts.index.astype(int).astype(str), counts.to_numpy(), color=PALETTE[0])
    for x, v in zip(range(len(counts)), counts.to_numpy()):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="CTCAE grade", ylabel="Number of adverse events",
           title="Adverse events by CTCAE grade")
    return ax


def plot_toxicity_over_time(ae: pd.DataFrame, timeline: pd.DataFrame,
                            bin_days: int = 14, max_day: int = 336, ax=None):
    """Toxicity-over-time view: severe-AE incidence per 100 patient-months at risk.

    This is the ToxT-style descriptive analysis that motivates the temporal model -
    a maximum-grade table cannot show how the hazard moves during treatment.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.2))

    edges = np.arange(0, max_day + bin_days, bin_days)
    sev = ae[ae["is_severe"] & (ae["start_day"] > 0)]

    n_at_risk, rate, centres = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = (timeline["on_treatment_end"] >= lo).sum()
        n_events = ((sev["start_day"] > lo) & (sev["start_day"] <= hi)).sum()
        person_months = at_risk * bin_days / 30.4375
        n_at_risk.append(at_risk)
        rate.append(100 * n_events / person_months if person_months else np.nan)
        centres.append((lo + hi) / 2)

    ax.plot(centres, rate, marker="o", lw=1.8, color=PALETTE[1])
    ax.set(xlabel="Study day", ylabel="Severe AEs per 100 patient-months",
           title="Severe (Grade $\\geq$3) adverse-event incidence over time")
    ax2 = ax.twinx()
    ax2.fill_between(centres, n_at_risk, alpha=0.12, color=PALETTE[0])
    ax2.set_ylabel("Patients on treatment")
    ax2.grid(False)
    return ax


def plot_risk_trajectories(df: pd.DataFrame, pred_col: str, n_patients: int = 12,
                           seed: int = 42, ax=None):
    """Individual predicted-risk trajectories, highlighting realised events.

    This is the clinical deliverable of the temporal model: a per-patient risk curve
    that updates at every landmark rather than a single fixed baseline score.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(seed)

    with_event = df[df["y"] == 1]["PID"].unique()
    without = np.setdiff1d(df["PID"].unique(), with_event)
    pick = np.concatenate([
        rng.choice(with_event, min(n_patients // 2, len(with_event)), replace=False),
        rng.choice(without, min(n_patients // 2, len(without)), replace=False),
    ])

    for pid in pick:
        sub = df[df["PID"] == pid].sort_values("landmark_day")
        had_event = sub["y"].max() == 1
        ax.plot(sub["landmark_day"], sub[pred_col], lw=1.4, alpha=0.85,
                color=PALETTE[1] if had_event else PALETTE[0])
        ev = sub[sub["y"] == 1]
        ax.scatter(ev["landmark_day"], ev[pred_col], color=PALETTE[1], s=26, zorder=3)

    ax.set(xlabel="Landmark day", ylabel="Predicted 30-day severe-AE risk",
           title="Individual in-trial risk trajectories")
    handles = [plt.Line2D([], [], color=PALETTE[1], lw=2,
                          label="Patient with a severe AE in some window"),
               plt.Line2D([], [], color=PALETTE[0], lw=2, label="Patient without")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    return ax


def plot_missingness(df: pd.DataFrame, top_n: int = 25, ax=None):
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    miss = miss[miss > 0].iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.3 * len(miss) + 1.2))
    ax.barh(miss.index.astype(str), miss.to_numpy(), color=PALETTE[4])
    ax.set(xlabel="% missing", title=f"Missingness (top {top_n} columns)")
    ax.tick_params(axis="y", labelsize=8)
    return ax
