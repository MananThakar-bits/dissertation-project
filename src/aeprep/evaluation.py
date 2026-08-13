"""Evaluation: discrimination, calibration, clinical utility and temporal breakdown.

Reporting discrimination alone would repeat the weakness that Osterman et al.
criticise in oncology risk models, so every model is also assessed for
*calibration* (are the probabilities honest?) and *net benefit* (would acting on
them help?), and the temporal model is additionally broken down by landmark to
show whether risk updating actually improves over the static baseline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import CFG

__all__ = [
    "binary_metrics",
    "metrics_table",
    "bootstrap_metric_ci",
    "threshold_table",
    "operating_point",
    "enrolment_screening_table",
    "sensitivity_threshold",
    "calibration_table",
    "decision_curve",
    "metrics_by_landmark",
    "compare_static_vs_temporal",
    "delong_roc_test",
]


# --------------------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------------------
def _calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox calibration: regress the outcome on the logit of the predicted risk.

    A perfectly calibrated model has intercept 0 and slope 1; slope < 1 signals
    over-fitted (too extreme) predictions.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    """Discrimination + calibration summary for one set of predictions."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]

    out: dict[str, float] = {"n": len(y), "n_events": int(y.sum()),
                             "event_rate": float(y.mean()) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        out.update({k: np.nan for k in
                    ["auroc", "auprc", "brier", "log_loss", "calib_intercept",
                     "calib_slope", "scaled_brier"]})
        return out

    prevalence = y.mean()
    out["auroc"] = roc_auc_score(y, p)
    out["auprc"] = average_precision_score(y, p)
    out["auprc_lift"] = out["auprc"] / prevalence
    out["brier"] = brier_score_loss(y, p)
    out["scaled_brier"] = 1.0 - out["brier"] / (prevalence * (1 - prevalence))
    out["log_loss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    ci, cs = _calibration_intercept_slope(y, p)
    out["calib_intercept"], out["calib_slope"] = ci, cs

    if threshold is not None:
        out.update(operating_point(y, p, threshold))
    return out


def metrics_table(predictions: dict[str, np.ndarray], y: np.ndarray,
                  sort_by: str = "auroc") -> pd.DataFrame:
    """Stack :func:`binary_metrics` over a ``{model_name: probabilities}`` mapping."""
    rows = []
    for name, p in predictions.items():
        m = binary_metrics(y, p)
        m["model"] = name
        rows.append(m)
    cols = ["model", "n", "n_events", "event_rate", "auroc", "auprc", "auprc_lift",
            "brier", "scaled_brier", "log_loss", "calib_intercept", "calib_slope"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------
def bootstrap_metric_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray | None = None,
                        metric: str = "auroc", cfg=CFG,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI, resampling *patients* rather than rows.

    Resampling at the patient level is essential in the landmark design: rows from
    the same patient are correlated, so a naive row bootstrap would understate the
    uncertainty.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    fn = {"auroc": roc_auc_score,
          "auprc": average_precision_score,
          "brier": brier_score_loss}[metric]

    point = fn(y, p)
    rng = np.random.default_rng(cfg.random_state)

    if groups is None:
        idx_pool = [np.array([i]) for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]

    stats = []
    for _ in range(cfg.n_bootstrap):
        pick = rng.integers(0, len(idx_pool), len(idx_pool))
        idx = np.concatenate([idx_pool[i] for i in pick])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        stats.append(fn(yy, pp))
    if not stats:
        return point, np.nan, np.nan
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def delong_roc_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                    groups: np.ndarray | None = None, cfg=CFG) -> dict:
    """Bootstrap test for the AUROC difference between two models on the same data.

    A patient-level bootstrap of the *paired* difference is used rather than the
    parametric DeLong variance, because landmark rows are clustered within patients.
    """
    y = np.asarray(y).astype(int)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    observed = roc_auc_score(y, p1) - roc_auc_score(y, p2)

    rng = np.random.default_rng(cfg.random_state)
    if groups is None:
        idx_pool = [np.array([i]) for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]

    diffs = []
    for _ in range(cfg.n_bootstrap):
        pick = rng.integers(0, len(idx_pool), len(idx_pool))
        idx = np.concatenate([idx_pool[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value for H0: difference = 0.
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_difference": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p_value, 1.0))}


# --------------------------------------------------------------------------------------
# Operating points and clinical utility
# --------------------------------------------------------------------------------------
def operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix derived metrics at a fixed probability threshold."""
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens) else np.nan
    return {"threshold": threshold, "tp": int(tp), "fp": int(fp), "tn": int(tn),
            "fn": int(fn), "sensitivity": sens, "specificity": spec, "ppv": ppv,
            "npv": npv, "f1": f1, "balanced_accuracy": (sens + spec) / 2,
            "alert_rate": pred.mean()}


def threshold_table(y: np.ndarray, p: np.ndarray,
                    thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Sensitivity/specificity/PPV/NPV across candidate alerting thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.95, 0.05), 2)
    return pd.DataFrame([operating_point(y, p, t) for t in thresholds])


def enrolment_screening_table(y: np.ndarray, p: np.ndarray,
                              thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Cost of using a *baseline* risk score to exclude patients at enrolment.

    Answers a question an evaluator is entitled to ask of any pre-treatment risk
    model: could it be used to keep a patient out of the trial? The framing differs
    from :func:`threshold_table` because the consequence of a positive call is not an
    extra blood test but denial of second-line therapy, so the two error types are not
    remotely symmetric and must be counted separately.

    Columns
    -------
    ``n_excluded``            patients the rule would refuse
    ``events_averted``        excluded patients who would in fact have had a severe event
    ``events_missed``         severe events occurring among patients still enrolled
    ``wrongly_excluded``      excluded patients who would *not* have had an event -
                              each one denied treatment for nothing
    ``nne``                   number needed to exclude to avert a single severe event
    ``harm_ratio``            wrongly excluded per event averted
    ``pct_cohort_excluded``   share of the cohort refused entry

    A screening rule is only defensible if ``nne`` and ``harm_ratio`` are small; a
    model whose discrimination is close to chance produces neither.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.85, 0.05), 2)

    rows = []
    for t in thresholds:
        excl = p >= t
        n_excl = int(excl.sum())
        averted = int(y[excl].sum())
        missed = int(y[~excl].sum())
        wrong = n_excl - averted
        rows.append({
            "threshold": float(t),
            "n_excluded": n_excl,
            "pct_cohort_excluded": n_excl / len(y),
            "events_averted": averted,
            "events_missed": missed,
            "wrongly_excluded": wrong,
            "nne": n_excl / averted if averted else np.nan,
            "harm_ratio": wrong / averted if averted else np.nan,
        })
    return pd.DataFrame(rows)


def sensitivity_threshold(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    """Lowest threshold achieving at least ``target`` sensitivity."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.0


def decision_curve(y: np.ndarray, predictions: dict[str, np.ndarray],
                   thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Net benefit across threshold probabilities (Vickers & Elkin decision curve).

    Net benefit = TP/n - FP/n * pt/(1-pt); the 'treat all' and 'treat none'
    strategies are included as reference lines.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.62, 0.02), 3)

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        rows.append({"threshold": pt, "strategy": "Treat all",
                     "net_benefit": y.mean() - (1 - y.mean()) * w})
        rows.append({"threshold": pt, "strategy": "Treat none", "net_benefit": 0.0})
        for name, p in predictions.items():
            pred = np.asarray(p) >= pt
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            rows.append({"threshold": pt, "strategy": name,
                         "net_benefit": tp / n - (fp / n) * w})
    return pd.DataFrame(rows)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted event rate in equal-count risk deciles."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, float)})
    df = df.dropna()
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"),
                                observed=("y", "mean"))
    out["se"] = np.sqrt(out["observed"] * (1 - out["observed"]) / out["n"])
    return out.reset_index()


# --------------------------------------------------------------------------------------
# Temporal breakdown
# --------------------------------------------------------------------------------------
def metrics_by_landmark(df: pd.DataFrame, pred_col: str, y_col: str = "y",
                        landmark_col: str = "landmark_day",
                        min_events: int = 5) -> pd.DataFrame:
    """Discrimination recomputed inside each landmark stratum.

    Shows whether the model keeps working as the trial progresses and the risk-set
    composition changes (a time-dependent generalisation of AUROC in the spirit of
    Antolini's time-dependent discrimination index).
    """
    rows = []
    for lm, sub in df.groupby(landmark_col):
        y, p = sub[y_col].to_numpy(), sub[pred_col].to_numpy()
        row = {"landmark_day": lm, "n_rows": len(sub), "n_patients": sub["PID"].nunique(),
               "n_events": int(y.sum()), "event_rate": float(y.mean())}
        if int(y.sum()) >= min_events and len(np.unique(y)) > 1:
            row["auroc"] = roc_auc_score(y, p)
            row["auprc"] = average_precision_score(y, p)
            row["brier"] = brier_score_loss(y, p)
        else:
            row["auroc"] = row["auprc"] = row["brier"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("landmark_day").reset_index(drop=True)


def compare_static_vs_temporal(df: pd.DataFrame, static_col: str, temporal_col: str,
                               y_col: str = "y", group_col: str = "PID",
                               cfg=CFG) -> pd.DataFrame:
    """Head-to-head comparison on the identical set of landmark rows.

    The static model is applied unchanged at every landmark (its predictions never
    update), which is exactly the comparison the dissertation objectives call for:
    *static baseline models versus temporal models that update risk over time*.
    """
    y = df[y_col].to_numpy()
    groups = df[group_col].to_numpy()
    rows = []
    for name, col in [("Static baseline (frozen)", static_col),
                      ("Temporal landmark", temporal_col)]:
        p = df[col].to_numpy()
        auroc, lo, hi = bootstrap_metric_ci(y, p, groups, "auroc", cfg)
        auprc, plo, phi = bootstrap_metric_ci(y, p, groups, "auprc", cfg)
        m = binary_metrics(y, p)
        rows.append({"model": name,
                     "auroc": auroc, "auroc_ci": f"[{lo:.3f}, {hi:.3f}]",
                     "auprc": auprc, "auprc_ci": f"[{plo:.3f}, {phi:.3f}]",
                     "brier": m["brier"], "scaled_brier": m["scaled_brier"],
                     "calib_slope": m["calib_slope"]})
    out = pd.DataFrame(rows)

    test = delong_roc_test(y, df[temporal_col].to_numpy(), df[static_col].to_numpy(),
                           groups, cfg)
    out.attrs["auc_difference_test"] = test
    return out
