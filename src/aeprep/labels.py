"""Target-variable construction.

Primary target (as registered in the mid-semester report)

    *Occurrence of any treatment-emergent CTCAE Grade >= 3 adverse event within a
    forward prediction window of 30 days.*

Two operationalisations are provided:

``static_label``
    One row per patient. The window starts at the first dose of study treatment,
    i.e. "will this patient experience a severe AE during the first 30 days?".

``landmark_label``
    One row per (patient, landmark day *t*). The window is ``(t, t + horizon]``,
    i.e. "given everything observed up to day *t*, will a severe AE occur in the
    next 30 days?". This is the temporal / in-trial risk-updating formulation.

Censoring
---------
A landmark row is *informative* when either
  (a) a severe AE starts inside the window, or
  (b) the window is fully covered by the patient's adverse-event observation period.
Rows failing both conditions are administratively censored and dropped when
``CFG.require_complete_window`` is true; they are retained (labelled 0) otherwise
and flagged via ``window_complete`` so a sensitivity analysis can be run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG

__all__ = [
    "treatment_emergent_events",
    "severe_events",
    "static_label",
    "make_landmark_grid",
    "landmark_label",
    "time_to_first_severe_ae",
]


def treatment_emergent_events(ae: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Restrict adverse events to those emerging on/after the first dose.

    Events with a start day before the first administration of study treatment are
    pre-existing conditions; they are excluded from the target but are retained
    elsewhere as baseline features (see :mod:`aeprep.features_static`).
    """
    ref = timeline[["PID", "first_dose_day", "ae_obs_end"]]
    m = ae.merge(ref, on="PID", how="left")
    m["treatment_emergent"] = m["start_day"] >= m["first_dose_day"]
    return m


def severe_events(ae: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Treatment-emergent events at or above the severity threshold."""
    out = ae[(ae["grade"] >= cfg.severity_threshold)]
    if "treatment_emergent" in ae.columns:
        out = out[out["treatment_emergent"]]
    return out.copy()


# --------------------------------------------------------------------------------------
# Static (patient-level) target
# --------------------------------------------------------------------------------------
def static_label(ae: pd.DataFrame, timeline: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Patient-level target: severe AE within ``horizon`` days of the first dose.

    Returns one row per patient with

    ``y``                 1 if a severe treatment-emergent AE started in the window
    ``window_start``      first dose day (exclusive lower bound is ``>=``)
    ``window_end``        ``window_start + horizon``
    ``window_complete``   the AE observation window covers ``window_end``
    ``first_severe_day``  day of the earliest severe AE in the window (NaN if none)
    ``n_severe_in_window``
    """
    horizon = cfg.prediction_horizon_days
    tl = timeline[["PID", "first_dose_day", "ae_obs_end"]].copy()
    tl["window_start"] = tl["first_dose_day"]
    tl["window_end"] = tl["window_start"] + horizon

    te = treatment_emergent_events(ae, timeline)
    sev = severe_events(te, cfg)

    sev = sev.merge(tl[["PID", "window_start", "window_end"]], on="PID", how="left")
    in_win = sev[(sev["start_day"] >= sev["window_start"]) &
                 (sev["start_day"] <= sev["window_end"])]

    agg = in_win.groupby("PID").agg(
        n_severe_in_window=("start_day", "size"),
        first_severe_day=("start_day", "min"),
        max_grade_in_window=("grade", "max"),
    )

    out = tl.merge(agg, on="PID", how="left")
    out["n_severe_in_window"] = out["n_severe_in_window"].fillna(0).astype(int)
    out["y"] = (out["n_severe_in_window"] > 0).astype(int)
    out["window_complete"] = out["ae_obs_end"] >= out["window_end"]
    out["informative"] = out["y"].eq(1) | out["window_complete"]
    return out[["PID", "window_start", "window_end", "y", "n_severe_in_window",
                "first_severe_day", "max_grade_in_window", "window_complete",
                "informative"]]


# --------------------------------------------------------------------------------------
# Landmark (temporal) target
# --------------------------------------------------------------------------------------
def make_landmark_grid(timeline: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Cartesian product of patients and landmark days, restricted to the risk set.

    A landmark *t* is retained for a patient when
    ``first_dose_day - 1 <= t <= on_treatment_end``: the patient must still be
    receiving study treatment, because the clinical question is *in-trial* risk
    updating rather than post-treatment follow-up.
    """
    grid_days = np.arange(cfg.landmark_start_day,
                          cfg.landmark_max_day + cfg.landmark_step_days,
                          cfg.landmark_step_days)

    tl = timeline[["PID", "first_dose_day", "on_treatment_end", "ae_obs_end"]].copy()
    rows = tl.merge(pd.DataFrame({"landmark_day": grid_days}), how="cross")

    keep = (rows["landmark_day"] >= rows["first_dose_day"] - 1) & \
           (rows["landmark_day"] <= rows["on_treatment_end"])
    rows = rows[keep].copy()
    rows["time_on_study"] = rows["landmark_day"] - rows["first_dose_day"] + 1
    return rows.sort_values(["PID", "landmark_day"]).reset_index(drop=True)


def landmark_label(grid: pd.DataFrame, ae: pd.DataFrame, timeline: pd.DataFrame,
                   cfg=CFG) -> pd.DataFrame:
    """Attach the forward-window target to a landmark grid.

    The window is ``(landmark_day, landmark_day + horizon]`` - strictly forward, so
    an event starting exactly at the landmark belongs to the *history*, never to the
    label. This is what prevents target leakage in the temporal design.
    """
    horizon = cfg.prediction_horizon_days
    g = grid.copy()
    g["window_end"] = g["landmark_day"] + horizon

    te = treatment_emergent_events(ae, timeline)
    sev = severe_events(te, cfg)[["PID", "start_day", "grade"]].dropna(subset=["start_day"])

    # Vectorised interval join: for every landmark count the severe AEs whose start
    # day falls strictly after the landmark and no later than the window end.
    counts = np.zeros(len(g), dtype=int)
    first_day = np.full(len(g), np.nan)
    max_grade = np.full(len(g), np.nan)

    sev_by_pid = {pid: sdf[["start_day", "grade"]].to_numpy()
                  for pid, sdf in sev.groupby("PID", sort=False)}

    lm_pid = g["PID"].to_numpy()
    lm_day = g["landmark_day"].to_numpy(dtype=float)
    lm_end = g["window_end"].to_numpy(dtype=float)

    for i in range(len(g)):
        arr = sev_by_pid.get(lm_pid[i])
        if arr is None:
            continue
        days = arr[:, 0]
        mask = (days > lm_day[i]) & (days <= lm_end[i])
        if mask.any():
            counts[i] = int(mask.sum())
            first_day[i] = float(days[mask].min())
            max_grade[i] = float(arr[mask, 1].max())

    g["n_severe_in_window"] = counts
    g["first_severe_day"] = first_day
    g["max_grade_in_window"] = max_grade
    g["y"] = (counts > 0).astype(int)
    g["window_complete"] = g["ae_obs_end"] >= g["window_end"]
    g["informative"] = g["y"].eq(1) | g["window_complete"]

    if cfg.require_complete_window:
        g = g[g["informative"]].copy()

    return g.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Survival formulation (used by the time-to-event comparison)
# --------------------------------------------------------------------------------------
def time_to_first_severe_ae(ae: pd.DataFrame, timeline: pd.DataFrame,
                            cfg=CFG) -> pd.DataFrame:
    """Time from first dose to the first severe treatment-emergent AE.

    Patients without a severe event are censored at ``ae_obs_end``.
    """
    te = treatment_emergent_events(ae, timeline)
    sev = severe_events(te, cfg)
    first = sev.groupby("PID")["start_day"].min().rename("first_severe_day")

    out = timeline[["PID", "first_dose_day", "ae_obs_end"]].merge(
        first, on="PID", how="left")
    out["event"] = out["first_severe_day"].notna().astype(int)
    out["duration"] = np.where(
        out["event"].eq(1),
        out["first_severe_day"] - out["first_dose_day"],
        out["ae_obs_end"] - out["first_dose_day"],
    )
    # Guard against zero/negative durations produced by same-day events.
    out["duration"] = out["duration"].clip(lower=0.5)
    return out[["PID", "duration", "event", "first_severe_day"]]
