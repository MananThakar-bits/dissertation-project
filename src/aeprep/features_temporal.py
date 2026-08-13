"""Temporal (landmark) feature engineering - the core of the dissertation.

For every landmark day *t* the patient's entire recorded history **up to and
including day t** is condensed into a fixed-length feature vector. Nothing after
*t* is ever touched, which is what makes the resulting risk estimate a genuine
forward prediction rather than a retrospective description.

Feature families
----------------
=========================  ==========================================================
``time / exposure``        days on study, cycle number, days since last dose
``dose``                   cumulative dose, dose in the last 28 days, relative dose
                           intensity, dose-reduction flags (per FOLFIRI component)
``performance status``     latest ECOG, change from baseline, recent worst value
``vital signs``            latest value, change from baseline, rolling mean/min/max
                           and trend over the look-back window
``toxicity history``       cumulative and recent counts of AEs by grade, seriousness,
                           relatedness, MedDRA cluster and treatment action taken
``supportive care``        active concomitant medications and drug-class flags
``physical examination``   number of abnormal body systems at the last examination
=========================  ==========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CFG,
    CONMED_CLASSES,
    LAB_CORE,
    LAB_DELTA_BASELINE,
    LAB_GRADED,
    LAB_SECONDARY,
    TOXICITY_CLUSTERS,
)
from .io import (
    lab_wide,
    tidy_adverse,
    tidy_lab,
    tidy_response,
    tidy_testdrug,
    tidy_tumour,
    tidy_vitals,
)

__all__ = [
    "assign_toxicity_clusters",
    "time_and_exposure_features",
    "dose_features",
    "ecog_features",
    "vitals_features",
    "lab_features",
    "tumour_features",
    "ae_history_features",
    "conmed_features",
    "physical_exam_features",
    "build_landmark_features",
]


# --------------------------------------------------------------------------------------
# Generic history aggregators
# --------------------------------------------------------------------------------------
def _history_aggregates(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                        value_cols: list[str], windows: tuple[int, ...],
                        prefix: str, landmark_col: str = "landmark_day") -> pd.DataFrame:
    """Cumulative and rolling-window sums of ``value_cols`` up to each landmark.

    Implemented with per-patient cumulative sums plus ``searchsorted`` so the cost
    is O(n_events log n_events) rather than a quadratic interval join.
    """
    n = len(grid)
    out: dict[str, np.ndarray] = {}
    for c in value_cols:
        out[f"{prefix}{c}_cum"] = np.zeros(n)
        for w in windows:
            out[f"{prefix}{c}_{w}d"] = np.zeros(n)

    ev = events.dropna(subset=[day_col])
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, sub in ev.groupby("PID", sort=False):
        sub = sub.sort_values(day_col)
        days = sub[day_col].to_numpy(dtype=float)
        vals = sub[value_cols].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        cums = np.vstack([np.zeros(len(value_cols)), np.cumsum(vals, axis=0)])
        lookup[pid] = (days, cums)

    pids = grid["PID"].to_numpy()
    ts = grid[landmark_col].to_numpy(dtype=float)

    for i in range(n):
        entry = lookup.get(pids[i])
        if entry is None:
            continue
        days, cums = entry
        hi = int(np.searchsorted(days, ts[i], side="right"))
        cum = cums[hi]
        for j, c in enumerate(value_cols):
            out[f"{prefix}{c}_cum"][i] = cum[j]
        for w in windows:
            lo = int(np.searchsorted(days, ts[i] - w, side="right"))
            win = cums[hi] - cums[lo]
            for j, c in enumerate(value_cols):
                out[f"{prefix}{c}_{w}d"][i] = win[j]

    return pd.DataFrame(out, index=grid.index)


def _last_observation(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                      value_cols: list[str], prefix: str,
                      landmark_col: str = "landmark_day",
                      add_days_since: bool = True) -> pd.DataFrame:
    """Most recent non-missing value of each column at or before each landmark.

    Uses :func:`pandas.merge_asof`, evaluating each measurement independently so a
    missing heart rate does not discard a valid blood pressure from the same visit.
    """
    g = grid[["PID", landmark_col]].copy()
    g[landmark_col] = g[landmark_col].astype(float)
    g["_row"] = np.arange(len(g))
    g = g.sort_values(landmark_col)

    pieces = [pd.Series(np.arange(len(grid)), name="_row")]
    for c in value_cols:
        e = (events.dropna(subset=[day_col, c])[["PID", day_col, c]]
                   .sort_values(day_col))
        e = e.astype({day_col: float})
        if e.empty:
            merged = g[["_row"]].copy()
            merged[f"{prefix}{c}"] = np.nan
            if add_days_since:
                merged[f"{prefix}{c}_days_since"] = np.nan
        else:
            merged = pd.merge_asof(g, e, left_on=landmark_col, right_on=day_col,
                                   by="PID", direction="backward")
            merged = merged.rename(columns={c: f"{prefix}{c}"})
            if add_days_since:
                merged[f"{prefix}{c}_days_since"] = merged[landmark_col] - merged[day_col]
            keep = ["_row", f"{prefix}{c}"] + (
                [f"{prefix}{c}_days_since"] if add_days_since else [])
            merged = merged[keep]
        pieces.append(merged.set_index("_row"))

    out = pd.concat(pieces[1:], axis=1).sort_index()
    out.index = grid.index
    return out


def _window_stats(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                  value_cols: list[str], window: int, prefix: str,
                  landmark_col: str = "landmark_day") -> pd.DataFrame:
    """Mean / min / max / linear trend of a measurement over the trailing window."""
    n = len(grid)
    stats = ["mean", "min", "max", "slope"]
    out = {f"{prefix}{c}_{s}_{window}d": np.full(n, np.nan)
           for c in value_cols for s in stats}

    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, sub in events.groupby("PID", sort=False):
        sub = sub.sort_values(day_col)
        lookup[pid] = (sub[day_col].to_numpy(dtype=float),
                       sub[value_cols].to_numpy(dtype=float))

    pids = grid["PID"].to_numpy()
    ts = grid[landmark_col].to_numpy(dtype=float)

    for i in range(n):
        entry = lookup.get(pids[i])
        if entry is None:
            continue
        days, vals = entry
        lo = int(np.searchsorted(days, ts[i] - window, side="right"))
        hi = int(np.searchsorted(days, ts[i], side="right"))
        if hi <= lo:
            continue
        d_win = days[lo:hi]
        v_win = vals[lo:hi]
        for j, c in enumerate(value_cols):
            col = v_win[:, j]
            ok = ~np.isnan(col)
            if not ok.any():
                continue
            x, y = d_win[ok], col[ok]
            out[f"{prefix}{c}_mean_{window}d"][i] = y.mean()
            out[f"{prefix}{c}_min_{window}d"][i] = y.min()
            out[f"{prefix}{c}_max_{window}d"][i] = y.max()
            if len(y) >= 2 and np.ptp(x) > 0:
                out[f"{prefix}{c}_slope_{window}d"][i] = np.polyfit(x, y, 1)[0]

    return pd.DataFrame(out, index=grid.index)


# --------------------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------------------
def assign_toxicity_clusters(ae: pd.DataFrame) -> pd.DataFrame:
    """Tag each adverse event with the clinical clusters its preferred term matches."""
    a = ae.copy()
    low = a["pt"].fillna("").str.lower()
    for cluster, keywords in TOXICITY_CLUSTERS.items():
        a[f"cl_{cluster}"] = low.apply(lambda s, kw=keywords: any(k in s for k in kw))
    return a


def time_and_exposure_features(grid: pd.DataFrame, testdrug: pd.DataFrame,
                               timeline: pd.DataFrame) -> pd.DataFrame:
    """Elapsed time, treatment cycle reached and recency of the last administration."""
    t = tidy_testdrug(testdrug).dropna(subset=["start_day"])

    out = pd.DataFrame(index=grid.index)
    out["time_on_study"] = grid["time_on_study"].to_numpy()
    out["weeks_on_study"] = out["time_on_study"] / 7.0

    last = _last_observation(t.assign(_cycle=t["cycle"]), grid, "start_day",
                             ["_cycle"], prefix="", add_days_since=True)
    out["cycle_number"] = last["_cycle"]
    out["days_since_last_dose"] = last["_cycle_days_since"]

    counts = _history_aggregates(t.assign(_one=1.0), grid, "start_day",
                                 ["_one"], windows=(28,), prefix="dose_records")
    out["n_dose_records_cum"] = counts["dose_records_one_cum"]
    out["n_dose_records_28d"] = counts["dose_records_one_28d"]
    return out


def dose_features(grid: pd.DataFrame, testdrug: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Cumulative exposure, recent exposure and relative dose intensity per component.

    Relative dose intensity (RDI) compares the dose delivered in the most recent
    28 days with the dose delivered in the first 28 days of treatment; values well
    below 1 indicate dose reductions or omitted administrations, which the oncology
    literature links to accumulating toxicity.
    """
    t = tidy_testdrug(testdrug).dropna(subset=["start_day"])
    t["dose"] = t["dose"].fillna(0.0)

    groups = [g for g in t["drug_group"].unique() if g != "other"]
    wide = t.pivot_table(index=["PID", "start_day"], columns="drug_group",
                         values="dose", aggfunc="sum").reset_index()
    for g in groups:
        if g not in wide.columns:
            wide[g] = 0.0
    wide[groups] = wide[groups].fillna(0.0)

    agg = _history_aggregates(wide, grid, "start_day", groups,
                              windows=(cfg.short_lookback_days,), prefix="dose_")

    # Reference exposure: dose delivered during the first 28 days of treatment.
    first28 = (wide.merge(
        grid.groupby("PID")["landmark_day"].min().rename("t0").reset_index(),
        on="PID", how="left"))
    first28 = first28[first28["start_day"] <= first28["t0"] + cfg.short_lookback_days]
    ref = first28.groupby("PID")[groups].sum().add_prefix("ref28_")

    ref_on_grid = grid[["PID"]].merge(ref, on="PID", how="left")
    ref_on_grid.index = grid.index

    w = cfg.short_lookback_days
    for g in groups:
        recent = agg[f"dose_{g}_{w}d"]
        reference = ref_on_grid[f"ref28_{g}"].replace(0.0, np.nan)
        agg[f"rdi_{g}"] = (recent / reference).clip(upper=3.0)
        agg[f"no_recent_dose_{g}"] = (recent <= 0).astype(float)

    # Dose-level trajectory: latest administered dose versus the starting dose.
    last_dose = _last_observation(wide, grid, "start_day", groups,
                                  prefix="last_dose_", add_days_since=False)
    start_dose = (wide.sort_values("start_day").groupby("PID")[groups].first()
                      .add_prefix("start_dose_"))
    start_on_grid = grid[["PID"]].merge(start_dose, on="PID", how="left")
    start_on_grid.index = grid.index

    for g in groups:
        ratio = last_dose[f"last_dose_{g}"] / start_on_grid[f"start_dose_{g}"].replace(0.0, np.nan)
        agg[f"dose_ratio_{g}"] = ratio.clip(upper=3.0)
        agg[f"dose_reduced_{g}"] = (ratio < 0.9).astype(float)

    return agg


def ecog_features(grid: pd.DataFrame, pfm: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Latest ECOG performance status, its change from baseline and recent worst."""
    p = pfm.copy()
    p["day"] = pd.to_numeric(p["EFDAY"], errors="coerce")
    p["ecog"] = pd.to_numeric(p["PFMECOG"], errors="coerce")
    p = p.dropna(subset=["day", "ecog"])
    p = p[p["ecog"] <= 4]

    last = _last_observation(p, grid, "day", ["ecog"], prefix="")
    stats = _window_stats(p, grid, "day", ["ecog"], cfg.long_lookback_days, prefix="")

    out = pd.DataFrame(index=grid.index)
    out["ecog_current"] = last["ecog"]
    out["ecog_days_since"] = last["ecog_days_since"]
    out["ecog_max_recent"] = stats[f"ecog_max_{cfg.long_lookback_days}d"]
    out["ecog_mean_recent"] = stats[f"ecog_mean_{cfg.long_lookback_days}d"]

    counts = _history_aggregates(p.assign(_one=1.0), grid, "day", ["_one"],
                                 windows=(cfg.long_lookback_days,), prefix="ecog_n")
    out["n_ecog_assessments"] = counts["ecog_n_one_cum"]
    return out


def vitals_features(grid: pd.DataFrame, vitals: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Latest vital signs plus rolling summaries over the long look-back window."""
    v = tidy_vitals(vitals)
    cols = [c for c in ["sbp", "dbp", "hr", "rr", "temp", "weight"] if c in v.columns]

    last = _last_observation(v, grid, "day", cols, prefix="vs_")
    stats = _window_stats(v, grid, "day", cols, cfg.long_lookback_days, prefix="vs_")
    out = pd.concat([last, stats], axis=1)

    if "vs_sbp" in out.columns and "vs_dbp" in out.columns:
        out["vs_pulse_pressure"] = out["vs_sbp"] - out["vs_dbp"]
        # Days-since is identical for all vitals from the same visit; keep one copy.
        out = out.drop(columns=[c for c in out.columns
                                if c.endswith("_days_since") and c != "vs_sbp_days_since"])
        out = out.rename(columns={"vs_sbp_days_since": "vs_days_since_assessment"})
    return out


def lab_features(grid: pd.DataFrame, lab_safe: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Safety laboratory trajectories - the block that closes the study's largest gap.

    Neutropenia alone is 210 of the 655 Grade >= 3 events on this trial, and it is
    *defined* by an absolute neutrophil count. Before ``LAB_SAFE`` was available the
    model could only infer myelosuppression indirectly, from prior graded events,
    growth-factor use and dose reductions.

    Features per analyte
    --------------------
    ``lab_{code}``               range-relative level at the most recent draw
    ``lab_{code}_min_56d``       nadir over the long look-back - the cytopenia signal
    ``lab_{code}_max_56d``       peak over the long look-back - the hepatic/renal signal
    ``lab_{code}_slope_56d``     linear trend, i.e. which way the analyte is moving
    ``lab_grade_{code}``         CTCAE v3.0 grade at the most recent draw

    plus a panel summary (worst grade, number of out-of-range and Grade >= 3 analytes)
    and the counts of draws and of Grade >= 3 laboratory abnormalities accumulated so
    far. Core analytes get the full treatment; the secondary panel contributes its
    latest level only, to keep the design matrix from being dominated by chemistry.

    Only draws at or before the landmark are read, exactly as for every other block.
    """
    tidy = tidy_lab(lab_safe)
    rel = lab_wide(tidy, "rel")
    grades = lab_wide(tidy, "ctcae")

    core = [c for c in LAB_CORE if c in rel.columns]
    secondary = [c for c in LAB_SECONDARY if c in rel.columns]
    all_codes = core + secondary

    # ---- latest level for every analyte ---------------------------------------------
    last = _last_observation(rel, grid, "day", all_codes, prefix="lab_")
    # Days-since is a property of the draw, not of the analyte; keep one copy.
    since_cols = [c for c in last.columns if c.endswith("_days_since")]
    out = last.drop(columns=since_cols)

    # ---- trajectory over the long look-back, core analytes only ----------------------
    if core:
        stats = _window_stats(rel, grid, "day", core, cfg.long_lookback_days, prefix="lab_")
        # The 56-day mean adds little beyond the nadir, the peak and the trend.
        stats = stats.drop(columns=[c for c in stats.columns if "_mean_" in c])
        out = pd.concat([out, stats], axis=1)

    # ---- CTCAE grade at the most recent draw ----------------------------------------
    graded = [c for c in LAB_GRADED if c in grades.columns]
    if graded:
        last_g = _last_observation(grades, grid, "day", graded, prefix="lab_grade_")
        out = pd.concat([out, last_g.drop(columns=[c for c in last_g.columns
                                                   if c.endswith("_days_since")])],
                        axis=1)
        gcols = [f"lab_grade_{c}" for c in graded]
        out["lab_max_grade"] = out[gcols].max(axis=1)
        out["lab_n_abnormal"] = (out[gcols] > 0).sum(axis=1)
        out["lab_n_grade3plus"] = (out[gcols] >= 3).sum(axis=1)

        # Accumulated laboratory toxicity: how often the panel has already been
        # severely deranged, over the whole history and over the look-back windows.
        per_draw = grades[["PID", "day"]].copy()
        per_draw["_g3_draw"] = (grades[graded] >= 3).any(axis=1).astype(float)
        per_draw["_g2_draw"] = (grades[graded] >= 2).any(axis=1).astype(float)
        hist = _history_aggregates(
            per_draw, grid, "day", ["_g3_draw", "_g2_draw"],
            (cfg.short_lookback_days, cfg.long_lookback_days), prefix="lab_n")
        out = pd.concat([out, hist], axis=1)

    # ---- draw recency and frequency --------------------------------------------------
    draws = rel[["PID", "day"]].copy()
    draws["_draw"] = 1.0
    recency = _last_observation(draws, grid, "day", ["_draw"], prefix="lab_last")
    out["lab_days_since_draw"] = recency["lab_last_draw_days_since"]
    counts = _history_aggregates(draws, grid, "day", ["_draw"],
                                 (cfg.short_lookback_days,), prefix="lab_n")
    out["lab_n_draws_cum"] = counts["lab_n_draw_cum"]
    out["lab_n_draws_28d"] = counts[f"lab_n_draw_{cfg.short_lookback_days}d"]

    return out


def tumour_features(grid: pd.DataFrame, domains: dict[str, pd.DataFrame],
                    cfg=CFG) -> pd.DataFrame:
    """Disease trajectory: how the tumour burden and RECIST response are moving.

    The baseline probe showed that tumour burden *at screening* carries almost no
    association with early severe toxicity. Whether its **trajectory** does is a different
    question, and it is the one this block is built to answer - which is the same logic
    that motivates the dissertation as a whole.

    The mechanism, if there is one, is indirect: a patient whose disease is progressing on
    treatment is deteriorating clinically, and a deteriorating patient tolerates cytotoxic
    therapy less well. This is association, not a causal pathway, and it is interpreted as
    a state marker exactly as the dose features are.

    Features
    --------
    ``tum_sld``                    burden at the most recent scan, mm
    ``tum_sld_pct_change_bl``      change from the screening scan, per cent
    ``tum_sld_pct_change_nadir``   rise from the smallest burden seen so far, which is the
                                   quantity RECIST progression is actually defined on
    ``tum_response``               latest overall response, 0 complete to 3 progressive
    ``tum_ever_progression``       any progressive assessment recorded so far
    ``tum_new_lesion_cum``         new lesions recorded so far
    ``tum_days_since_scan``        staleness of the assessment the features rest on

    Scans are roughly six-weekly against 14-day landmarks, so the median assessment is
    about three weeks old at the landmark; ``tum_days_since_scan`` is carried so the model
    can discount a stale reading.
    """
    out = pd.DataFrame(index=grid.index)

    if "tmm_p" in domains:
        tum = tidy_tumour(domains["tmm_p"])
        last = _last_observation(tum, grid, "day",
                                 ["sld", "n_target_lesions", "n_disease_sites"],
                                 prefix="tum_")
        out["tum_sld"] = last["tum_sld"]
        out["tum_n_target_lesions"] = last["tum_n_target_lesions"]
        out["tum_n_disease_sites"] = last["tum_n_disease_sites"]
        out["tum_days_since_scan"] = last["tum_sld_days_since"]

        # Baseline and running-nadir burden, both computed from history only.
        first_sld = (tum.dropna(subset=["sld"]).sort_values("day")
                        .groupby("PID")["sld"].first().rename("_bl_sld"))
        bl_on_grid = grid[["PID"]].merge(first_sld, on="PID", how="left")
        bl_on_grid.index = grid.index
        out["tum_sld_pct_change_bl"] = 100.0 * (
            out["tum_sld"] - bl_on_grid["_bl_sld"]) / bl_on_grid["_bl_sld"].replace(0.0, np.nan)

        nadir = _running_min(tum.dropna(subset=["sld"]), grid, "day", "sld")
        out["tum_sld_nadir"] = nadir
        out["tum_sld_pct_change_nadir"] = 100.0 * (
            out["tum_sld"] - nadir) / pd.Series(nadir, index=grid.index).replace(0.0, np.nan)

        stats = _window_stats(tum.dropna(subset=["sld"]), grid, "day", ["sld"],
                              cfg.long_lookback_days, prefix="tum_")
        out["tum_sld_slope_56d"] = stats[f"tum_sld_slope_{cfg.long_lookback_days}d"]

        counts = _history_aggregates(
            tum.assign(_scan=1.0, _new_les=tum["n_new_lesions"]), grid, "day",
            ["_scan", "_new_les"], (), prefix="tum_n")
        out["tum_n_scans_cum"] = counts["tum_n_scan_cum"]
        out["tum_new_lesion_cum"] = counts["tum_n_new_les_cum"]

    if "iota_p" in domains:
        resp = tidy_response(domains["iota_p"])
        last_r = _last_observation(resp, grid, "day", ["response"], prefix="tum_")
        out["tum_response"] = last_r["tum_response"]
        out["tum_days_since_response"] = last_r["tum_response_days_since"]
        out["tum_response_worst_so_far"] = _running_max(resp, grid, "day", "response")

        prog = _history_aggregates(resp, grid, "day", ["is_progression", "new_lesion"],
                                   (), prefix="tum_")
        out["tum_ever_progression"] = (prog["tum_is_progression_cum"] > 0).astype(float)
        out["tum_new_lesion_reported"] = (prog["tum_new_lesion_cum"] > 0).astype(float)

    return out


def _running_min(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                 value_col: str, landmark_col: str = "landmark_day") -> np.ndarray:
    """Smallest value observed at or before each landmark (the RECIST nadir)."""
    n = len(grid)
    res = np.full(n, np.nan)
    lookup = {}
    for pid, sub in events.dropna(subset=[day_col]).groupby("PID", sort=False):
        sub = sub.sort_values(day_col)
        days = sub[day_col].to_numpy(dtype=float)
        vals = sub[value_col].to_numpy(dtype=float)
        lookup[pid] = (days, np.fmin.accumulate(np.nan_to_num(vals, nan=np.inf)))
    pids, ts = grid["PID"].to_numpy(), grid[landmark_col].to_numpy(dtype=float)
    for i in range(n):
        e = lookup.get(pids[i])
        if e is None:
            continue
        days, run = e
        hi = int(np.searchsorted(days, ts[i], side="right"))
        if hi > 0 and np.isfinite(run[hi - 1]):
            res[i] = run[hi - 1]
    return res


def ae_history_features(grid: pd.DataFrame, adverse: pd.DataFrame,
                        timeline: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Accumulated toxicity burden - one of the two dominant temporal signal sources.

    Counts are split by grade, seriousness, causality, MedDRA cluster and the
    action taken on study treatment, each over the full history and over the short
    look-back window, so the model can distinguish a patient with a distant single
    toxicity from one deteriorating right now.

    On study A6181122 this block accounts for ~26% of total SHAP attribution, second
    only to the vital-sign trajectories (~32%); see notebook 06.
    """
    a = tidy_adverse(adverse)
    a = a.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")
    a = a[a["start_day"] >= a["first_dose_day"]]           # treatment-emergent only
    a = assign_toxicity_clusters(a)

    a["_any"] = 1.0
    a["_g2plus"] = (a["grade"] >= 2).astype(float)
    a["_g3plus"] = (a["grade"] >= cfg.severity_threshold).astype(float)
    a["_g4plus"] = (a["grade"] >= 4).astype(float)
    a["_serious"] = a["is_serious"].astype(float)
    a["_related"] = a["is_related"].astype(float)
    a["_reduction"] = a["led_to_reduction"].astype(float)
    a["_interruption"] = a["led_to_interruption"].astype(float)
    a["_discontinuation"] = a["led_to_discontinuation"].astype(float)

    # Leading underscore keeps the generated names readable: "n_ae" + "_cl_infection"
    # + "_cum"  ->  "n_ae_cl_infection_cum".
    cluster_cols = [f"_cl_{c}" for c in TOXICITY_CLUSTERS]
    for src, dst in zip((f"cl_{c}" for c in TOXICITY_CLUSTERS), cluster_cols):
        a[dst] = a[src].astype(float)
    a_sev = a[a["_g3plus"] > 0].copy()

    base_cols = ["_any", "_g2plus", "_g3plus", "_g4plus", "_serious", "_related",
                 "_reduction", "_interruption", "_discontinuation", *cluster_cols]

    windows = (cfg.short_lookback_days, cfg.long_lookback_days)
    counts = _history_aggregates(a, grid, "start_day", base_cols, windows, prefix="n_ae")
    sev_clusters = _history_aggregates(a_sev, grid, "start_day", cluster_cols,
                                       (cfg.short_lookback_days,), prefix="n_severe")

    out = pd.concat([counts, sev_clusters], axis=1)

    # Recency of the last event of any grade and of the last severe event.
    last_any = _last_observation(a.assign(_grade=a["grade"]), grid, "start_day",
                                 ["_grade"], prefix="last_ae")
    out["days_since_last_ae"] = last_any["last_ae_grade_days_since"]
    out["last_ae_grade"] = last_any["last_ae_grade"]

    last_sev = _last_observation(a_sev.assign(_grade=a_sev["grade"]), grid, "start_day",
                                 ["_grade"], prefix="last_sev")
    out["days_since_last_severe_ae"] = last_sev["last_sev_grade_days_since"]
    out["has_prior_severe_ae"] = last_sev["last_sev_grade"].notna().astype(float)

    # Running maximum grade and distinct preferred terms seen so far.
    out["max_grade_so_far"] = _running_max(a, grid, "start_day", "grade")
    out["n_distinct_pt_so_far"] = _running_nunique(a, grid, "start_day", "pt")

    # Events still unresolved at the landmark - observable in real time.
    out["n_ae_ongoing"] = _ongoing_counts(a, grid)
    out["n_severe_ae_ongoing"] = _ongoing_counts(a_sev, grid)

    return out


def _running_max(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                 value_col: str, landmark_col: str = "landmark_day") -> np.ndarray:
    n = len(grid)
    res = np.full(n, np.nan)
    lookup = {}
    for pid, sub in events.dropna(subset=[day_col]).groupby("PID", sort=False):
        sub = sub.sort_values(day_col)
        days = sub[day_col].to_numpy(dtype=float)
        vals = sub[value_col].to_numpy(dtype=float)
        lookup[pid] = (days, np.fmax.accumulate(np.nan_to_num(vals, nan=-np.inf)))
    pids, ts = grid["PID"].to_numpy(), grid[landmark_col].to_numpy(dtype=float)
    for i in range(n):
        e = lookup.get(pids[i])
        if e is None:
            continue
        days, run = e
        hi = int(np.searchsorted(days, ts[i], side="right"))
        if hi > 0 and np.isfinite(run[hi - 1]):
            res[i] = run[hi - 1]
    return res


def _running_nunique(events: pd.DataFrame, grid: pd.DataFrame, day_col: str,
                     value_col: str, landmark_col: str = "landmark_day") -> np.ndarray:
    n = len(grid)
    res = np.zeros(n)
    lookup = {}
    for pid, sub in events.dropna(subset=[day_col]).groupby("PID", sort=False):
        sub = sub.sort_values(day_col)
        days = sub[day_col].to_numpy(dtype=float)
        seen, running = set(), []
        for v in sub[value_col].fillna("<missing>"):
            seen.add(v)
            running.append(len(seen))
        lookup[pid] = (days, np.asarray(running, dtype=float))
    pids, ts = grid["PID"].to_numpy(), grid[landmark_col].to_numpy(dtype=float)
    for i in range(n):
        e = lookup.get(pids[i])
        if e is None:
            continue
        days, run = e
        hi = int(np.searchsorted(days, ts[i], side="right"))
        if hi > 0:
            res[i] = run[hi - 1]
    return res


def _ongoing_counts(events: pd.DataFrame, grid: pd.DataFrame,
                    landmark_col: str = "landmark_day") -> np.ndarray:
    """Number of events started at or before *t* and not yet resolved at *t*."""
    n = len(grid)
    res = np.zeros(n)
    lookup = {}
    for pid, sub in events.dropna(subset=["start_day"]).groupby("PID", sort=False):
        start = sub["start_day"].to_numpy(dtype=float)
        end = sub["end_day"].to_numpy(dtype=float)
        end = np.where(np.isnan(end), np.inf, end)     # unresolved at data cut-off
        lookup[pid] = (start, end)
    pids, ts = grid["PID"].to_numpy(), grid[landmark_col].to_numpy(dtype=float)
    for i in range(n):
        e = lookup.get(pids[i])
        if e is None:
            continue
        start, end = e
        res[i] = int(((start <= ts[i]) & (end > ts[i])).sum())
    return res


def conmed_features(grid: pd.DataFrame, condrug: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Supportive-care intensity: active concomitant drugs and drug-class flags.

    Growth-factor support in particular is a direct clinical marker of ongoing
    myelosuppression and is expected to carry substantial predictive weight.
    """
    c = condrug.copy()
    c["start_day"] = pd.to_numeric(c["CDFDAY"], errors="coerce")
    c["end_day"] = pd.to_numeric(c["CDTDAY"], errors="coerce")
    text = c["CDRGCOPT"].fillna("") + " " + c["CDRGREAS"].fillna("")
    low = text.str.lower()
    for name, kws in CONMED_CLASSES.items():
        c[f"cm_{name}"] = low.apply(lambda s, kw=kws: any(k in s for k in kw)).astype(float)

    class_cols = [f"_{n}" for n in CONMED_CLASSES]
    for name in CONMED_CLASSES:
        c[f"_{name}"] = c[f"cm_{name}"]
    c["_started"] = 1.0

    started = _history_aggregates(c, grid, "start_day", ["_started", *class_cols],
                                  (cfg.short_lookback_days,), prefix="n_conmed")

    out = started.copy()
    out["n_conmed_active"] = _ongoing_counts(c, grid)
    for name in CONMED_CLASSES:
        sub = c[c[f"cm_{name}"] > 0]
        out[f"conmed_{name}_active"] = (_ongoing_counts(sub, grid) > 0).astype(float)
    return out


def physical_exam_features(grid: pd.DataFrame, phyexam: pd.DataFrame) -> pd.DataFrame:
    """Abnormal-system count at the most recent physical examination before *t*."""
    p = phyexam.copy()
    p["day"] = pd.to_numeric(p["COLLDAY"], errors="coerce")
    p = p[p["RSLTCODC"].notna()]
    p["abnormal"] = p["RSLTCODC"].str.upper().eq("ABNORMAL").astype(float)
    per_visit = p.groupby(["PID", "day"], as_index=False).agg(
        n_abnormal_systems=("abnormal", "sum"),
        n_systems_examined=("abnormal", "size"))

    last = _last_observation(per_visit, grid, "day",
                             ["n_abnormal_systems", "n_systems_examined"],
                             prefix="pe_", add_days_since=False)
    return last


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def build_landmark_features(grid: pd.DataFrame,
                            domains: dict[str, pd.DataFrame],
                            timeline: pd.DataFrame,
                            baseline: pd.DataFrame,
                            cfg=CFG) -> pd.DataFrame:
    """Full landmark design matrix: baseline block + all time-varying blocks.

    Parameters
    ----------
    grid
        Landmark rows (``PID``, ``landmark_day``, ``time_on_study``, label columns).
    baseline
        Patient-level baseline matrix from
        :func:`aeprep.features_static.build_baseline_features`.
    """
    g = grid.reset_index(drop=True)

    blocks = [
        time_and_exposure_features(g, domains["testdrug"], timeline),
        dose_features(g, domains["testdrug"], cfg),
        ecog_features(g, domains["pfm_p"], cfg),
        vitals_features(g, domains["vitals"], cfg),
        ae_history_features(g, domains["adverse"], timeline, cfg),
        conmed_features(g, domains["condrug"], cfg),
        physical_exam_features(g, domains["phyexam"]),
    ]

    # The safety laboratory domain is optional: the pipeline still runs without it,
    # which is how the first analysis round was produced.
    if "lab_safe" in domains:
        blocks.append(lab_features(g, domains["lab_safe"], cfg))
    if {"tmm_p", "iota_p"} & set(domains):
        blocks.append(tumour_features(g, domains, cfg))

    temporal = pd.concat(blocks, axis=1)

    # ``time_on_study`` is regenerated inside the exposure block; keep a single copy.
    keys = g.drop(columns=[c for c in temporal.columns if c in g.columns])

    bl = baseline.add_prefix("bl__").reset_index()
    out = pd.concat([keys, temporal], axis=1).merge(bl, on="PID", how="left")

    # ---- derived deltas that need both blocks --------------------------------------
    if "bl__ecog_baseline" in out.columns:
        out["ecog_delta_baseline"] = out["ecog_current"] - out["bl__ecog_baseline"]
        out["ecog_worsened"] = (out["ecog_delta_baseline"] > 0).astype(float)
    for vs, bl_col in [("vs_weight", "bl__bl_weight"), ("vs_sbp", "bl__bl_sbp"),
                       ("vs_dbp", "bl__bl_dbp"), ("vs_hr", "bl__bl_hr"),
                       ("vs_temp", "bl__bl_temp")]:
        if vs in out.columns and bl_col in out.columns:
            out[f"{vs}_delta_baseline"] = out[vs] - out[bl_col]
    if {"vs_weight", "bl__bl_weight"}.issubset(out.columns):
        out["weight_pct_change"] = 100.0 * (out["vs_weight"] - out["bl__bl_weight"]) \
                                   / out["bl__bl_weight"].replace(0.0, np.nan)

    # Change in each key analyte since screening. A neutrophil count of 2.0 means one
    # thing in a patient who started at 2.2 and quite another in one who started at 6.
    for code in LAB_DELTA_BASELINE:
        cur, base = f"lab_{code}", f"bl__bl_lab_{code}"
        if cur in out.columns and base in out.columns:
            out[f"lab_{code}_delta_bl"] = out[cur] - out[base]

    # Counts are structurally zero when a patient has no record in a domain.
    count_like = [c for c in out.columns if c.startswith(("n_ae", "n_severe", "n_conmed",
                                                          "n_dose_records", "lab_n_draws",
                                                          "lab_n_g2_draw", "lab_n_g3_draw"))]
    out[count_like] = out[count_like].fillna(0.0)
    return out
