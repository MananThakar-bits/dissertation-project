"""Turning a landmark risk estimate into a proactive monitoring action.

A probability on its own does not change what happens to a patient. This module is the
step between the model's output and something a trial team could actually do before the
next cycle: it converts a predicted 30-day risk into a **risk tier**, and the patient's
current clinical state into a short list of **specific pre-emptive actions**, each one
carrying the model statistic that justifies it.

Design constraints, stated because they bound how the output may be read
------------------------------------------------------------------------
* **The tiers are the measured operating points, not round numbers.** Each threshold is
  taken from the held-out threshold table in notebook 05, and every recommendation is
  emitted together with the sensitivity, specificity, PPV and NPV actually observed at
  that threshold. A recommendation is never shown without the statistics that earn it.
* **The action rules are clinically motivated heuristics, not learned policy.** The model
  supplies *when* to act and *what the driver is*; the mapping from driver to action comes
  from standard FOLFIRI toxicity management, not from the data. Nothing here is a causal
  claim, and no rule was tuned against the outcome.
* **Every trigger is a feature the model already uses**, evaluated strictly at or before
  the landmark, so an action could genuinely have been suggested on the day.
* The output is decision *support*: it proposes what to look at, never what to prescribe.

The honest headline for this layer is the negative predictive value. At the 0.15
threshold a visit the model leaves in the routine tier is followed by 30 event-free days
about 92% of the time, which is what makes de-prioritisation - not escalation - the
defensible use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import CFG

__all__ = [
    "RISK_TIERS",
    "ACTION_RULES",
    "risk_tier",
    "tier_statistics",
    "recommend_for_row",
    "recommend_frame",
    "action_frequency_table",
    "tier_outcome_table",
    "format_monitoring_card",
]


# --------------------------------------------------------------------------------------
# Risk tiers
# --------------------------------------------------------------------------------------
#: ``(lower_bound, tier, posture)``. Bounds are the operating points measured on the
#: held-out landmark rows, not arbitrary round numbers - see notebook 05 Section 9.
RISK_TIERS: tuple[tuple[float, str, str], ...] = (
    (0.30, "high", "Escalate: review before the next cycle"),
    (0.15, "elevated", "Pre-empt: act on the drivers below"),
    (0.00, "routine", "De-prioritise: routine schedule is adequate"),
)


def risk_tier(p: float) -> str:
    """Tier for a single calibrated probability."""
    if not np.isfinite(p):
        return "unknown"
    for lower, tier, _ in RISK_TIERS:
        if p >= lower:
            return tier
    return "routine"


def tier_statistics(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Observed performance of each tier boundary, for display beside the advice.

    This is the evidence a recommendation is shown with. ``sensitivity`` and ``npv`` are
    computed at the tier's own lower bound, so "routine" is reported with the NPV that
    justifies de-prioritising it and "high" with the PPV that justifies escalating it.
    """
    from .evaluation import operating_point

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []
    for lower, tier, posture in RISK_TIERS:
        in_tier = p >= lower
        upper = min((lo for lo, _, _ in RISK_TIERS if lo > lower), default=1.01)
        band = in_tier & (p < upper)
        op = operating_point(y, p, lower)
        rows.append({
            "tier": tier,
            "threshold": lower,
            "posture": posture,
            "n_rows_in_band": int(band.sum()),
            "observed_event_rate_in_band": float(y[band].mean()) if band.any() else np.nan,
            "sensitivity_at_threshold": op["sensitivity"],
            "specificity_at_threshold": op["specificity"],
            "ppv_at_threshold": op["ppv"],
            "npv_at_threshold": op["npv"],
            "alert_rate_at_threshold": op["alert_rate"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Action rules
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionRule:
    """One pre-emptive action, its trigger, and why it is clinically reasonable."""

    key: str
    domain: str
    action: str
    rationale: str
    trigger: Callable[[pd.Series], bool]
    evidence: tuple[str, ...] = field(default_factory=tuple)
    min_tier: str = "elevated"


def _num(row: pd.Series, col: str, default: float = np.nan) -> float:
    """Feature value as a float, tolerating absent or missing columns."""
    if col not in row.index:
        return default
    v = row[col]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(v) else v


def _ge(row: pd.Series, col: str, thr: float) -> bool:
    v = _num(row, col)
    return bool(np.isfinite(v) and v >= thr)


def _le(row: pd.Series, col: str, thr: float) -> bool:
    v = _num(row, col)
    return bool(np.isfinite(v) and v <= thr)


#: Ordered by clinical priority; the myelosuppression rules come first because
#: neutropenia is 210 of the 655 severe events on this trial.
ACTION_RULES: tuple[ActionRule, ...] = (
    ActionRule(
        key="anc_monitoring",
        domain="Myelosuppression",
        action="Bring the next full blood count forward to day 7-10 of the cycle rather "
               "than waiting for the scheduled pre-dose draw.",
        rationale="The neutrophil nadir after FOLFIRI falls around days 10-14, between "
                  "two-weekly scheduled draws. In this cohort 36 patients had a Grade 3 "
                  "or higher neutropenia recorded clinically with no qualifying count in "
                  "the panel, so the schedule is systematically missing the trough.",
        trigger=lambda r: _ge(r, "lab_grade_anc", 1) or _le(r, "lab_anc_min_56d", 0.0),
        evidence=("lab_grade_anc", "lab_anc", "lab_anc_min_56d", "lab_anc_slope_56d"),
        min_tier="elevated",
    ),
    ActionRule(
        key="gcsf_prophylaxis",
        domain="Myelosuppression",
        action="Review growth-factor prophylaxis before the next cycle; none is currently "
               "active for this patient.",
        rationale="Grade 2 or worse neutropenia with no active growth-factor support is "
                  "the state in which primary prophylaxis is conventionally considered.",
        trigger=lambda r: (_ge(r, "lab_grade_anc", 2)
                           and not _ge(r, "conmed_growth_factor_active", 1)),
        evidence=("lab_grade_anc", "conmed_growth_factor_active", "lab_anc"),
        min_tier="elevated",
    ),
    ActionRule(
        key="febrile_neutropenia_watch",
        domain="Infection",
        action="Counsel on immediate fever reporting and set a low threshold for "
               "empirical antibiotics.",
        rationale="Grade 3 or worse neutropenia together with recent infection-cluster "
                  "events or a rising temperature is the febrile-neutropenia precursor "
                  "state; febrile neutropenia is 14 of the severe events here.",
        trigger=lambda r: (_ge(r, "lab_grade_anc", 3)
                           or (_ge(r, "lab_grade_anc", 2)
                               and (_ge(r, "n_ae_cl_infection_28d", 1)
                                    or _ge(r, "vs_temp", 37.5)))),
        evidence=("lab_grade_anc", "n_ae_cl_infection_28d", "vs_temp"),
        min_tier="elevated",
    ),
    ActionRule(
        key="anaemia_review",
        domain="Myelosuppression",
        action="Check haematinics and review the transfusion threshold.",
        rationale="Grade 2 or worse anaemia on a cytotoxic regimen warrants review "
                  "before it reaches a transfusion-dependent level.",
        trigger=lambda r: _ge(r, "lab_grade_hgb", 2),
        evidence=("lab_grade_hgb", "lab_hgb", "lab_hgb_min_56d"),
        min_tier="elevated",
    ),
    ActionRule(
        key="bleeding_precautions",
        domain="Myelosuppression",
        action="Review anticoagulant and NSAID exposure; advise bleeding precautions.",
        rationale="Thrombocytopenia with concurrent anticoagulation is the combination "
                  "that converts a laboratory abnormality into a bleeding event.",
        trigger=lambda r: (_ge(r, "lab_grade_plt", 2)
                           or (_ge(r, "lab_grade_plt", 1)
                               and _ge(r, "conmed_anticoagulant_active", 1))),
        evidence=("lab_grade_plt", "lab_plt", "conmed_anticoagulant_active"),
        # Fires from any tier: a platelet count low enough to matter alongside
        # anticoagulation is a safety check that should not be gated behind the
        # patient's overall risk score. It fires on only a handful of visits here
        # because thrombocytopenia is genuinely uncommon on FOLFIRI - 4 patients in
        # the whole cohort reach Grade 3 - which is a property of the regimen rather
        # than a defect in the rule.
        min_tier="routine",
    ),
    ActionRule(
        key="gi_prophylaxis",
        domain="Gastrointestinal",
        action="Reinforce the loperamide protocol and oral hydration plan; check "
               "antiemetic adequacy.",
        rationale="Irinotecan-associated diarrhoea is the second-largest severe cluster "
                  "and has the shortest onset of any cluster here, a median of about six "
                  "days, so it needs pre-emptive rather than reactive management.",
        trigger=lambda r: (_ge(r, "n_ae_cl_gastrointestinal_28d", 1)
                           or _ge(r, "n_severe_cl_gastrointestinal_cum", 1)),
        evidence=("n_ae_cl_gastrointestinal_28d", "n_severe_cl_gastrointestinal_cum",
                  "conmed_antidiarrhoeal_active"),
        min_tier="elevated",
    ),
    ActionRule(
        key="nutrition_referral",
        domain="Constitutional",
        action="Refer for dietetic assessment and nutritional support.",
        rationale="Falling albumin and weight loss are the model's strongest single "
                  "drivers; the 56-day minimum albumin is the highest-ranked feature in "
                  "the entire model by mean absolute SHAP.",
        trigger=lambda r: (_le(r, "weight_pct_change", -5.0)
                           or _ge(r, "lab_grade_alb", 1)
                           or _le(r, "lab_alb_min_56d", 0.0)),
        evidence=("weight_pct_change", "lab_alb", "lab_alb_min_56d", "lab_grade_alb"),
        min_tier="elevated",
    ),
    ActionRule(
        key="hepatic_review",
        domain="Hepatic",
        action="Repeat liver function tests before the next cycle and review the "
               "irinotecan dose against hepatic-impairment guidance.",
        rationale="Irinotecan clearance is hepatically mediated, so a rising bilirubin "
                  "or transaminase changes the effective exposure from an unchanged dose.",
        trigger=lambda r: (_ge(r, "lab_grade_bili", 1) or _ge(r, "lab_grade_alt", 2)
                           or _ge(r, "lab_grade_ast", 2)),
        evidence=("lab_grade_bili", "lab_grade_alt", "lab_bili", "lab_alt"),
        min_tier="elevated",
    ),
    ActionRule(
        key="renal_review",
        domain="Renal",
        action="Recheck creatinine, review nephrotoxic co-medication and confirm the "
               "hydration plan.",
        rationale="Rising creatinine alters clearance of the regimen and commonly "
                  "reflects under-hydration secondary to diarrhoea or vomiting.",
        trigger=lambda r: (_ge(r, "lab_grade_creat", 1)
                           or _ge(r, "lab_grade_na", 1) or _ge(r, "lab_grade_k", 1)),
        evidence=("lab_grade_creat", "lab_creat", "lab_grade_na", "lab_grade_k"),
        min_tier="elevated",
    ),
    ActionRule(
        key="dose_review",
        domain="Treatment exposure",
        action="Put dose reduction or a cycle delay on the agenda for the next cycle "
               "review; full dose intensity is currently being maintained.",
        rationale="Accumulating Grade 2 events at maintained relative dose intensity is "
                  "the pattern that precedes a Grade 3 event. Note this is a marker, not "
                  "a causal lever - the dose analysis is explicitly associational.",
        trigger=lambda r: (_ge(r, "n_ae_g2plus_28d", 2)
                           and _ge(r, "rdi_fluorouracil", 0.9)
                           and not _ge(r, "dose_reduced_fluorouracil", 1)),
        evidence=("n_ae_g2plus_28d", "rdi_fluorouracil", "dose_reduced_fluorouracil"),
        min_tier="high",
    ),
    ActionRule(
        key="unresolved_events",
        domain="Toxicity history",
        action="Review the adverse events still unresolved at this visit before "
               "administering the next cycle.",
        rationale="Unresolved toxicity at the point of re-dosing is both observable in "
                  "real time and among the model's leading predictors.",
        trigger=lambda r: _ge(r, "n_ae_ongoing", 3) or _ge(r, "n_severe_ae_ongoing", 1),
        evidence=("n_ae_ongoing", "n_severe_ae_ongoing", "days_since_last_severe_ae"),
        min_tier="elevated",
    ),
    ActionRule(
        key="stale_laboratory",
        domain="Data adequacy",
        action="Obtain a current safety panel: the most recent result predates this "
               "visit by more than three weeks.",
        rationale="The risk estimate is only as current as its inputs. This rule fires "
                  "on the model's own uncertainty about the patient rather than on the "
                  "patient's condition.",
        trigger=lambda r: _ge(r, "lab_days_since_draw", 21),
        evidence=("lab_days_since_draw", "lab_n_draws_28d"),
        min_tier="routine",
    ),
)

_TIER_ORDER = {"routine": 0, "elevated": 1, "high": 2, "unknown": 0}


# --------------------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------------------
def recommend_for_row(row: pd.Series, p: float,
                      shap_row: pd.Series | None = None,
                      stats: pd.DataFrame | None = None,
                      top_drivers: int = 5) -> dict:
    """Full recommendation for one patient-visit.

    Parameters
    ----------
    row
        The landmark feature row, exactly as the model saw it.
    p
        Calibrated predicted probability of a Grade >= 3 event in the next
        :data:`config.CFG.prediction_horizon_days` days.
    shap_row
        Optional per-feature SHAP values for this row; used to report *why* the model
        assigned this risk, alongside *what to do* from the rules.
    stats
        Optional output of :func:`tier_statistics`, so the returned record carries the
        held-out operating characteristics of the tier it landed in.
    """
    tier = risk_tier(p)
    fired = [r for r in ACTION_RULES
             if _TIER_ORDER[tier] >= _TIER_ORDER[r.min_tier] and bool(r.trigger(row))]

    drivers: list[dict] = []
    if shap_row is not None and len(shap_row):
        s = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
        for name, val in s.head(top_drivers).items():
            drivers.append({"feature": str(name), "shap": float(val),
                            "direction": "increases risk" if val > 0 else "reduces risk"})

    record = {
        "PID": row.get("PID"),
        "landmark_day": row.get("landmark_day"),
        "horizon_days": CFG.prediction_horizon_days,
        "risk": float(p),
        "tier": tier,
        "posture": next(po for lo, t, po in RISK_TIERS if t == tier) if tier != "unknown" else "",
        "n_actions": len(fired),
        "actions": [{"key": r.key, "domain": r.domain, "action": r.action,
                     "rationale": r.rationale,
                     "evidence": {c: _num(row, c) for c in r.evidence if c in row.index}}
                    for r in fired],
        "drivers": drivers,
    }
    if stats is not None and tier in set(stats["tier"]):
        st = stats.set_index("tier").loc[tier]
        record["tier_statistics"] = {
            "threshold": float(st["threshold"]),
            "observed_event_rate_in_band": float(st["observed_event_rate_in_band"]),
            "sensitivity": float(st["sensitivity_at_threshold"]),
            "specificity": float(st["specificity_at_threshold"]),
            "ppv": float(st["ppv_at_threshold"]),
            "npv": float(st["npv_at_threshold"]),
        }
    return record


def recommend_frame(df: pd.DataFrame, prob_col: str = "p_temporal_calibrated",
                    stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vectorised tier and fired-rule assignment for a whole set of landmark rows."""
    p = df[prob_col].to_numpy(dtype=float)
    out = pd.DataFrame({
        "PID": df["PID"].to_numpy() if "PID" in df.columns else np.arange(len(df)),
        "landmark_day": df["landmark_day"].to_numpy() if "landmark_day" in df.columns else np.nan,
        "risk": p,
        "tier": [risk_tier(v) for v in p],
    }, index=df.index)
    if "y" in df.columns:
        out["y"] = df["y"].to_numpy()

    for rule in ACTION_RULES:
        fires = df.apply(rule.trigger, axis=1).astype(bool)
        eligible = out["tier"].map(lambda t: _TIER_ORDER[t] >= _TIER_ORDER[rule.min_tier])
        out[f"act_{rule.key}"] = (fires & eligible).astype(int)

    act_cols = [c for c in out.columns if c.startswith("act_")]
    out["n_actions"] = out[act_cols].sum(axis=1)
    return out


def action_frequency_table(rec: pd.DataFrame) -> pd.DataFrame:
    """How often each action fires, and the observed event rate when it does.

    The ``event_rate_when_fired`` column is what makes the advice auditable: an action
    that fires on visits no likelier than average to be followed by a severe event is
    not earning its place, whatever its clinical face validity.
    """
    rows = []
    base = float(rec["y"].mean()) if "y" in rec.columns else np.nan
    for rule in ACTION_RULES:
        col = f"act_{rule.key}"
        if col not in rec.columns:
            continue
        fired = rec[col] == 1
        rows.append({
            "action": rule.key,
            "domain": rule.domain,
            "fires_from_tier": rule.min_tier,
            "n_fired": int(fired.sum()),
            "pct_of_visits": float(fired.mean()),
            "event_rate_when_fired": float(rec.loc[fired, "y"].mean())
            if ("y" in rec.columns and fired.any()) else np.nan,
            "lift_vs_base_rate": (float(rec.loc[fired, "y"].mean()) / base)
            if ("y" in rec.columns and fired.any() and base) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_fired", ascending=False).reset_index(drop=True)


def tier_outcome_table(rec: pd.DataFrame) -> pd.DataFrame:
    """Observed outcome by tier - the statistical backing for the posture of each tier."""
    g = (rec.groupby("tier")
            .agg(n_visits=("risk", "size"), n_patients=("PID", "nunique"),
                 mean_predicted_risk=("risk", "mean"),
                 observed_event_rate=("y", "mean"), n_events=("y", "sum"))
            .reindex([t for _, t, _ in RISK_TIERS]).dropna(how="all"))
    g["share_of_all_events"] = g["n_events"] / g["n_events"].sum()
    g["share_of_visits"] = g["n_visits"] / g["n_visits"].sum()
    return g.reset_index()


def format_monitoring_card(record: dict, width: int = 88) -> str:
    """Render one recommendation as the plain-text card a monitor would receive."""
    line = "=" * width
    L = [line,
         f" PATIENT {record['PID']}   study day {record['landmark_day']:.0f}"
         f"   |   {record['horizon_days']}-DAY RISK WINDOW",
         line,
         f" Predicted risk of a Grade >=3 adverse event in the next "
         f"{record['horizon_days']} days: {record['risk']:.1%}",
         f" Tier: {record['tier'].upper()}  -  {record['posture']}"]

    st = record.get("tier_statistics")
    if st:
        L += [f" Held-out performance at this threshold (p >= {st['threshold']:.2f}): "
              f"sensitivity {st['sensitivity']:.2f}, specificity {st['specificity']:.2f},",
              f"   PPV {st['ppv']:.2f}, NPV {st['npv']:.2f}; observed event rate in this "
              f"band {st['observed_event_rate_in_band']:.1%}."]

    L += ["", " WHY THE MODEL SCORED THIS VISIT (top SHAP contributions)"]
    if record["drivers"]:
        for d in record["drivers"]:
            L.append(f"   {d['shap']:+.4f}  {d['feature']:<34s} {d['direction']}")
    else:
        L.append("   (not computed for this row)")

    L += ["", f" SUGGESTED PRE-EMPTIVE ACTIONS ({record['n_actions']})"]
    if record["actions"]:
        for i, a in enumerate(record["actions"], 1):
            L.append(f"   {i}. [{a['domain']}] {a['action']}")
            L.append(f"      Why: {a['rationale']}")
            if a["evidence"]:
                ev = ", ".join(f"{k}={v:.3g}" for k, v in a["evidence"].items()
                               if np.isfinite(v))
                if ev:
                    L.append(f"      Triggering values: {ev}")
    else:
        L.append("   None. Routine monitoring schedule is adequate for this visit.")

    L += ["", " Decision support only. Proposes what to review, never what to prescribe;",
          " no rule was tuned against the outcome and none carries a causal claim.",
          line]
    return "\n".join(L)
