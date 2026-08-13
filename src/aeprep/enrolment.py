"""Pre-enrolment screening: a recommendation on whether a patient should start treatment.

This module implements the enrolment pre-check requested at the mid-semester review. It
runs *before* the first dose, using only information available at screening, and returns a
categorical recommendation with the reasons that produced it.

Why the recommendation is not simply a model threshold
------------------------------------------------------
Chapter 6 establishes that the baseline model's held-out discrimination is 0.607 with a
confidence interval that includes chance. Section 6.1.4 shows what happens if that score is
used on its own as an exclusion rule: at its most favourable threshold it refuses roughly
three patients to avert one severe event and two of those three would never have had one.
A system built that way would be indefensible.

So the pre-check is built the way trial screening is actually done, in three layers:

``Layer 1 - protocol safety criteria``
    Objective organ-function and performance thresholds of the kind written into oncology
    eligibility sections. These are what can legitimately produce a *do not enrol* or
    *defer* recommendation, because they are measured facts about whether the patient can
    metabolise and tolerate the regimen at all - not predictions.

``Layer 2 - model-based baseline risk``
    The static model's calibrated probability of a Grade >= 3 event in days 1-31. This is
    deliberately **not** allowed to exclude anyone. It escalates *monitoring intensity*,
    which is a decision the evidence supports.

``Layer 3 - combined recommendation``
    The two layers are resolved into one of four postures, and every recommendation is
    returned with the criteria that fired, their measured values and their thresholds.

The thresholds encode standard second-line colorectal practice for a FOLFIRI regimen: the
irinotecan bilirubin constraint, marrow-reserve floors, renal clearance and performance
status. They were fixed from clinical guidance before any outcome was inspected, and were
not tuned against the data - which is what allows the validation in notebook 03 to mean
something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import CFG
from .io import tidy_lab

__all__ = [
    "SCREENING_CRITERIA",
    "RECOMMENDATIONS",
    "baseline_screening_panel",
    "evaluate_criteria",
    "screen_patient",
    "screen_cohort",
    "validate_screening",
    "criterion_outcome_table",
    "format_screening_report",
]


# --------------------------------------------------------------------------------------
# Screening criteria
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """One pre-enrolment check.

    ``severity`` is ``"exclusion"`` for a value that should stop treatment starting, or
    ``"caution"`` for one that should not stop it but should change how the patient is
    watched. ``reversible`` marks a criterion that a repeat test may clear, which is what
    separates *defer and recheck* from *do not enrol*.
    """

    key: str
    label: str
    severity: str
    reversible: bool
    rationale: str
    test: Callable[[pd.Series], bool]
    describe: Callable[[pd.Series], str]


def _v(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    try:
        x = float(row[col])
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _lt(row: pd.Series, col: str, thr: float) -> bool:
    x = _v(row, col)
    return bool(np.isfinite(x) and x < thr)


def _between(row: pd.Series, col: str, lo: float, hi: float) -> bool:
    x = _v(row, col)
    return bool(np.isfinite(x) and lo <= x < hi)


def _gt(row: pd.Series, col: str, thr: float) -> bool:
    x = _v(row, col)
    return bool(np.isfinite(x) and x > thr)


def _fmt(row: pd.Series, col: str, unit: str = "", dp: int = 2) -> str:
    x = _v(row, col)
    return "not measured" if not np.isfinite(x) else f"{x:.{dp}f}{unit}"


#: Ordered so that the most consequential organ-function checks are reported first.
SCREENING_CRITERIA: tuple[Criterion, ...] = (
    # ---- marrow reserve -------------------------------------------------------------
    Criterion(
        key="anc_inadequate", label="Absolute neutrophil count below 1.5 x10^9/L",
        severity="exclusion", reversible=True,
        rationale="FOLFIRI is myelosuppressive and neutropenia is 210 of the 655 severe "
                  "events on this trial. Starting below the conventional 1.5 floor means "
                  "starting without the marrow reserve to absorb the first cycle.",
        test=lambda r: _lt(r, "anc", 1.5),
        describe=lambda r: f"ANC {_fmt(r, 'anc')} x10^9/L (floor 1.5)"),
    Criterion(
        key="anc_borderline", label="Neutrophil count 1.5 to 2.0 x10^9/L",
        severity="caution", reversible=True,
        rationale="Adequate to start but with little margin; the first-cycle nadir is "
                  "likely to breach Grade 3.",
        test=lambda r: _between(r, "anc", 1.5, 2.0),
        describe=lambda r: f"ANC {_fmt(r, 'anc')} x10^9/L (adequate but marginal)"),
    Criterion(
        key="plt_inadequate", label="Platelet count below 100 x10^9/L",
        severity="exclusion", reversible=True,
        rationale="Standard marrow-function floor for cytotoxic chemotherapy; below it the "
                  "bleeding risk of a further fall is not acceptable.",
        test=lambda r: _lt(r, "plt", 100.0),
        describe=lambda r: f"Platelets {_fmt(r, 'plt', dp=0)} x10^9/L (floor 100)"),
    Criterion(
        key="plt_borderline", label="Platelet count 100 to 150 x10^9/L",
        severity="caution", reversible=True,
        rationale="Below the usual reference range; warrants a confirmed count before "
                  "each of the first two cycles.",
        test=lambda r: _between(r, "plt", 100.0, 150.0),
        describe=lambda r: f"Platelets {_fmt(r, 'plt', dp=0)} x10^9/L (low-normal)"),
    Criterion(
        key="hgb_inadequate", label="Haemoglobin below 9.0 g/dL",
        severity="exclusion", reversible=True,
        rationale="Transfusion-threshold anaemia before any cytotoxic exposure; correct "
                  "first, then reassess.",
        test=lambda r: _lt(r, "hgb", 9.0),
        describe=lambda r: f"Haemoglobin {_fmt(r, 'hgb')} g/dL (floor 9.0)"),
    Criterion(
        key="hgb_borderline", label="Haemoglobin 9.0 to 10.0 g/dL",
        severity="caution", reversible=True,
        rationale="Anaemic at entry, so a further treatment-related fall is likely to "
                  "become symptomatic.",
        test=lambda r: _between(r, "hgb", 9.0, 10.0),
        describe=lambda r: f"Haemoglobin {_fmt(r, 'hgb')} g/dL (low)"),
    # ---- hepatic: the irinotecan constraint -----------------------------------------
    Criterion(
        key="bili_high", label="Total bilirubin above 1.5 x upper limit of normal",
        severity="exclusion", reversible=True,
        rationale="Irinotecan is cleared hepatically and its active metabolite SN-38 is "
                  "glucuronidated; raised bilirubin sharply increases exposure and the "
                  "risk of severe neutropenia and diarrhoea. This is the single firmest "
                  "pharmacological contraindication in the regimen.",
        test=lambda r: _gt(r, "bili_ratio_uln", 1.5),
        describe=lambda r: f"Bilirubin {_fmt(r, 'bili_ratio_uln')} x ULN (limit 1.5)"),
    Criterion(
        key="bili_borderline", label="Total bilirubin 1.0 to 1.5 x upper limit of normal",
        severity="caution", reversible=True,
        rationale="Within the permitted range but with reduced clearance margin; a "
                  "starting-dose review is reasonable.",
        test=lambda r: _between(r, "bili_ratio_uln", 1.0, 1.5),
        describe=lambda r: f"Bilirubin {_fmt(r, 'bili_ratio_uln')} x ULN (upper range)"),
    Criterion(
        key="transaminase_high", label="ALT or AST above 2.5 x upper limit of normal",
        severity="exclusion", reversible=True,
        rationale="Hepatocellular injury at this level compromises clearance of both "
                  "cytotoxic components.",
        test=lambda r: _gt(r, "alt_ratio_uln", 2.5) or _gt(r, "ast_ratio_uln", 2.5),
        describe=lambda r: (f"ALT {_fmt(r, 'alt_ratio_uln')} x ULN, "
                            f"AST {_fmt(r, 'ast_ratio_uln')} x ULN (limit 2.5)")),
    # ---- renal ------------------------------------------------------------------------
    Criterion(
        key="creat_high", label="Creatinine above 1.5 x upper limit of normal",
        severity="exclusion", reversible=True,
        rationale="Impaired renal clearance raises fluorouracil exposure and compounds "
                  "the dehydration that follows irinotecan-associated diarrhoea.",
        test=lambda r: _gt(r, "creat_ratio_uln", 1.5),
        describe=lambda r: f"Creatinine {_fmt(r, 'creat_ratio_uln')} x ULN (limit 1.5)"),
    Criterion(
        key="creat_borderline", label="Creatinine 1.2 to 1.5 x upper limit of normal",
        severity="caution", reversible=True,
        rationale="Reduced renal reserve; confirm hydration planning before cycle 1.",
        test=lambda r: _between(r, "creat_ratio_uln", 1.2, 1.5),
        describe=lambda r: f"Creatinine {_fmt(r, 'creat_ratio_uln')} x ULN (upper range)"),
    # ---- nutritional / general state --------------------------------------------------
    Criterion(
        key="albumin_low", label="Serum albumin below 2.5 g/dL",
        severity="exclusion", reversible=True,
        rationale="Profound hypoalbuminaemia indicates a nutritional and inflammatory "
                  "state incompatible with tolerating a doublet. Albumin is also the "
                  "strongest single feature in the temporal model.",
        test=lambda r: _lt(r, "alb", 2.5),
        describe=lambda r: f"Albumin {_fmt(r, 'alb')} g/dL (floor 2.5)"),
    Criterion(
        key="albumin_borderline", label="Serum albumin 2.5 to 3.5 g/dL",
        severity="caution", reversible=False,
        rationale="Below the reference range. Dietetic assessment before cycle 1 is "
                  "indicated; this is the commonest caution in this cohort.",
        test=lambda r: _between(r, "alb", 2.5, 3.5),
        describe=lambda r: f"Albumin {_fmt(r, 'alb')} g/dL (below reference range)"),
    Criterion(
        key="ecog_poor", label="ECOG performance status 2 or worse",
        severity="exclusion", reversible=False,
        rationale="Second-line doublet chemotherapy in metastatic disease is conventionally "
                  "restricted to ECOG 0-1; this trial enrolled 97 per cent at 0 or 1.",
        test=lambda r: _v(r, "ecog") >= 2 if np.isfinite(_v(r, "ecog")) else False,
        describe=lambda r: f"ECOG {_fmt(r, 'ecog', dp=0)} (limit 1)"),
    # ---- treatment history ------------------------------------------------------------
    Criterion(
        key="prior_toxicity_stop", label="Previous anti-cancer regimen stopped for toxicity",
        severity="caution", reversible=False,
        rationale="A patient who could not complete their last regimen because of toxicity "
                  "is the archetype an enrolment check exists to notice. Recorded for 17 "
                  "patients here; the agent and best response are absent from this release, "
                  "so the reason for stopping is the only prior-therapy signal recoverable.",
        test=lambda r: _v(r, "prior_stopped_for_toxicity") >= 1
        if np.isfinite(_v(r, "prior_stopped_for_toxicity")) else False,
        describe=lambda r: "previous regimen discontinued for toxicity"),
    Criterion(
        key="prior_pelvic_rt", label="Prior pelvic or abdominal radiotherapy",
        severity="caution", reversible=False,
        rationale="Radiotherapy to marrow-bearing pelvic or abdominal bone permanently "
                  "reduces haematopoietic reserve, which matters before a myelosuppressive "
                  "doublet. Not an exclusion - it is a reason to watch the counts, not to "
                  "refuse treatment.",
        test=lambda r: _v(r, "prior_pelvic_radiotherapy") >= 1
        if np.isfinite(_v(r, "prior_pelvic_radiotherapy")) else False,
        describe=lambda r: "prior pelvic/abdominal radiotherapy recorded"),
    Criterion(
        key="elderly", label="Age 75 or above",
        severity="caution", reversible=False,
        rationale="Not an exclusion, but reduced marrow and renal reserve make a "
                  "starting-dose discussion appropriate.",
        test=lambda r: _v(r, "age") >= 75 if np.isfinite(_v(r, "age")) else False,
        describe=lambda r: f"Age {_fmt(r, 'age', dp=0)} years"),
    # Severity "data" rather than "caution": this reports that the check could not be
    # completed, not that the patient is at higher risk, so it must not push anyone into
    # intensified monitoring. The audit in notebook 03 confirms the distinction - it flags
    # 72 patients at 0.85x the base event rate, i.e. it does not track risk at all, while
    # every clinical criterion does.
    Criterion(
        key="incomplete_panel", label="Screening panel incomplete",
        severity="data", reversible=True,
        rationale="One or more of the organ-function tests required to clear the patient "
                  "was not recorded, so the check cannot be completed as specified.",
        test=lambda r: any(not np.isfinite(_v(r, c))
                           for c in ("anc", "plt", "hgb", "bili_ratio_uln",
                                     "creat_ratio_uln", "alb")),
        describe=lambda r: "missing: " + ", ".join(
            c for c in ("anc", "plt", "hgb", "bili_ratio_uln", "creat_ratio_uln", "alb")
            if not np.isfinite(_v(r, c)))),
)

#: The four postures the system can return, worst first.
RECOMMENDATIONS: tuple[tuple[str, str], ...] = (
    ("DO NOT ENROL",
     "One or more protocol safety criteria are not met. Treatment should not start."),
    ("DEFER AND RECHECK",
     "A reversible criterion is not met. Repeat the failing test and reassess rather "
     "than refusing the patient outright."),
    ("ENROL WITH INTENSIFIED MONITORING",
     "Eligible to start. Cautions and/or elevated baseline risk warrant a tighter "
     "first-cycle monitoring schedule."),
    ("ENROL - STANDARD MONITORING",
     "Eligible to start, with no caution triggered and baseline risk in the routine band."),
)


# --------------------------------------------------------------------------------------
# Building the screening panel
# --------------------------------------------------------------------------------------
def baseline_screening_panel(domains: dict[str, pd.DataFrame],
                             timeline: pd.DataFrame) -> pd.DataFrame:
    """One row per patient of the values the criteria are evaluated on.

    Laboratory results are taken as the last measurement at or before the first dose, in
    *absolute* harmonised units together with the ratio to the record's own upper limit of
    normal, because eligibility criteria are written on absolute values and on multiples
    of the ULN rather than on the range-relative levels used for modelling.
    """
    cutoff = timeline.set_index("PID")["first_dose_day"]

    lab = tidy_lab(domains["lab_safe"])
    lab = lab.merge(cutoff.rename("cut").rename_axis("PID").reset_index(), on="PID", how="left")
    lab = lab[lab["day"] <= lab["cut"]].sort_values(["PID", "day"])

    out = pd.DataFrame(index=cutoff.index)
    for code in ("anc", "plt", "hgb", "alb", "bili", "alt", "ast", "creat"):
        sub = lab[lab["analyte"].eq(code)]
        val = sub.dropna(subset=["value"]).groupby("PID")["value"].last()
        out[code] = val
        if code in ("bili", "alt", "ast", "creat"):
            ratio = sub.dropna(subset=["ratio_uln"]).groupby("PID")["ratio_uln"].last()
            out[f"{code}_ratio_uln"] = ratio

    pfm = domains["pfm_p"].copy()
    pfm["day"] = pd.to_numeric(pfm["EFDAY"], errors="coerce")
    pfm["ecog"] = pd.to_numeric(pfm["PFMECOG"], errors="coerce")
    pfm = pfm.dropna(subset=["day", "ecog"])
    pfm = pfm[pfm["ecog"] <= 4].merge(
        cutoff.rename("cut").rename_axis("PID").reset_index(), on="PID", how="left")
    out["ecog"] = (pfm[pfm["day"] <= pfm["cut"]].sort_values("day")
                      .groupby("PID")["ecog"].last())

    demo = domains["demog"].copy()
    out["age"] = pd.to_numeric(demo["AGE"], errors="coerce").groupby(demo["PID"]).first()

    # Treatment history, where those domains are present.
    if "cd_b_p" in domains:
        from .io import tidy_prior_therapy
        pt = tidy_prior_therapy(domains["cd_b_p"]).set_index("PID")
        out["prior_stopped_for_toxicity"] = pt["prior_stopped_for_toxicity"]
        out["n_prior_regimens"] = pt["n_prior_regimens"]
    if "cn_6_p" in domains:
        from .io import tidy_radiotherapy
        rt = tidy_radiotherapy(domains["cn_6_p"]).set_index("PID")
        out["prior_pelvic_radiotherapy"] = rt["prior_pelvic_radiotherapy"]

    out.index.name = "PID"
    return out.reset_index()


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_criteria(row: pd.Series) -> list[Criterion]:
    """Every criterion the patient triggers, in declaration order."""
    return [c for c in SCREENING_CRITERIA if bool(c.test(row))]


def screen_patient(row: pd.Series, baseline_risk: float | None = None,
                   risk_threshold: float = 0.30) -> dict:
    """Full pre-enrolment recommendation for one patient.

    ``baseline_risk`` is the static model's calibrated probability of a Grade >= 3 event
    in days 1 to 31. It can raise the monitoring intensity but never produces an exclusion,
    for the reasons set out in the module docstring and quantified in Section 6.1.4.
    """
    fired = evaluate_criteria(row)
    exclusions = [c for c in fired if c.severity == "exclusion"]
    cautions = [c for c in fired if c.severity == "caution"]
    data_flags = [c for c in fired if c.severity == "data"]

    risk_elevated = (baseline_risk is not None and np.isfinite(baseline_risk)
                     and baseline_risk >= risk_threshold)

    # Data flags deliberately do not enter this decision - see the note on the
    # incomplete_panel criterion.
    if exclusions and any(not c.reversible for c in exclusions):
        posture = "DO NOT ENROL"
    elif exclusions:
        posture = "DEFER AND RECHECK"
    elif cautions or risk_elevated:
        posture = "ENROL WITH INTENSIFIED MONITORING"
    else:
        posture = "ENROL - STANDARD MONITORING"

    return {
        "PID": row.get("PID"),
        "recommendation": posture,
        "meaning": dict(RECOMMENDATIONS)[posture],
        "n_exclusions": len(exclusions),
        "n_cautions": len(cautions),
        "n_data_flags": len(data_flags),
        "baseline_risk": float(baseline_risk) if baseline_risk is not None else np.nan,
        "risk_elevated": bool(risk_elevated),
        "exclusions": [{"key": c.key, "label": c.label, "value": c.describe(row),
                        "reversible": c.reversible, "rationale": c.rationale}
                       for c in exclusions],
        "cautions": [{"key": c.key, "label": c.label, "value": c.describe(row),
                      "rationale": c.rationale} for c in cautions],
        "data_flags": [{"key": c.key, "label": c.label, "value": c.describe(row)}
                       for c in data_flags],
    }


def screen_cohort(panel: pd.DataFrame, risks: pd.Series | None = None,
                  risk_threshold: float = 0.30) -> pd.DataFrame:
    """Vectorised screening over a whole cohort, one row per patient."""
    rows = []
    for _, r in panel.iterrows():
        risk = None
        if risks is not None and r["PID"] in risks.index:
            risk = risks.loc[r["PID"]]
        rec = screen_patient(r, risk, risk_threshold)
        flat = {k: v for k, v in rec.items()
                if k not in ("exclusions", "cautions", "data_flags", "meaning")}
        flat["exclusion_keys"] = "; ".join(c["key"] for c in rec["exclusions"])
        flat["caution_keys"] = "; ".join(c["key"] for c in rec["cautions"])
        flat["data_flag_keys"] = "; ".join(c["key"] for c in rec["data_flags"])
        rows.append(flat)
    out = pd.DataFrame(rows)
    order = [p for p, _ in RECOMMENDATIONS]
    out["recommendation"] = pd.Categorical(out["recommendation"], order, ordered=True)
    return out


def validate_screening(screened: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Do the flagged patients actually do worse? The only honest test of the pre-check.

    ``outcomes`` must carry ``PID`` and any of ``y_early`` (severe event in days 1-31),
    ``treatment_duration``, ``discont_due_to_ae`` and ``died``. Every enrolled patient
    passed the trial's own screening, so this asks whether the pre-check identifies, among
    people who *were* accepted, those who went on to tolerate treatment badly.
    """
    df = screened.merge(outcomes, on="PID", how="left")
    agg: dict[str, tuple] = {"n_patients": ("PID", "size")}
    if "y_early" in df.columns:
        agg["early_severe_ae_rate"] = ("y_early", "mean")
    if "treatment_duration" in df.columns:
        agg["median_treatment_days"] = ("treatment_duration", "median")
    if "discont_due_to_ae" in df.columns:
        agg["discont_due_to_ae_rate"] = ("discont_due_to_ae", "mean")
    if "died" in df.columns:
        agg["death_rate"] = ("died", "mean")

    out = df.groupby("recommendation", observed=False).agg(**agg)
    return out.reset_index()


def criterion_outcome_table(screened: pd.DataFrame, outcomes: pd.DataFrame,
                            outcome_col: str = "y_early") -> pd.DataFrame:
    """Per-criterion audit: what happened to the patients each individual check flagged.

    The aggregate gradient in :func:`validate_screening` could be produced by one dominant
    criterion carrying all the others. This breaks it out so that a criterion which flags
    patients no likelier than average to run into trouble is visible as such, rather than
    hidden inside a favourable total.
    """
    df = screened.merge(outcomes, on="PID", how="left")
    base = float(df[outcome_col].mean())
    rows = []
    for c in SCREENING_CRITERIA:
        hit = pd.Series(False, index=df.index)
        for col in ("exclusion_keys", "caution_keys", "data_flag_keys"):
            if col in df.columns:
                hit |= df[col].fillna("").str.contains(c.key, regex=False)
        if not hit.any():
            rows.append({"criterion": c.key, "severity": c.severity, "n_flagged": 0,
                         "outcome_rate": np.nan, "lift_vs_base": np.nan,
                         "median_treatment_days": np.nan})
            continue
        rows.append({
            "criterion": c.key,
            "severity": c.severity,
            "n_flagged": int(hit.sum()),
            "outcome_rate": float(df.loc[hit, outcome_col].mean()),
            "lift_vs_base": float(df.loc[hit, outcome_col].mean()) / base if base else np.nan,
            "median_treatment_days": (float(df.loc[hit, "treatment_duration"].median())
                                      if "treatment_duration" in df.columns else np.nan),
        })
    out = pd.DataFrame(rows)
    out.attrs["base_rate"] = base
    return out.sort_values("n_flagged", ascending=False).reset_index(drop=True)


def format_screening_report(record: dict, width: int = 88) -> str:
    """Render one pre-enrolment decision as the screening note a trial team would file."""
    line = "=" * width
    L = [line,
         f" PRE-ENROLMENT SCREENING  -  PATIENT {record['PID']}",
         line,
         f" RECOMMENDATION: {record['recommendation']}",
         f"   {record['meaning']}"]

    if np.isfinite(record.get("baseline_risk", np.nan)):
        L.append(f" Model baseline risk of a Grade >=3 event in days 1-{CFG.prediction_horizon_days + 1}: "
                 f"{record['baseline_risk']:.1%}"
                 f"{'  (elevated)' if record['risk_elevated'] else ''}")
        L.append("   Note: this score informs monitoring intensity only. It never excludes a")
        L.append("   patient - its held-out interval includes chance (Section 6.1.4).")

    if record["exclusions"]:
        L += ["", f" PROTOCOL CRITERIA NOT MET ({len(record['exclusions'])})"]
        for i, c in enumerate(record["exclusions"], 1):
            L.append(f"   {i}. {c['label']}")
            L.append(f"      Measured: {c['value']}")
            L.append(f"      {'Reversible - repeat the test' if c['reversible'] else 'Not reversible'}")
            L.append(f"      Why: {c['rationale']}")

    if record["cautions"]:
        L += ["", f" CAUTIONS ({len(record['cautions'])}) - do not prevent enrolment"]
        for i, c in enumerate(record["cautions"], 1):
            L.append(f"   {i}. {c['label']}  [{c['value']}]")

    if record.get("data_flags"):
        L += ["", " DATA COMPLETENESS - does not affect the recommendation"]
        for c in record["data_flags"]:
            L.append(f"   - {c['label']}  [{c['value']}]")

    if not record["exclusions"] and not record["cautions"]:
        L += ["", " No protocol criterion or clinical caution triggered."]

    L += ["", " Decision support for the screening clinician. Protocol criteria are measured",
          " facts about organ function, not predictions; the model score is advisory only.",
          line]
    return "\n".join(L)
