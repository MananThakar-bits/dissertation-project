"""Baseline (static) feature engineering.

Produces the patient-level design matrix used by the *static baseline model* and,
reused verbatim, as the time-invariant block of the *temporal landmark model*.

Everything here is computed from information available at or before the first dose
of study treatment, so the same feature values would have been available to a
clinician on the day the patient started therapy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CFG,
    CONMED_CLASSES,
    LAB_ANALYTES,
    LAB_GRADED,
    TUMOUR_SITES,
)
from .io import (
    lab_wide,
    tidy_adverse,
    tidy_lab,
    tidy_prior_therapy,
    tidy_radiotherapy,
    tidy_testdrug,
    tidy_tumour,
    tidy_vitals,
)

__all__ = [
    "demographic_features",
    "baseline_ecog",
    "baseline_vitals",
    "baseline_labs",
    "baseline_tumour_burden",
    "prior_therapy_features",
    "medical_history_features",
    "baseline_conmed_features",
    "starting_dose_features",
    "disease_history_features",
    "baseline_physical_exam_features",
    "pretreatment_ae_features",
    "build_baseline_features",
]

#: Medical-history categories (substring match on the MedDRA lowest level term).
MEDICAL_HISTORY_CLASSES: dict[str, tuple[str, ...]] = {
    "mh_cardiovascular": ("hypertension", "myocardial", "angina", "cardiac", "arrhythmia",
                          "atrial fibrillation", "heart failure", "coronary"),
    "mh_diabetes": ("diabetes", "diabetic"),
    "mh_thromboembolic": ("thrombosis", "embolism", "thrombo"),
    "mh_hepatic": ("hepatic", "liver", "hepatitis", "cirrhosis"),
    "mh_renal": ("renal", "kidney", "nephro"),
    "mh_respiratory": ("asthma", "copd", "pulmonary", "chronic obstructive"),
    "mh_gastrointestinal": ("gastro", "ulcer", "reflux", "colitis", "constipation",
                            "diarrhoea", "diarrhea"),
    "mh_prior_malignancy": ("cancer", "carcinoma", "neoplasm", "tumour", "tumor"),
    "mh_anaemia": ("anemia", "anaemia"),
    "mh_infection": ("infection", "pneumonia"),
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _last_before(df: pd.DataFrame, day_col: str, cutoff: pd.Series,
                 value_cols: list[str], prefix: str) -> pd.DataFrame:
    """Latest non-missing value of each ``value_cols`` at or before a per-patient cutoff."""
    cut = cutoff.rename("cutoff").rename_axis("PID").reset_index()
    d = df.merge(cut, on="PID", how="left")
    d = d[d[day_col] <= d["cutoff"]]
    d = d.sort_values(["PID", day_col])
    out = {}
    for c in value_cols:
        sub = d.dropna(subset=[c])
        out[f"{prefix}{c}"] = sub.groupby("PID")[c].last()
    return pd.DataFrame(out)


def _keyword_flags(text: pd.Series, classes: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """One boolean column per class, true when any keyword occurs in ``text``."""
    low = text.fillna("").str.lower()
    return pd.DataFrame({name: low.apply(lambda s, kw=kws: any(k in s for k in kw))
                         for name, kws in classes.items()},
                        index=text.index)


# --------------------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------------------
def demographic_features(demog: pd.DataFrame) -> pd.DataFrame:
    """Age, sex, race, height, weight, BMI and body surface area (Du Bois)."""
    d = demog.copy()
    d["age"] = pd.to_numeric(d["AGE"], errors="coerce")
    d["height_cm"] = pd.to_numeric(d["HT"], errors="coerce")
    d["weight_kg"] = pd.to_numeric(d["WT"], errors="coerce")
    d["sex_male"] = d["SEXC"].str.strip().str.lower().eq("male").astype(float)
    d["race"] = d["RACESC"].str.strip().fillna("Unknown")

    d["bmi"] = d["weight_kg"] / (d["height_cm"] / 100.0) ** 2
    # Du Bois body-surface area - the scaling used for FOLFIRI dosing.
    d["bsa_m2"] = 0.007184 * (d["height_cm"] ** 0.725) * (d["weight_kg"] ** 0.425)
    d["age_ge_65"] = (d["age"] >= 65).astype(float)

    cols = ["PID", "age", "age_ge_65", "sex_male", "race",
            "height_cm", "weight_kg", "bmi", "bsa_m2"]
    return d[cols].drop_duplicates("PID").set_index("PID")


def baseline_ecog(pfm: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Baseline ECOG performance status (latest assessment up to the first dose)."""
    p = pfm.copy()
    p["day"] = pd.to_numeric(p["EFDAY"], errors="coerce")
    p["ecog"] = pd.to_numeric(p["PFMECOG"], errors="coerce")
    p = p.dropna(subset=["day", "ecog"])
    # ECOG 5 = death; not a valid baseline performance status.
    p = p[p["ecog"] <= 4]

    cutoff = timeline.set_index("PID")["first_dose_day"]
    out = _last_before(p[["PID", "day", "ecog"]], "day", cutoff, ["ecog"], "baseline_")
    out = out.rename(columns={"baseline_ecog": "ecog_baseline"})
    out["ecog_baseline_ge1"] = (out["ecog_baseline"] >= 1).astype(float)
    return out


def baseline_vitals(vitals: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Baseline vital signs (latest measurement up to and including the first dose)."""
    v = tidy_vitals(vitals)
    cutoff = timeline.set_index("PID")["first_dose_day"]
    value_cols = [c for c in ["sbp", "dbp", "hr", "rr", "temp", "weight"] if c in v.columns]
    out = _last_before(v, "day", cutoff, value_cols, "bl_")
    if {"bl_sbp", "bl_dbp"}.issubset(out.columns):
        out["bl_pulse_pressure"] = out["bl_sbp"] - out["bl_dbp"]
    return out


def baseline_labs(lab_safe: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Screening safety laboratory panel: the last result at or before the first dose.

    Each analyte is represented by its range-relative level (see :func:`io.tidy_lab`),
    which is dimensionless and absorbs the between-site variation in reference ranges.
    The block also carries the worst CTCAE grade and the number of out-of-range
    analytes at that draw, so a patient entering treatment with marginal marrow
    reserve is distinguishable from one entering with a normal panel.

    Adding this block to the *static* model is what keeps the static-versus-temporal
    comparison honest: both models then see the laboratory panel, and the contrast is
    purely between a value frozen at enrolment and one that updates.
    """
    tidy = tidy_lab(lab_safe)
    cutoff = timeline.set_index("PID")["first_dose_day"]

    rel = lab_wide(tidy, "rel")
    codes = [c for c in LAB_ANALYTES.values() if c in rel.columns]
    out = _last_before(rel, "day", cutoff, codes, "bl_lab_")

    grades = lab_wide(tidy, "ctcae")
    graded = [c for c in LAB_GRADED if c in grades.columns]
    if graded:
        g = grades.merge(cutoff.rename("cutoff").rename_axis("PID").reset_index(),
                         on="PID", how="left")
        g = g[g["day"] <= g["cutoff"]].sort_values(["PID", "day"])
        g["_worst"] = g[graded].max(axis=1)
        g["_n_abn"] = (g[graded] > 0).sum(axis=1)
        g["_n_g3"] = (g[graded] >= 3).sum(axis=1)
        last_draw = g.groupby("PID").last()
        out["bl_lab_max_grade"] = last_draw["_worst"]
        out["bl_lab_n_abnormal"] = last_draw["_n_abn"]
        out["bl_lab_n_grade3plus"] = last_draw["_n_g3"]

    out.index.name = "PID"
    return out


def baseline_tumour_burden(tmm: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Disease burden at the screening scan: RECIST sum of diameters, extent and sites.

    Burden is a prognostic quantity rather than a toxicity mechanism, so it is included as
    an auxiliary baseline block. The exception is liver involvement, which is mechanistic
    here: hepatic metastases impair clearance of irinotecan's active metabolite SN-38.
    """
    tum = tidy_tumour(tmm)
    cutoff = timeline.set_index("PID")["first_dose_day"]
    t = tum.merge(cutoff.rename("cut").rename_axis("PID").reset_index(), on="PID", how="left")
    base = t[t["day"] <= t["cut"]]
    if base.empty:
        return pd.DataFrame(index=pd.Index([], name="PID"))

    # The screening assessment is the last one recorded at or before the first dose.
    last = base.sort_values("day").groupby("PID").last()

    out = pd.DataFrame(index=last.index)
    out["bl_sld_mm"] = last["sld"]
    out["bl_n_target_lesions"] = last["n_target_lesions"]
    out["bl_n_disease_sites"] = last["n_disease_sites"]
    for s in TUMOUR_SITES:
        col = f"site_{s.lower()}"
        if col in last.columns:
            out[f"bl_{col}"] = last[col].fillna(0.0)
    out["bl_days_since_screening_scan"] = last["cut"] - last["day"]
    out.index.name = "PID"
    return out


def prior_therapy_features(domains: dict[str, pd.DataFrame],
                           timeline: pd.DataFrame) -> pd.DataFrame:
    """Treatment history before study entry: prior systemic regimens and radiotherapy.

    For a second-line trial this is the block a screening oncologist reaches for first.
    Two caveats travel with it. The agent and the best response to each prior regimen are
    empty throughout this release, so only the count, the setting and the reason for
    stopping are recoverable. And the regimen number is recorded for 62 per cent of
    patients, so the count is missing for the rest rather than zero - it is left missing
    and imputed inside the model pipeline, not silently read as "no prior therapy".
    """
    cutoff = timeline.set_index("PID")["first_dose_day"]
    out = pd.DataFrame(index=cutoff.index)

    if "cd_b_p" in domains:
        pt = tidy_prior_therapy(domains["cd_b_p"]).set_index("PID")
        out["bl_n_prior_regimens"] = pt["n_prior_regimens"]
        out["bl_prior_adjuvant"] = pt["prior_adjuvant"]
        out["bl_prior_neoadjuvant"] = pt["prior_neoadjuvant"]
        out["bl_prior_stopped_for_toxicity"] = pt["prior_stopped_for_toxicity"]
        out["bl_prior_stopped_for_progression"] = pt["prior_stopped_for_progression"]
        gap = cutoff - pt["prior_therapy_end_day"]
        out["bl_days_since_prior_therapy"] = gap.where(gap.between(0, 3650))

    if "cn_6_p" in domains:
        rt = tidy_radiotherapy(domains["cn_6_p"]).set_index("PID")
        out["bl_prior_radiotherapy"] = rt["prior_radiotherapy"]
        out["bl_prior_pelvic_radiotherapy"] = rt["prior_pelvic_radiotherapy"]
        gap = cutoff - rt["radiotherapy_end_day"]
        out["bl_days_since_radiotherapy"] = gap.where(gap.between(0, 3650))

    out.index.name = "PID"
    return out


def medical_history_features(prevdis: pd.DataFrame) -> pd.DataFrame:
    """Counts and clinically grouped flags derived from the medical-history domain."""
    p = prevdis.dropna(subset=["MHDECD1"]).copy()
    flags = _keyword_flags(p["MHDECD1"], MEDICAL_HISTORY_CLASSES)
    p = pd.concat([p[["PID", "DISSTATC"]], flags], axis=1)
    p["is_ongoing"] = p["DISSTATC"].str.upper().eq("PRESENT")

    agg = p.groupby("PID").agg(
        n_medical_history=("is_ongoing", "size"),
        n_ongoing_conditions=("is_ongoing", "sum"),
        **{c: (c, "max") for c in flags.columns},
    )
    for c in flags.columns:
        agg[c] = agg[c].astype(float)
    return agg


def disease_history_features(primdiag: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Time from primary cancer diagnosis to the start of study treatment (months)."""
    p = primdiag.copy()
    p["PRIMDAY"] = pd.to_numeric(p["PRIMDAY"], errors="coerce")
    p = p.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")
    p["months_since_diagnosis"] = (p["first_dose_day"] - p["PRIMDAY"]) / 30.4375
    out = p.groupby("PID").agg(months_since_diagnosis=("months_since_diagnosis", "max"))
    out["newly_diagnosed_lt6m"] = (out["months_since_diagnosis"] < 6).astype(float)
    return out


def baseline_conmed_features(condrug: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Concomitant medication burden present at the start of study treatment."""
    c = condrug.copy()
    c["start_day"] = pd.to_numeric(c["CDFDAY"], errors="coerce")
    c["stop_day"] = pd.to_numeric(c["CDTDAY"], errors="coerce")
    c = c.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")

    started_before = c["start_day"] <= c["first_dose_day"]
    still_active = c["stop_day"].isna() | (c["stop_day"] >= c["first_dose_day"])
    base = c[started_before & still_active].copy()

    text = base["CDRGCOPT"].fillna("") + " " + base["CDRGREAS"].fillna("")
    flags = _keyword_flags(text, CONMED_CLASSES).add_prefix("bl_conmed_")
    base = pd.concat([base[["PID"]], flags], axis=1)
    base["_one"] = 1

    agg = base.groupby("PID").agg(
        n_baseline_conmeds=("_one", "sum"),
        **{c_: (c_, "max") for c_ in flags.columns},
    )
    for c_ in flags.columns:
        agg[c_] = agg[c_].astype(float)
    return agg


def starting_dose_features(testdrug: pd.DataFrame, timeline: pd.DataFrame,
                           demographics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dose actually delivered on the first day of treatment, per drug component.

    These are the *prescribed starting doses* - known at the moment the patient
    enters treatment - and are the dissertation's "treatment exposure at baseline"
    auxiliary predictors. Doses are also expressed per m^2 of body-surface area,
    which is how FOLFIRI is prescribed.
    """
    t = tidy_testdrug(testdrug)
    t = t.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")
    first = t[(t["start_day"] <= t["first_dose_day"] + 2) & t["dose"].notna()]

    wide = (first.pivot_table(index="PID", columns="drug_group", values="dose",
                              aggfunc="max")
                 .add_prefix("start_dose_"))

    if demographics is not None and "bsa_m2" in demographics.columns:
        bsa = demographics["bsa_m2"]
        for col in [c for c in wide.columns if c != "start_dose_blinded"]:
            wide[f"{col}_per_m2"] = wide[col] / bsa
    return wide


def baseline_physical_exam_features(phyexam: pd.DataFrame,
                                    timeline: pd.DataFrame) -> pd.DataFrame:
    """Number of abnormal body systems at the screening physical examination."""
    p = phyexam.copy()
    p["day"] = pd.to_numeric(p["COLLDAY"], errors="coerce")
    p = p.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")
    p = p[(p["day"] <= p["first_dose_day"]) & p["RSLTCODC"].notna()]
    p["abnormal"] = p["RSLTCODC"].str.upper().eq("ABNORMAL")

    return p.groupby("PID").agg(
        n_systems_examined=("abnormal", "size"),
        n_abnormal_systems=("abnormal", "sum"),
    ).astype(float)


def pretreatment_ae_features(adverse: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Adverse events already recorded before the first dose (pre-existing toxicity)."""
    a = tidy_adverse(adverse).merge(timeline[["PID", "first_dose_day"]],
                                    on="PID", how="left")
    pre = a[a["start_day"] < a["first_dose_day"]]
    return pre.groupby("PID").agg(
        n_pretreatment_ae=("start_day", "size"),
        max_pretreatment_grade=("grade", "max"),
    ).astype(float)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def build_baseline_features(domains: dict[str, pd.DataFrame],
                            timeline: pd.DataFrame,
                            cfg=CFG) -> pd.DataFrame:
    """Assemble every baseline block into one patient-level matrix.

    Returns a frame indexed by ``PID``; categorical columns are left as plain
    strings and are one-hot encoded inside the modelling pipeline.
    """
    demo = demographic_features(domains["demog"])

    blocks = [
        demo,
        baseline_ecog(domains["pfm_p"], timeline),
        baseline_vitals(domains["vitals"], timeline),
        medical_history_features(domains["prevdis"]),
        disease_history_features(domains["primdiag"], timeline),
        baseline_conmed_features(domains["condrug"], timeline),
        starting_dose_features(domains["testdrug"], timeline, demo),
        baseline_physical_exam_features(domains["phyexam"], timeline),
        pretreatment_ae_features(domains["adverse"], timeline),
    ]

    # The safety laboratory domain is optional: the pipeline still runs without it,
    # which is how the first analysis round was produced.
    if "lab_safe" in domains:
        blocks.append(baseline_labs(domains["lab_safe"], timeline))
    if "tmm_p" in domains:
        blocks.append(baseline_tumour_burden(domains["tmm_p"], timeline))
    if {"cd_b_p", "cn_6_p"} & set(domains):
        blocks.append(prior_therapy_features(domains, timeline))

    out = blocks[0]
    for b in blocks[1:]:
        out = out.join(b, how="left")

    # Count-type features are genuinely zero when the domain has no record for a patient.
    zero_fill = [c for c in out.columns
                 if c.startswith(("n_", "mh_", "bl_conmed_", "max_pretreatment"))]
    out[zero_fill] = out[zero_fill].fillna(0.0)

    out.index.name = "PID"
    return out.sort_index()
