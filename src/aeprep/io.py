"""Ingestion layer for the Project Data Sphere A6181122 SAS transport files.

Responsibilities
----------------
* read ``*.sas7bdat`` into tidy pandas frames (with byte-string decoding);
* normalise the subject identifier ``PID_A`` into a compact ``PID``;
* coerce the character-typed vital signs into numeric SI units;
* derive the per-patient timeline anchors (first dose, end of treatment,
  end of study, adverse-event observation end) that every downstream step needs.

Study-day convention
--------------------
All ``*DAY`` variables in this study are integer study days relative to the first
dose of study treatment (Day 1). Negative values are screening/pre-treatment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CFG,
    CORE_DOMAINS,
    DATA_DIR,
    DRUG_GROUPS,
    LAB_ANALYTES,
    LAB_CTCAE_HIGH_ABS,
    LAB_CTCAE_HIGH_MULT,
    LAB_CTCAE_LOW,
    LAB_MISSING_TOKENS,
    LAB_NO_REFERENCE_RANGE,
    LAB_PLAUSIBLE,
    LAB_UNIT_FIX,
    OPTIONAL_DOMAINS,
    PELVIC_RT_PATTERN,
    RECIST_ORDER,
    SPECS_FILE,
    TUMOUR_SITES,
    VITALS_NUMERIC,
)

__all__ = [
    "normalise_pid",
    "read_domain",
    "load_all_domains",
    "load_specs",
    "available_domains",
    "tidy_vitals",
    "tidy_adverse",
    "tidy_testdrug",
    "tidy_lab",
    "lab_wide",
    "tidy_tumour",
    "tidy_response",
    "tidy_prior_therapy",
    "tidy_radiotherapy",
    "build_patient_timeline",
]


# --------------------------------------------------------------------------------------
# Low-level readers
# --------------------------------------------------------------------------------------
def normalise_pid(series: pd.Series) -> pd.Series:
    """Turn ``'A6181122    1'`` into ``'0001'``.

    ``PID_A`` concatenates the protocol number and a right-aligned subject number.
    Collapsing it to a zero-padded subject number keeps joins cheap, plots legible
    and - importantly - makes lexicographic sorting agree with numeric order.
    """
    s = series.astype("string").str.strip()
    # Drop the leading protocol token when present.
    subj = s.str.replace(r"^A?\d{7,8}\s*", "", regex=True).str.strip()
    subj = subj.where(subj.str.len() > 0, s)
    numeric = pd.to_numeric(subj, errors="coerce")
    padded = numeric.astype("Int64").astype("string").str.zfill(4)
    return padded.fillna(subj).astype("string")


def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode any residual byte-strings and strip padding whitespace."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], bytes):
                df[col] = df[col].str.decode("latin-1", errors="replace")
            if df[col].dtype == object:
                df[col] = df[col].str.strip().replace({"": np.nan})
    return df


def read_domain(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read one SAS domain and attach a normalised ``PID`` column.

    Parameters
    ----------
    name
        Domain stem, e.g. ``"adverse"`` (case-insensitive, no extension).
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    path = data_dir / f"{name.lower()}.sas7bdat"
    if not path.exists():
        raise FileNotFoundError(f"SAS domain not found: {path}")

    df = pd.read_sas(path, encoding="latin-1")
    df = _decode_bytes(df)
    if "PID_A" in df.columns:
        df.insert(0, "PID", normalise_pid(df["PID_A"]))
    df.attrs["domain"] = name.lower()
    return df


def available_domains(data_dir: Path | None = None) -> dict[str, bool]:
    """Report which core/optional domains are physically present in ``data-files``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    status: dict[str, bool] = {}
    for d in (*CORE_DOMAINS, *OPTIONAL_DOMAINS):
        status[d] = (data_dir / f"{d}.sas7bdat").exists()
    return status


def load_all_domains(data_dir: Path | None = None,
                     domains: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every available domain into a ``{name: DataFrame}`` dictionary."""
    domains = tuple(domains) if domains is not None else (*CORE_DOMAINS, *OPTIONAL_DOMAINS)
    out: dict[str, pd.DataFrame] = {}
    for d in domains:
        try:
            out[d] = read_domain(d, data_dir)
        except FileNotFoundError:
            continue
    return out


def load_specs(path: Path | None = None) -> pd.DataFrame:
    """Load the study data dictionary (``A6181122_SPECS.xlsx``)."""
    path = Path(path) if path is not None else SPECS_FILE
    specs = pd.read_excel(path)
    specs = specs.rename(columns={specs.columns[0]: "DOMAIN"})
    specs["DOMAIN"] = specs["DOMAIN"].ffill().str.strip()
    return specs


# --------------------------------------------------------------------------------------
# Domain-specific tidying
# --------------------------------------------------------------------------------------
def tidy_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    """Coerce character vital signs to numeric and harmonise temperature units.

    Returns a long-to-wide tidy frame with one row per (PID, visit day) and
    columns ``sbp, dbp, hr, rr, temp, weight`` plus ``bmi`` when height is known.
    """
    v = vitals.copy()
    v = v[v.get("VSND", "DONE").fillna("DONE").str.upper() != "NOT DONE"]

    for src, dst in VITALS_NUMERIC.items():
        v[dst] = pd.to_numeric(v[src], errors="coerce") if src in v.columns else np.nan

    # A handful of sites recorded temperature in degrees Fahrenheit.
    if "TMPUNIT" in v.columns:
        is_f = v["TMPUNIT"].str.upper().str.contains("FAHRENHEIT", na=False)
        v.loc[is_f, "temp"] = (v.loc[is_f, "temp"] - 32.0) * 5.0 / 9.0
    # Defensive: anything still above 45 C is a mis-keyed Fahrenheit value.
    v.loc[v["temp"] > 45, "temp"] = (v.loc[v["temp"] > 45, "temp"] - 32.0) * 5.0 / 9.0

    v = v.rename(columns={"COLLDAY": "day"})
    keep = ["PID", "day", "CPEVENT", *VITALS_NUMERIC.values()]
    v = v[[c for c in keep if c in v.columns]].dropna(subset=["day"])

    # One row per patient-day: average duplicate assessments on the same day.
    num_cols = [c for c in VITALS_NUMERIC.values() if c in v.columns]
    v = (v.groupby(["PID", "day"], as_index=False)
           .agg({**{c: "mean" for c in num_cols}, "CPEVENT": "first"}))
    return v.sort_values(["PID", "day"]).reset_index(drop=True)


def tidy_adverse(adverse: pd.DataFrame) -> pd.DataFrame:
    """Standardise the adverse-event domain.

    Adds
    ----
    ``start_day``/``end_day``  : numeric study days
    ``grade``                  : CTCAE grade (1-5)
    ``is_severe``              : grade >= :data:`config.CFG.severity_threshold`
    ``is_serious``             : regulatory seriousness flag
    ``dose_action``            : action taken on study treatment
    ``led_to_discontinuation`` / ``led_to_reduction`` / ``led_to_interruption``
    ``pt``                     : MedDRA preferred term (title case)
    """
    a = adverse.copy()
    a["start_day"] = pd.to_numeric(a["AEFDAY"], errors="coerce")
    a["end_day"] = pd.to_numeric(a["AETDAY"], errors="coerce")
    a["grade"] = pd.to_numeric(a["AEGRADE"], errors="coerce")
    a["is_severe"] = a["grade"] >= CFG.severity_threshold
    a["is_serious"] = a["AESERC"].str.upper().eq("YES")
    a["is_related"] = a.get("AERCAUSC", pd.Series(index=a.index, dtype=object)).str.upper().eq("YES")
    a["still_present"] = a.get("AEPRESC", pd.Series(index=a.index, dtype=object)).str.upper().eq("YES")
    a["pt"] = a["PREFTEXT"].str.strip()

    # Action taken on the two study treatment components.
    act1 = a.get("AEST1TRC", pd.Series(index=a.index, dtype=object)).fillna("")
    act2 = a.get("AEST2TRC", pd.Series(index=a.index, dtype=object)).fillna("")
    combined = (act1 + "|" + act2).str.upper()
    a["led_to_discontinuation"] = combined.str.contains("PERMANENTLY DISCONTINUED")
    a["led_to_reduction"] = combined.str.contains("REDUCED")
    a["led_to_interruption"] = combined.str.contains("STOPPED TEMPORARILY")
    a["dose_action"] = np.select(
        [a["led_to_discontinuation"], a["led_to_reduction"], a["led_to_interruption"]],
        ["discontinued", "reduced", "interrupted"],
        default="none",
    )
    return a.sort_values(["PID", "start_day"]).reset_index(drop=True)


def tidy_testdrug(testdrug: pd.DataFrame) -> pd.DataFrame:
    """Standardise study-treatment administration records.

    Adds ``drug_group`` (fluorouracil / irinotecan / leucovorin / blinded),
    numeric ``dose``, ``start_day``/``stop_day`` and the parsed ``cycle`` number.
    """
    t = testdrug.copy()
    t["dose"] = pd.to_numeric(t["DOSTOT"], errors="coerce")
    t["start_day"] = pd.to_numeric(t["FROMDAY"], errors="coerce")
    t["stop_day"] = pd.to_numeric(t["TODAY"], errors="coerce")

    lut = {name: group for group, names in DRUG_GROUPS.items() for name in names}
    t["drug_group"] = t["DRGNAME"].map(lut).fillna("other")

    cyc = t["CPEVENT"].str.extract(r"CYCLE(\d+)", expand=False)
    t["cycle"] = pd.to_numeric(cyc, errors="coerce")
    day_in_cycle = t["CPEVENT"].str.extract(r"DAY(\d+)", expand=False)
    t["cycle_day"] = pd.to_numeric(day_in_cycle, errors="coerce")

    return t.sort_values(["PID", "start_day"]).reset_index(drop=True)


def _ctcae_grade(code: str, value: pd.Series, lln: pd.Series,
                 uln: pd.Series) -> pd.Series:
    """CTCAE v3.0 grade for one analyte, using each record's own reference range.

    Decreases and increases are graded independently and the worse of the two is
    returned, so a single column covers analytes that are toxic in both directions
    (sodium, potassium, calcium). Records without the relevant limit of normal are
    graded only on the absolute thresholds, and are left missing if neither applies.
    """
    grade = pd.Series(np.nan, index=value.index, dtype="float64")
    gradeable = (code in LAB_CTCAE_LOW or code in LAB_CTCAE_HIGH_MULT
                 or code in LAB_CTCAE_HIGH_ABS)
    known = value.notna()
    if not gradeable or not known.any():
        return grade
    grade[known] = 0.0

    # ---- decreases ------------------------------------------------------------------
    if code in LAB_CTCAE_LOW:
        g2, g3, g4 = LAB_CTCAE_LOW[code]
        low = pd.Series(0.0, index=value.index)
        low[known & lln.notna() & (value < lln)] = 1.0
        low[known & (value < g2)] = np.fmax(low[known & (value < g2)], 2.0)
        low[known & (value < g3)] = np.fmax(low[known & (value < g3)], 3.0)
        low[known & (value < g4)] = np.fmax(low[known & (value < g4)], 4.0)
        grade[known] = np.fmax(grade[known], low[known])

    # ---- increases, as multiples of the upper limit of normal ------------------------
    if code in LAB_CTCAE_HIGH_MULT:
        m2, m3, m4 = LAB_CTCAE_HIGH_MULT[code]
        ratio = value / uln.replace(0.0, np.nan)
        ok = known & ratio.notna()
        high = pd.Series(0.0, index=value.index)
        high[ok & (ratio > 1.0)] = 1.0
        high[ok & (ratio > m2)] = 2.0
        high[ok & (ratio > m3)] = 3.0
        high[ok & (ratio > m4)] = 4.0
        grade[ok] = np.fmax(grade[ok], high[ok])

    # ---- increases, on absolute thresholds -------------------------------------------
    if code in LAB_CTCAE_HIGH_ABS:
        a2, a3, a4 = LAB_CTCAE_HIGH_ABS[code]
        high = pd.Series(0.0, index=value.index)
        high[known & uln.notna() & (value > uln)] = 1.0
        high[known & (value > a2)] = np.fmax(high[known & (value > a2)], 2.0)
        high[known & (value > a3)] = np.fmax(high[known & (value > a3)], 3.0)
        high[known & (value > a4)] = np.fmax(high[known & (value > a4)], 4.0)
        grade[known] = np.fmax(grade[known], high[known])

    return grade


def tidy_lab(lab: pd.DataFrame) -> pd.DataFrame:
    """Standardise the safety laboratory domain into one tidy row per result.

    ``LABVALUE`` is a character field mixing numbers with qualitative results
    (``NEGATIVE``) and explicit non-measurement markers (``ND``, ``NOT DONE``); the
    latter must become missing rather than zero, which is why the parse is explicit
    rather than a bare :func:`pandas.to_numeric`.

    Adds
    ----
    ``analyte``    short analyte code (see :data:`config.LAB_ANALYTES`)
    ``day``        study day of collection (``COLLDAY``)
    ``value``      numeric result, unit-repaired for the CTCAE-graded analytes
    ``lln``/``uln``  the record's own reference range
    ``rel``        range-relative level ``(value - lln) / (uln - lln)``: 0 at the lower
                   limit of normal, 1 at the upper limit, dimensionless
    ``ratio_uln``  ``value / uln`` - the scale CTCAE uses for hepatic and renal toxicity
    ``ratio_lln``  ``value / lln`` - the scale that matters for the cytopenias
    ``ctcae``      CTCAE v3.0 grade (0-4) where the analyte is gradeable

    Only analytes listed in :data:`config.LAB_ANALYTES` are retained; the urinalysis
    dipstick results are dropped because they are almost entirely qualitative.
    """
    d = lab.copy()
    d["analyte"] = d["LBTEST"].str.strip().str.upper().map(LAB_ANALYTES)
    d = d[d["analyte"].notna()].copy()

    d["day"] = pd.to_numeric(d["COLLDAY"], errors="coerce")

    raw = d["LABVALUE"].astype("string").str.strip()
    token = raw.str.upper().str.rstrip(".")
    # Comparator-prefixed results ("<0.1", ">1000") keep the reported magnitude.
    cleaned = raw.str.replace(r"^[<>=~]\s*", "", regex=True).str.replace(",", ".", regex=False)
    value = pd.to_numeric(cleaned, errors="coerce")
    value[token.isin(LAB_MISSING_TOKENS)] = np.nan
    d["value"] = value

    d["lln"] = pd.to_numeric(d["MIN_NORM"], errors="coerce")
    d["uln"] = pd.to_numeric(d["MAX_NORM"], errors="coerce")

    # Repair the handful of records reported in the alternative unit. The reference
    # range travels with the result, so it is rescaled by the same factor.
    for code, (above, divisor) in LAB_UNIT_FIX.items():
        hit = d["analyte"].eq(code) & d["value"].gt(above)
        if hit.any():
            d.loc[hit, "value"] = d.loc[hit, "value"] / divisor
            for col in ("lln", "uln"):
                rescale = hit & d[col].gt(above)
                d.loc[rescale, col] = d.loc[rescale, col] / divisor

    # Transcription errors: physiologically impossible results become missing.
    for code, (lo, hi) in LAB_PLAUSIBLE.items():
        bad = d["analyte"].eq(code) & d["value"].notna() & (
            (d["value"] < lo) | (d["value"] > hi))
        d.loc[bad, "value"] = np.nan

    span = (d["uln"] - d["lln"]).replace(0.0, np.nan)
    d["rel"] = (d["value"] - d["lln"]) / span
    d["ratio_uln"] = d["value"] / d["uln"].replace(0.0, np.nan)
    d["ratio_lln"] = d["value"] / d["lln"].replace(0.0, np.nan)

    # Analytes collected without a reference range cannot have a range-relative level;
    # a log level keeps them on a comparable scale for the tree ensembles.
    for code in LAB_NO_REFERENCE_RANGE:
        hit = d["analyte"].eq(code) & d["value"].notna()
        d.loc[hit, "rel"] = np.log10(d.loc[hit, "value"].clip(lower=0.0) + 1.0)

    d["ctcae"] = np.nan
    for code, sub in d.groupby("analyte", sort=False):
        d.loc[sub.index, "ctcae"] = _ctcae_grade(
            code, sub["value"], sub["lln"], sub["uln"])

    keep = ["PID", "day", "analyte", "value", "lln", "uln",
            "rel", "ratio_uln", "ratio_lln", "ctcae", "CPEVENT"]
    d = d[[c for c in keep if c in d.columns]].dropna(subset=["day"])
    return d.sort_values(["PID", "day", "analyte"]).reset_index(drop=True)


def lab_wide(lab_tidy: pd.DataFrame, value_col: str = "rel") -> pd.DataFrame:
    """Pivot :func:`tidy_lab` output to one row per (patient, collection day).

    Repeat measurements of the same analyte on the same day - a re-draw after a
    haemolysed sample, typically - are averaged, except for the CTCAE grade where the
    worst value of the day is carried forward.
    """
    d = lab_tidy.dropna(subset=[value_col])
    how = "max" if value_col == "ctcae" else "mean"
    wide = (d.pivot_table(index=["PID", "day"], columns="analyte",
                          values=value_col, aggfunc=how)
             .reset_index())
    wide.columns.name = None
    return wide.sort_values(["PID", "day"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Disease burden, response and prior therapy
# --------------------------------------------------------------------------------------
def tidy_tumour(tmm: pd.DataFrame) -> pd.DataFrame:
    """Collapse target-lesion measurements into one tumour-burden row per assessment.

    The RECIST summary of burden is the **sum of longest diameters** of the target
    lesions, so the per-lesion rows are aggregated to one ``sld`` per (patient, day).
    Non-target lesions carry no measurement and contribute only through the presence of
    new lesions.

    Columns
    -------
    ``day``               study day of the assessment
    ``sld``               sum of longest diameters, mm
    ``n_target_lesions``  target lesions measured at that assessment
    ``n_new_lesions``     lesions flagged as new
    ``site_*``            indicator for each site in :data:`config.TUMOUR_SITES`
    """
    t = tmm.copy()
    t["day"] = pd.to_numeric(t["EFDAY"], errors="coerce")
    t["dia"] = pd.to_numeric(t["TMMDIA"], errors="coerce")
    t = t.dropna(subset=["day"])

    target = t[t["LESTYPE"].astype("string").str.strip().str.lower().eq("target")]
    measured = target.dropna(subset=["dia"])

    burden = (measured.groupby(["PID", "day"])
                      .agg(sld=("dia", "sum"), n_target_lesions=("dia", "size"))
                      .reset_index())

    new_les = (t.assign(_new=t["TMMNLES"].notna().astype(float))
                 .groupby(["PID", "day"])["_new"].sum()
                 .rename("n_new_lesions").reset_index())
    out = burden.merge(new_les, on=["PID", "day"], how="left")
    out["n_new_lesions"] = out["n_new_lesions"].fillna(0.0)

    site = t["TMMDIS"].astype("string").str.upper().fillna("")
    for s in TUMOUR_SITES:
        flag = (t.assign(_s=site.str.contains(s, na=False).astype(float))
                  .groupby(["PID", "day"])["_s"].max()
                  .rename(f"site_{s.lower()}").reset_index())
        out = out.merge(flag, on=["PID", "day"], how="left")

    n_sites = (t.dropna(subset=["TMMDIS"]).groupby(["PID", "day"])["TMMDIS"].nunique()
                 .rename("n_disease_sites").reset_index())
    out = out.merge(n_sites, on=["PID", "day"], how="left")

    return out.sort_values(["PID", "day"]).reset_index(drop=True)


def tidy_response(iota: pd.DataFrame) -> pd.DataFrame:
    """Standardise the overall RECIST assessment into an ordinal response per visit.

    ``response`` runs 0 (complete response) to 3 (progressive disease) via
    :data:`config.RECIST_ORDER`; assessments recorded as indeterminate or not assessed
    become missing rather than being forced onto the scale.
    """
    r = iota.copy()
    r["day"] = pd.to_numeric(r["EFDAY"], errors="coerce")
    r = r.dropna(subset=["day"])

    overall = r["IOTAALL"].astype("string").str.strip().str.upper()
    r["response"] = overall.map(RECIST_ORDER)
    r["is_progression"] = overall.eq("PROGRESSIVE DISEASE").astype(float)
    r["new_lesion"] = r["IOTANL"].astype("string").str.strip().str.upper().eq("YES").astype(float)

    keep = ["PID", "day", "response", "is_progression", "new_lesion"]
    return (r[keep].sort_values(["PID", "day"])
                   .groupby(["PID", "day"], as_index=False)
                   .agg({"response": "max", "is_progression": "max", "new_lesion": "max"}))


def tidy_prior_therapy(cd_b_p: pd.DataFrame) -> pd.DataFrame:
    """Prior systemic anti-cancer therapy, one row per patient.

    Note on completeness: ``CMDECOD`` (the agent) and ``ONCBRP`` (best response to the
    prior regimen) are **empty for every record** in this release, so which drugs a patient
    previously received, and how well they worked, cannot be recovered. What is usable is
    the number of prior regimens, the treatment setting, why the last one stopped, and
    when. The reason for stopping is the field that matters most here: a patient whose
    previous therapy was abandoned for toxicity is the archetype an enrolment check exists
    to notice.
    """
    c = cd_b_p.copy()
    c["reg"] = pd.to_numeric(c["CDRGRNM"], errors="coerce")
    c["from_day"] = pd.to_numeric(c["CDBPFDAY"], errors="coerce")
    c["to_day"] = pd.to_numeric(c["CDBPTDAY"], errors="coerce")

    setting = c["ONCCTYP"].astype("string").str.upper()
    reason = c["ONCSPRS"].astype("string").str.upper()

    out = pd.DataFrame(index=pd.Index(sorted(c["PID"].unique()), name="PID"))
    out["n_prior_regimens"] = c.groupby("PID")["reg"].max()
    out["prior_adjuvant"] = (c.assign(v=setting.eq("ADJUVANT").astype(float))
                               .groupby("PID")["v"].max())
    out["prior_neoadjuvant"] = (c.assign(v=setting.eq("NEOADJUVANT").astype(float))
                                  .groupby("PID")["v"].max())
    out["prior_stopped_for_toxicity"] = (c.assign(v=reason.eq("TOXICITY").astype(float))
                                           .groupby("PID")["v"].max())
    out["prior_stopped_for_progression"] = (c.assign(v=reason.eq("PROGRESSION").astype(float))
                                              .groupby("PID")["v"].max())
    out["prior_therapy_end_day"] = c.groupby("PID")["to_day"].max()
    return out.reset_index()


def tidy_radiotherapy(cn_6_p: pd.DataFrame) -> pd.DataFrame:
    """Prior radiotherapy, one row per patient.

    ``NONEF`` is the study's own "none recorded" marker, so it - rather than the
    missingness of the detail fields - determines whether a patient had radiotherapy.
    Pelvic and abdominal fields are separated out because they irradiate marrow-bearing
    bone; see :data:`config.PELVIC_RT_PATTERN`.
    """
    r = cn_6_p.copy()
    r["from_day"] = pd.to_numeric(r["CN6PFDAY"], errors="coerce")
    r["to_day"] = pd.to_numeric(r["CN6PTDAY"], errors="coerce")

    had = r["NONEF"].astype("string").str.strip().str.upper().eq("SOME").astype(float)
    site = r["RADSITE"].astype("string").str.upper().fillna("")
    pelvic = site.str.contains(PELVIC_RT_PATTERN, regex=True, na=False).astype(float)

    out = pd.DataFrame(index=pd.Index(sorted(r["PID"].unique()), name="PID"))
    out["prior_radiotherapy"] = r.assign(v=had).groupby("PID")["v"].max()
    out["prior_pelvic_radiotherapy"] = r.assign(v=pelvic).groupby("PID")["v"].max()
    out["radiotherapy_end_day"] = r.groupby("PID")["to_day"].max()
    return out.reset_index()


# --------------------------------------------------------------------------------------
# Patient timeline
# --------------------------------------------------------------------------------------
def build_patient_timeline(domains: dict[str, pd.DataFrame],
                           cfg=CFG) -> pd.DataFrame:
    """Derive the per-patient observation window used for labelling and risk sets.

    Columns
    -------
    ``first_dose_day``   first recorded administration of study treatment
    ``last_dose_day``    last recorded administration
    ``eot_day``          end-of-treatment day (``FINAL``, ``CPEVENT='END_OF_TREATMENT'``)
    ``eos_day``          end-of-study day (``FINAL``, ``CPEVENT='END_OF_STUDY'``)
    ``ae_obs_end``       last day on which an adverse event could still be captured
    ``on_treatment_end`` last day the patient is considered 'in trial' for prediction
    ``died``             disposition indicates death
    ``discont_reason``   end-of-treatment withdrawal reason

    ``ae_obs_end`` is ``min(eos_day, last_dose_day + ae_reporting_tail_days)`` and is
    extended when a patient has adverse events recorded beyond that point, so that no
    observed event is silently discarded.
    """
    td = tidy_testdrug(domains["testdrug"])
    fin = domains["final"].copy()
    ae = tidy_adverse(domains["adverse"])

    dose = (td.dropna(subset=["start_day"])
              .groupby("PID")
              .agg(first_dose_day=("start_day", "min"),
                   last_dose_day=("stop_day", "max"),
                   n_dose_records=("start_day", "size"),
                   max_cycle=("cycle", "max")))
    dose["last_dose_day"] = dose["last_dose_day"].fillna(dose["first_dose_day"])

    fin["WDRLDAY"] = pd.to_numeric(fin["WDRLDAY"], errors="coerce")
    eot = (fin[fin["CPEVENT"].eq("END_OF_TREATMENT")]
           .groupby("PID")
           .agg(eot_day=("WDRLDAY", "max"),
                discont_reason=("FINSTATC", "first")))
    eos = (fin[fin["CPEVENT"].eq("END_OF_STUDY")]
           .groupby("PID")
           .agg(eos_day=("WDRLDAY", "max"),
                final_status=("FINSTATC", "first")))

    ae_extent = ae.groupby("PID").agg(
        last_ae_day=("start_day", "max"),
        first_ae_day=("start_day", "min"),
        n_ae_total=("start_day", "size"),
    )

    tl = (dose.join(eot, how="outer").join(eos, how="outer").join(ae_extent, how="outer"))
    tl.index.name = "PID"
    tl = tl.reset_index()

    tl["first_dose_day"] = tl["first_dose_day"].fillna(1.0)
    tl["eot_day"] = tl["eot_day"].fillna(tl["last_dose_day"])
    tl["eos_day"] = tl["eos_day"].fillna(tl["eot_day"])

    # Adverse-event capture window: last dose + protocol reporting tail, bounded by
    # the end-of-study visit, then extended if events were actually reported later.
    tail_end = tl["last_dose_day"].fillna(tl["eot_day"]) + cfg.ae_reporting_tail_days
    tl["ae_obs_end"] = np.fmin(tl["eos_day"], tail_end)
    tl["ae_obs_end"] = np.fmax(tl["ae_obs_end"], tl["last_ae_day"].fillna(-np.inf))
    tl["ae_obs_end"] = tl["ae_obs_end"].fillna(tl["eos_day"])

    # A prediction is only meaningful while the patient is still in trial treatment.
    tl["on_treatment_end"] = np.fmin(tl["eot_day"], tl["ae_obs_end"])

    tl["died"] = tl["final_status"].fillna("").str.upper().str.contains("DIED")
    tl["discont_due_to_ae"] = tl["discont_reason"].fillna("").str.upper().eq("ADVERSE EVENT")
    tl["treatment_duration"] = tl["eot_day"] - tl["first_dose_day"]
    tl["study_duration"] = tl["eos_day"] - tl["first_dose_day"]

    return tl.sort_values("PID").reset_index(drop=True)
