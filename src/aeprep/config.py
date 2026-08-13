"""Central configuration: paths, study conventions and analysis parameters.

Every analytical choice that a reviewer might want to challenge (prediction
horizon, landmark spacing, look-back windows, severity threshold) lives here so
that sensitivity analyses can be run by changing a single value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# src/aeprep/config.py -> src/aeprep -> src -> project-root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data-files"
DOCS_DIR: Path = PROJECT_ROOT / "requirements"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

DATASET_DIR: Path = OUTPUT_DIR / "datasets"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
TABLE_DIR: Path = OUTPUT_DIR / "tables"
MODEL_DIR: Path = OUTPUT_DIR / "models"

SPECS_FILE: Path = DATA_DIR / "A6181122_SPECS.xlsx"

#: SAS transport files shipped by Project Data Sphere for study A6181122.
#: ``optional`` domains are declared in the SPECS workbook but are not part of the
#: current download; the pipeline degrades gracefully when they are absent.
CORE_DOMAINS: tuple[str, ...] = (
    "adverse",     # adverse events (CTCAE grade, start/stop day, action taken)
    "demog",       # demographics
    "testdrug",    # study treatment administration / dosing
    "vitals",      # vital signs (longitudinal)
    "pfm_p",       # ECOG performance status (longitudinal)
    "final",       # disposition / end of treatment / end of study
    "random",      # randomisation day
    "primdiag",    # primary diagnosis
    "prevdis",     # medical history / previous diseases
    "condrug",     # concomitant medication
    "contrt",      # concomitant non-drug treatment
    "phyexam",     # physical examination
)

OPTIONAL_DOMAINS: tuple[str, ...] = (
    "lab_safe",    # safety laboratory results - added to the download in round two
    "iota_p",      # overall RECIST tumour assessment (longitudinal) - round three
    "tmm_p",       # target lesion measurements (longitudinal) - round three
    "cd_b_p",      # prior systemic anti-cancer therapy - round three
    "cn_6_p",      # prior radiotherapy - round three
    "ecg",         # documented in SPECS, still not downloaded
)


def ensure_output_dirs() -> None:
    """Create the output folder tree (idempotent)."""
    for d in (OUTPUT_DIR, DATASET_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisConfig:
    """Study-design parameters for the prediction problem.

    Defaults reproduce the design registered in the mid-semester report:
    *any treatment-emergent CTCAE Grade >= 3 adverse event within the next 30 days*.
    """

    # ---- target definition -------------------------------------------------------
    severity_threshold: int = 3
    """CTCAE grade at or above which an adverse event counts as 'severe'."""

    prediction_horizon_days: int = 30
    """Length of the forward prediction window (the 'next 30 days')."""

    # ---- landmark (temporal) design ----------------------------------------------
    landmark_start_day: int = 0
    landmark_step_days: int = 14
    """FOLFIRI is administered on a 14-day rhythm, so landmarks are placed every 2 weeks."""

    landmark_max_day: int = 336
    """Latest landmark: 24 landmarks at 14-day spacing, i.e. 8 six-week protocol cycles.
    Beyond this fewer than 80 of the 379 patients remain on treatment."""

    require_complete_window: bool = True
    """Drop landmark rows whose 30-day window is not fully observed and event-free
    (administrative censoring). Rows where the event *did* occur are always kept."""

    # ---- look-back windows for time-varying features -----------------------------
    short_lookback_days: int = 28
    long_lookback_days: int = 56

    # ---- observation-window conventions ------------------------------------------
    ae_reporting_tail_days: int = 28
    """Adverse events are collected up to N days after the last dose of study drug."""

    # ---- modelling ---------------------------------------------------------------
    random_state: int = 42
    n_splits: int = 5
    test_size: float = 0.30
    """Fraction of *patients* held out for the final (grouped) test evaluation."""

    n_bootstrap: int = 500
    """Bootstrap replicates used for confidence intervals on test metrics."""

    # ---- evaluation landmarks used for the 'risk updating' figure ----------------
    reporting_landmarks: Sequence[int] = field(default_factory=lambda: (0, 42, 84, 126, 168))

    def as_dict(self) -> dict:
        return asdict(self)


CFG = AnalysisConfig()


# --------------------------------------------------------------------------------------
# Clinical groupings used for feature engineering
# --------------------------------------------------------------------------------------
#: MedDRA preferred terms grouped into clinically meaningful toxicity clusters.
#: Matching is case-insensitive substring matching on ``PREFTEXT``.
TOXICITY_CLUSTERS: dict[str, tuple[str, ...]] = {
    "haematologic": (
        "neutropenia", "leukopenia", "anaemia", "anemia", "thrombocytopenia",
        "lymphopenia", "pancytopenia", "neutrophil count decreased",
        "white blood cell count decreased", "platelet count decreased",
        "haemoglobin decreased", "febrile neutropenia",
    ),
    "gastrointestinal": (
        "diarrhoea", "diarrhea", "nausea", "vomiting", "stomatitis", "mucosal inflammation",
        "abdominal pain", "constipation", "ileus", "intestinal obstruction", "dyspepsia",
    ),
    "constitutional": (
        "fatigue", "asthenia", "pyrexia", "anorexia", "decreased appetite",
        "weight decreased", "malaise", "dehydration",
    ),
    "hepatobiliary": (
        "alanine aminotransferase", "aspartate aminotransferase", "bilirubin",
        "hepatic", "transaminases",
    ),
    "vascular_thrombotic": (
        "pulmonary embolism", "deep vein thrombosis", "thrombosis", "hypertension",
        "haemorrhage", "hemorrhage", "epistaxis",
    ),
    "infection": (
        "infection", "sepsis", "pneumonia", "cellulitis", "urinary tract",
    ),
    "metabolic": (
        "hypokalaemia", "hyponatraemia", "hyperglycaemia", "hypophosphataemia",
        "hypomagnesaemia", "hypocalcaemia", "hypoalbuminaemia",
    ),
}

#: Supportive-care concomitant medication classes (keyword match on the WHO-Drug
#: preferred term or the free-text reason for administration).
CONMED_CLASSES: dict[str, tuple[str, ...]] = {
    "growth_factor": ("filgrastim", "pegfilgrastim", "lenograstim", "neupogen",
                      "neulasta", "granocyte", "epoetin", "darbepoetin", "g-csf", "gcsf"),
    "antiemetic": ("ondansetron", "granisetron", "metoclopramide", "prochlorperazine",
                   "aprepitant", "domperidone", "dexamethasone", "decadron", "tropisetron"),
    "antidiarrhoeal": ("loperamide", "imodium", "octreotide", "diphenoxylate"),
    "antibiotic": ("ciprofloxacin", "amoxicillin", "ceftriaxone", "levofloxacin",
                   "metronidazole", "cefuroxime", "penicillin", "vancomycin", "meropenem"),
    "anticoagulant": ("enoxaparin", "warfarin", "heparin", "fragmin", "dalteparin",
                      "tinzaparin", "clexane"),
    "analgesic_opioid": ("morphine", "oxycodone", "fentanyl", "tramadol", "codeine"),
}

#: Study treatment components (``TESTDRUG.DRGNAME``). ``Folinic Acid/Vitamin K`` and
#: ``Calcium Folinate`` are two recordings of the same leucovorin component.
DRUG_GROUPS: dict[str, tuple[str, ...]] = {
    "fluorouracil": ("Fluorouracil",),
    "irinotecan": ("Irinotecan",),
    "leucovorin": ("Folinic Acid/Vitamin K", "Calcium Folinate"),
    "blinded": ("Placebo",),
}

#: Vital-sign columns in ``VITALS`` that are stored as character and must be coerced.
VITALS_NUMERIC: dict[str, str] = {
    "SYSBPF": "sbp",
    "DIABPF": "dbp",
    "HEARTF": "hr",
    "RESPF": "rr",
    "TEMPF": "temp",
    "WTF": "weight",
}


# --------------------------------------------------------------------------------------
# Safety laboratory panel (LAB_SAFE)
# --------------------------------------------------------------------------------------
# LAB_SAFE arrived after the first analysis round and closes the study's single largest
# gap: neutropenia alone accounts for 210 of the 655 Grade >= 3 events, and an absolute
# neutrophil count is the quantity that defines it.
#
# Representation strategy
# -----------------------
# Every record carries its own laboratory reference range (``MIN_NORM``/``MAX_NORM``) in
# the same units as ``LABVALUE``. The primary feature is therefore the *range-relative
# level*
#
#     rel = (value - LLN) / (ULN - LLN)
#
# which is 0 at the lower limit of normal, 1 at the upper limit, and dimensionless. This
# neutralises both the unit heterogeneity and the site-to-site variation in reference
# ranges without committing to a single conversion table. Absolute values are still
# harmonised (below) because CTCAE grading is defined on them.

#: ``LBTEST`` verbatim -> short analyte code used in feature names.
LAB_ANALYTES: dict[str, str] = {
    "HEMOGLOBIN": "hgb",
    "WHITE BLOOD CELLS": "wbc",
    "NEUTROPHILS (ABSOLUTE)": "anc",
    "PLATELETS": "plt",
    "LYMPHOCYTES (ABSOLUTE)": "lym",
    "MONOCYTES (ABSOLUTE)": "mono",
    "EOSINOPHILS (ABSOLUTE)": "eos",
    "ALANINE AMINOTRANSFERASE (ALT)": "alt",
    "ASPARTATE AMINOTRANSFERASE (AST)": "ast",
    "ALKALINE PHOSPHATASE": "alp",
    "BILIRUBIN (TOTAL)": "bili",
    "ALBUMIN": "alb",
    "PROTEIN (TOTAL)": "tprot",
    "CREATININE": "creat",
    "BLOOD UREA NITROGEN": "bun",
    "SODIUM": "na",
    "POTASSIUM": "k",
    "CHLORIDE": "cl",
    "CALCIUM": "ca",
    "PHOSPHATE": "phos",
    "GLUCOSE": "gluc",
    "URIC ACID": "urate",
    "LACTATE DEHYDROGENASE": "ldh",
    "CARCINOEMBRYONIC ANTIGEN (CEA)": "cea",
    "PROTHROMBIN TIME INR": "inr",
}

#: Analytes that receive the full temporal treatment (latest level, 56-day nadir and
#: peak, and trend). Chosen for completeness (>= 6,000 records) and for a documented
#: mechanistic link to FOLFIRI toxicity.
LAB_CORE: tuple[str, ...] = (
    "anc", "wbc", "plt", "hgb", "lym",
    "alt", "ast", "alp", "bili", "creat", "alb",
    "na", "k", "ca", "gluc", "ldh",
)

#: Analytes summarised by their latest level only - lower completeness or a weaker
#: prior link to short-term severe toxicity.
LAB_SECONDARY: tuple[str, ...] = (
    "mono", "eos", "tprot", "bun", "cl", "phos", "urate", "cea", "inr",
)

#: Analytes for which a change from the pre-treatment baseline is also computed.
LAB_DELTA_BASELINE: tuple[str, ...] = (
    "anc", "wbc", "plt", "hgb", "alt", "ast", "bili", "creat", "alb",
)

#: ``LABVALUE`` is a character field. These tokens are qualitative results or explicit
#: "not measured" markers and must become missing rather than zero.
LAB_MISSING_TOKENS: frozenset[str] = frozenset({
    "ND", "N/D", "N.D", "N.D.", "NOT DONE", "NOTDONE", "NA", "N/A", "N.A", "N.A.",
    "NOT FOUND", "NOT DETECTED", "NOT APPLICABLE", "NOT ASSESSED", "-", "--", ".",
    "NEGATIVE", "NEG", "NEG.", "NEGATIV", "NEGAT", "POSITIVE", "POS", "NORMAL",
    "ABNORMAL", "ABSENT", "PRESENT", "NIL", "NONE", "TRACE", "CLEAR", "YELLOW",
})

#: Stray-unit repair: ``code -> (above_this_value, divide_by)``. The sponsor reported
#: each test in a single unit for essentially every parseable record; a small number of
#: rows carry the alternative unit and are recognisable by magnitude alone. Applied only
#: to the analytes that CTCAE grades on absolute values.
LAB_UNIT_FIX: dict[str, tuple[float, float]] = {
    "anc": (100.0, 1000.0),      # cells/mm3            -> 10^9/L
    "lym": (100.0, 1000.0),      # cells/mm3            -> 10^9/L
    "mono": (100.0, 1000.0),
    "eos": (100.0, 1000.0),
    "wbc": (200.0, 1000.0),      # cells/mm3            -> 10^9/L
    "plt": (3000.0, 1000.0),     # cells/mm3            -> 10^9/L
    "hgb": (25.0, 10.0),         # g/L                  -> g/dL
    "bili": (50.0, 17.1),        # umol/L               -> mg/dL
    "creat": (20.0, 88.4),       # umol/L               -> mg/dL
    "alb": (15.0, 10.0),         # g/L                  -> g/dL
}

#: CTCAE v3.0 grading for *decreases*, as ``code -> (g2, g3, g4)`` absolute thresholds.
#: A value below a threshold earns at least that grade; a value merely below the
#: record's own lower limit of normal earns Grade 1. Where CTCAE defines no Grade 2
#: (sodium, potassium) the Grade 2 and Grade 3 bounds coincide, so no row is ever
#: assigned the non-existent grade.
LAB_CTCAE_LOW: dict[str, tuple[float, float, float]] = {
    "anc": (1.5, 1.0, 0.5),          # 10^9/L
    "wbc": (3.0, 2.0, 1.0),          # 10^9/L
    "plt": (75.0, 50.0, 25.0),       # 10^9/L
    "hgb": (10.0, 8.0, 6.5),         # g/dL
    "lym": (0.8, 0.5, 0.2),          # 10^9/L
    "alb": (3.0, 2.0, -1.0),         # g/dL - CTCAE defines no Grade 4
    "na": (130.0, 130.0, 120.0),     # mEq/L - no Grade 2
    "k": (3.0, 3.0, 2.5),            # mEq/L - no Grade 2
    "ca": (8.0, 7.0, 6.0),           # mg/dL
}

#: CTCAE v3.0 grading for *increases* expressed as multiples of the record's own upper
#: limit of normal, ``code -> (g2, g3, g4)``. Grade 1 is any value above the ULN.
LAB_CTCAE_HIGH_MULT: dict[str, tuple[float, float, float]] = {
    "alt": (2.5, 5.0, 20.0),
    "ast": (2.5, 5.0, 20.0),
    "alp": (2.5, 5.0, 20.0),
    "bili": (1.5, 3.0, 10.0),
    "creat": (1.5, 3.0, 6.0),
}

#: CTCAE v3.0 grading for *increases* on absolute thresholds, ``code -> (g2, g3, g4)``.
LAB_CTCAE_HIGH_ABS: dict[str, tuple[float, float, float]] = {
    "k": (5.5, 6.0, 7.0),            # mEq/L
    "na": (150.0, 155.0, 160.0),     # mEq/L
    "ca": (11.5, 12.5, 13.5),        # mg/dL
    "gluc": (160.0, 250.0, 500.0),   # mg/dL
}

#: Analytes carrying a CTCAE grade feature at the landmark.
LAB_GRADED: tuple[str, ...] = (
    "anc", "wbc", "plt", "hgb", "lym", "alb",
    "alt", "ast", "alp", "bili", "creat", "na", "k", "ca", "gluc",
)

#: Physiologically impossible results are transcription errors, not observations, and
#: are set to missing. Bounds are deliberately wide - the intent is to remove keying
#: mistakes (an albumin of 0.004 g/dL) without touching genuine extreme toxicity.
LAB_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "hgb": (3.0, 25.0),          # g/dL
    "alb": (0.5, 8.0),           # g/dL
    "tprot": (1.0, 15.0),        # g/dL
    "na": (100.0, 190.0),        # mEq/L
    "k": (1.0, 10.0),            # mEq/L
    "ca": (3.0, 20.0),           # mg/dL
    "creat": (0.05, 25.0),       # mg/dL
    "gluc": (10.0, 1500.0),      # mg/dL
    "anc": (0.0, 100.0),         # 10^9/L
    "wbc": (0.05, 300.0),        # 10^9/L
    "plt": (1.0, 3000.0),        # 10^9/L
    "lym": (0.0, 100.0),         # 10^9/L
}

# --------------------------------------------------------------------------------------
# Disease burden, response and prior therapy (TMM_P, IOTA_P, CD_B_P, CN_6_P)
# --------------------------------------------------------------------------------------
#: RECIST overall response mapped to an ordinal scale, worst-is-highest, so a single
#: numeric feature carries the direction a tree can split on.
RECIST_ORDER: dict[str, float] = {
    "COMPLETE RESPONSE": 0.0,
    "PARTIAL RESPONSE": 1.0,
    "STABLE DISEASE": 2.0,
    "PROGRESSIVE DISEASE": 3.0,
}

#: Disease sites recorded often enough in ``TMM_P.TMMDIS`` to support an indicator.
#: Liver involvement in particular alters the clearance of irinotecan's active metabolite.
TUMOUR_SITES: tuple[str, ...] = ("LIVER", "LUNG", "PERITONEUM", "BONE")

#: Radiotherapy sites that irradiate marrow-bearing bone. Prior pelvic or abdominal
#: radiotherapy permanently reduces haematopoietic reserve, which is a recognised
#: consideration before starting a myelosuppressive doublet.
PELVIC_RT_PATTERN: str = "PELVI|RECT|SACRAL|ABDOMIN|COLON|PUBIC|ILIAC|SIGMOID"

#: Analytes with no laboratory reference range anywhere in the study. CEA is a tumour
#: marker rather than a safety analyte, so no normal range was collected; its level
#: feature is ``log10(value + 1)`` instead of a range-relative position.
LAB_NO_REFERENCE_RANGE: tuple[str, ...] = ("cea",)
