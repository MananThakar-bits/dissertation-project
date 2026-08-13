"""Data-quality audit utilities.

Phase 4 of the dissertation plan ("Static Dataset Construction and Data Quality
Audit") requires an explicit, reproducible assessment of completeness and
consistency. These helpers produce the tables that the notebooks render.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "profile_frame",
    "profile_all",
    "missingness_report",
    "domain_overview",
    "consistency_checks",
    "constant_or_empty_columns",
]


def profile_frame(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Column-level profile: dtype, completeness, cardinality and example values."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        uniques = s.dropna().unique()
        rows.append({
            "domain": name,
            "column": col,
            "dtype": str(s.dtype),
            "n_rows": n,
            "n_non_null": non_null,
            "pct_missing": round(100.0 * (n - non_null) / n, 2) if n else np.nan,
            "n_unique": int(s.nunique(dropna=True)),
            "example_values": ", ".join(map(lambda x: str(x)[:28], uniques[:4])),
        })
    return pd.DataFrame(rows)


def profile_all(domains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack :func:`profile_frame` over every loaded domain."""
    return pd.concat([profile_frame(df, name) for name, df in domains.items()],
                     ignore_index=True)


def domain_overview(domains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per domain: shape, patient coverage and usable-column count."""
    rows = []
    for name, df in domains.items():
        n_pat = df["PID"].nunique() if "PID" in df.columns else np.nan
        empty_cols = int((df.notna().sum() == 0).sum())
        rows.append({
            "domain": name,
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_patients": n_pat,
            "n_empty_columns": empty_cols,
            "n_usable_columns": df.shape[1] - empty_cols,
            "rows_per_patient": round(len(df) / n_pat, 2) if n_pat else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)


def constant_or_empty_columns(domains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Columns that carry no information (fully missing, or a single value).

    Project Data Sphere de-identification blanks several fields (e.g. ``COUNTRY``,
    ``CPEVENT`` in some domains); listing them explicitly documents why they are
    excluded from feature engineering.
    """
    rows = []
    for name, df in domains.items():
        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                rows.append({
                    "domain": name,
                    "column": col,
                    "reason": "all missing" if n_unique == 0 else "single value",
                    "value": (df[col].dropna().iloc[0] if n_unique == 1 else None),
                })
    return pd.DataFrame(rows)


def missingness_report(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Per-column missingness of a modelling matrix, sorted worst-first."""
    n = len(df)
    miss = df.isna().sum().rename("n_missing").to_frame()
    miss["pct_missing"] = (100.0 * miss["n_missing"] / n).round(2)
    miss = miss[miss["pct_missing"] > threshold]
    return miss.sort_values("pct_missing", ascending=False)


def consistency_checks(domains: dict[str, pd.DataFrame],
                       timeline: pd.DataFrame) -> pd.DataFrame:
    """Cross-domain logical checks with an explicit pass/inspect verdict.

    Each check returns the number of offending records; a non-zero count is not
    necessarily an error (clinical trial data legitimately contains ongoing events
    and pre-treatment history) but must be visible and explained in the report.
    """
    from .io import tidy_adverse, tidy_testdrug

    ae = tidy_adverse(domains["adverse"])
    td = tidy_testdrug(domains["testdrug"])
    checks: list[dict] = []

    def add(check: str, n: int, note: str) -> None:
        checks.append({"check": check, "n_records": int(n),
                       "status": "OK" if n == 0 else "REVIEW", "interpretation": note})

    add("AE with missing CTCAE grade", ae["grade"].isna().sum(),
        "Ungraded events cannot contribute to the severity target.")
    add("AE with missing start day", ae["start_day"].isna().sum(),
        "Start day is mandatory for windowed labelling.")
    add("AE stop day earlier than start day",
        (ae["end_day"] < ae["start_day"]).sum(),
        "Would indicate a data-entry inversion.")
    add("AE ongoing at data cut (no stop day)", ae["end_day"].isna().sum(),
        "Expected: unresolved events are recorded without a stop day.")
    add("AE starting before first dose",
        (ae.merge(timeline[["PID", "first_dose_day"]], on="PID", how="left")
           .eval("start_day < first_dose_day").sum()),
        "Pre-treatment events; excluded from the target, used as baseline history.")
    add("Dosing record with missing start day", td["start_day"].isna().sum(),
        "Planned-but-not-administered cycle rows.")
    add("Dosing record with zero total dose", (td["dose"] == 0).sum(),
        "Expected for the blinded placebo component and for held doses.")
    add("Patients with end-of-treatment after end-of-study",
        (timeline["eot_day"] > timeline["eos_day"]).sum(),
        "Would break the observation-window logic.")
    add("Patients with no dosing record",
        timeline["n_dose_records"].isna().sum(),
        "Randomised but never treated.")
    add("Patients with adverse events after the derived observation end",
        (ae.merge(timeline[["PID", "ae_obs_end"]], on="PID", how="left")
           .eval("start_day > ae_obs_end").sum()),
        "Should be zero by construction (window is extended to cover late events).")

    return pd.DataFrame(checks)
