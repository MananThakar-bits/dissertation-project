"""
aeprep - Adverse Event PREdiction Package
==========================================

Supporting library for the dissertation

    "Temporal Prediction of In-Trial Adverse Events in Oncology Clinical Trials
     Using Machine Learning"
    Thakar Manan Pradipbhai (2024DA04324), M.Tech Data Science & Engineering, BITS Pilani

The package encapsulates the reusable logic (ingestion, label construction, feature
engineering, modelling and evaluation) so that the Jupyter notebooks under
``notebooks/`` stay readable and focus on analysis and narrative.

Study data: Project Data Sphere study **A6181122** (comparator arm of a randomised
phase III trial of FOLFIRI +/- sunitinib in metastatic colorectal cancer).
"""

from . import config, io, quality, labels, features_static, features_temporal
from . import modeling, evaluation, enrolment, recommendation, plots

__version__ = "1.2.0"

__all__ = [
    "config",
    "io",
    "quality",
    "labels",
    "features_static",
    "features_temporal",
    "modeling",
    "evaluation",
    "enrolment",
    "recommendation",
    "plots",
]
