"""Model zoo, patient-grouped resampling and the temporal sequence model.

Design rules enforced here
--------------------------
1. **Patient-grouped splitting everywhere.** In the landmark design one patient
   contributes many rows; a random row split would place correlated rows of the
   same patient on both sides and inflate performance. Every split and every CV
   fold is grouped by ``PID``.
2. **Preprocessing inside the pipeline.** Imputation and scaling are fitted on the
   training fold only, so no information leaks from the validation fold.
3. **Interpretable benchmark first.** Penalised logistic regression is always
   included as the reference model, per the dissertation's design principle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CFG

__all__ = [
    "NON_FEATURE_COLUMNS",
    "split_columns",
    "drop_uninformative_columns",
    "make_preprocessor",
    "fit_with_calibration",
    "feature_names_out",
    "build_model_zoo",
    "grouped_train_test_split",
    "cross_validate_models",
    "fit_and_predict",
    "CVResult",
    "fit_cox_time_varying",
    "SequenceDataBuilder",
    "train_gru_model",
    "HAS_TORCH",
    "HAS_XGBOOST",
    "HAS_LIGHTGBM",
]

try:                                     # optional gradient-boosting backends
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:                        # pragma: no cover
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:                        # pragma: no cover
    HAS_LIGHTGBM = False

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:                        # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
#: Columns that identify a row or encode the label; never used as predictors.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "PID", "y", "landmark_day", "window_end", "window_start", "n_severe_in_window",
    "first_severe_day", "max_grade_in_window", "window_complete", "informative",
    "first_dose_day", "on_treatment_end", "ae_obs_end",
)


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition a design matrix into numeric and categorical column names."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def drop_uninformative_columns(X: pd.DataFrame,
                               max_missing: float = 0.95,
                               verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove constant and near-empty columns, returning the reason for each drop.

    In this dataset the blinded (placebo) treatment component is uniformly zero -
    Project Data Sphere released only the comparator arm of A6181122 - so every
    placebo-derived feature is structurally constant and must be dropped before
    modelling.
    """
    reasons = []
    for c in X.columns:
        s = X[c]
        miss = s.isna().mean()
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            reasons.append({"column": c,
                            "reason": "all missing" if n_unique == 0 else "constant",
                            "value": (s.dropna().iloc[0] if n_unique == 1 else None)})
        elif miss > max_missing:
            reasons.append({"column": c, "reason": f">{max_missing:.0%} missing",
                            "value": None})
    dropped = pd.DataFrame(reasons, columns=["column", "reason", "value"])
    if verbose and len(dropped):
        print(f"  dropped {len(dropped)} uninformative columns")
    return X.drop(columns=dropped["column"].tolist()), dropped


def _replace_non_finite(X):
    """Turn +/-inf into NaN so the imputer (rather than the scaler) handles them."""
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)
    return np.where(np.isfinite(X), X, np.nan)


def make_preprocessor(X: pd.DataFrame, scale: bool = True) -> Pipeline:
    """Median-impute + optionally scale numerics; most-frequent-impute + one-hot others.

    Division-derived features (dose ratios, relative dose intensity) can produce
    infinities when the denominator is zero, so a cleaning step runs first.
    """
    from sklearn.preprocessing import FunctionTransformer

    numeric, categorical = split_columns(X)

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                 sparse_output=False)),
    ])

    ct = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric),
         ("cat", cat_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("clean", FunctionTransformer(_replace_non_finite, feature_names_out="one-to-one")),
                     ("encode", ct)])


def build_model_zoo(X: pd.DataFrame, cfg=CFG,
                    include: Sequence[str] | None = None) -> dict[str, Pipeline]:
    """Return the candidate models as end-to-end scikit-learn pipelines.

    ``Logistic Regression`` is the interpretable benchmark; the tree ensembles are
    the flexible comparators. Class imbalance is handled with class weighting rather
    than resampling so that predicted probabilities stay interpretable after
    recalibration.
    """
    rs = cfg.random_state
    zoo: dict[str, Pipeline] = {}

    zoo["Logistic Regression"] = Pipeline([
        ("prep", make_preprocessor(X, scale=True)),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, penalty="l2",
                                   class_weight="balanced", random_state=rs)),
    ])

    zoo["Elastic-Net Logistic"] = Pipeline([
        ("prep", make_preprocessor(X, scale=True)),
        ("clf", LogisticRegression(max_iter=8000, penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, C=0.5, class_weight="balanced",
                                   random_state=rs)),
    ])

    zoo["Random Forest"] = Pipeline([
        ("prep", make_preprocessor(X, scale=False)),
        ("clf", RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                       max_features="sqrt", class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=rs)),
    ])

    zoo["Hist Gradient Boosting"] = Pipeline([
        ("prep", make_preprocessor(X, scale=False)),
        ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=rs)),
    ])

    if HAS_XGBOOST:
        zoo["XGBoost"] = Pipeline([
            ("prep", make_preprocessor(X, scale=False)),
            ("clf", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=2.0, min_child_weight=5,
                                  eval_metric="logloss", tree_method="hist",
                                  n_jobs=-1, random_state=rs)),
        ])

    if HAS_LIGHTGBM:
        zoo["LightGBM"] = Pipeline([
            ("prep", make_preprocessor(X, scale=False)),
            ("clf", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=20, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8,
                                   reg_lambda=2.0, class_weight="balanced",
                                   n_jobs=-1, random_state=rs, verbose=-1)),
        ])

    if include is not None:
        zoo = {k: v for k, v in zoo.items() if k in include}
    return zoo


# --------------------------------------------------------------------------------------
# Grouped resampling
# --------------------------------------------------------------------------------------
def grouped_train_test_split(df: pd.DataFrame, y: pd.Series, groups: pd.Series,
                             cfg=CFG) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``test_size`` of the *patients*, keeping the event rate balanced.

    Returns positional index arrays for the training and test rows.
    """
    n_splits = max(2, int(round(1.0 / cfg.test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=cfg.random_state)
    train_idx, test_idx = next(sgkf.split(df, y, groups))
    return train_idx, test_idx


@dataclass
class CVResult:
    """Out-of-fold predictions and fold assignment for one model."""

    name: str
    oof_pred: np.ndarray
    fold_id: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    fitted_folds: list


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                          groups: pd.Series, cfg=CFG,
                          verbose: bool = True) -> dict[str, CVResult]:
    """Stratified, patient-grouped cross-validation producing out-of-fold probabilities."""
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True,
                                random_state=cfg.random_state)
    folds = list(sgkf.split(X, y, groups))

    results: dict[str, CVResult] = {}
    for name, model in models.items():
        oof = np.full(len(X), np.nan)
        fold_id = np.full(len(X), -1)
        fitted = []
        for k, (tr, va) in enumerate(folds):
            est = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X.iloc[tr], y.iloc[tr])
                oof[va] = est.predict_proba(X.iloc[va])[:, 1]
            fold_id[va] = k
            fitted.append(est)
        results[name] = CVResult(name, oof, fold_id, y.to_numpy(),
                                 groups.to_numpy(), fitted)
        if verbose:
            from .evaluation import binary_metrics
            m = binary_metrics(y.to_numpy(), oof)
            print(f"  {name:26s} AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}  "
                  f"Brier={m['brier']:.4f}")
    return results


def fit_and_predict(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    """Fit on the training set and return the fitted estimator plus test probabilities."""
    est = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_test)[:, 1]
    return est, p


def fit_with_calibration(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                         groups_train: pd.Series, X_test: pd.DataFrame,
                         method: str = "sigmoid", cfg=CFG):
    """Refit with patient-grouped cross-validated probability calibration.

    Class weighting improves ranking but shifts the probability level, which makes
    the raw Brier score and calibration plot misleading.

    Calibration is done with :class:`CalibratedClassifierCV` over patient-grouped
    folds rather than a single holdout. A single holdout was tried first and proved
    unusable at this sample size: with ~60 calibration patients the fitted Platt
    slope can come out negative, which *reverses* the ranking and destroys AUROC.
    Cross-validated calibration averages several calibrators fitted on disjoint
    folds and is stable. The test patients are never involved.
    """
    from sklearn.calibration import CalibratedClassifierCV

    n_splits = min(cfg.n_splits, int(groups_train.nunique() // 20) or 2)
    sgkf = StratifiedGroupKFold(n_splits=max(n_splits, 2), shuffle=True,
                                random_state=cfg.random_state)
    splits = list(sgkf.split(X_train, y_train, groups_train))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = CalibratedClassifierCV(clone(model), method=method, cv=splits,
                                     ensemble=True)
        cal.fit(X_train, y_train)
        p = cal.predict_proba(X_test)[:, 1]
    return cal, p


def feature_names_out(fitted_pipeline: Pipeline) -> list[str]:
    """Column names produced by the fitted preprocessing step of a model pipeline."""
    prep = fitted_pipeline.named_steps["prep"]
    return list(prep.get_feature_names_out())


# --------------------------------------------------------------------------------------
# Time-to-event comparator
# --------------------------------------------------------------------------------------
def fit_cox_time_varying(landmark_df: pd.DataFrame, feature_cols: Sequence[str],
                         cfg=CFG):
    """Cox model with time-varying covariates on the landmark (counting-process) data.

    Each landmark row becomes an interval ``(start, stop]`` for its patient, with the
    event indicator set when the first severe adverse event falls inside the interval.
    This is the classical statistical counterpart of the machine-learning landmark
    model and is reported alongside it as a methodological benchmark.
    """
    from lifelines import CoxTimeVaryingFitter

    df = landmark_df.sort_values(["PID", "landmark_day"]).copy()
    df["start"] = df["landmark_day"]
    df["stop"] = df.groupby("PID")["landmark_day"].shift(-1)
    df["stop"] = df["stop"].fillna(df["landmark_day"] + cfg.landmark_step_days)
    df = df[df["stop"] > df["start"]]

    # Event = a severe AE occurring inside this interval (not the 30-day window).
    df["event"] = ((df["first_severe_day"] > df["start"]) &
                   (df["first_severe_day"] <= df["stop"])).fillna(False).astype(int)
    # Once a patient has their first severe event, later intervals are dropped.
    df["cum_event"] = df.groupby("PID")["event"].cumsum()
    df = df[(df["cum_event"] == 0) | (df["event"] == 1)]

    cols = [c for c in feature_cols if c in df.columns]
    model_df = df[["PID", "start", "stop", "event", *cols]].copy()
    model_df[cols] = (model_df[cols].apply(pd.to_numeric, errors="coerce")
                                    .replace([np.inf, -np.inf], np.nan))
    model_df[cols] = model_df[cols].fillna(model_df[cols].median()).fillna(0.0)

    # Drop degenerate columns; the partial-likelihood optimiser fails silently on
    # constant covariates and on perfectly collinear pairs (e.g. days vs weeks).
    keep = [c for c in cols if model_df[c].std(ddof=0) > 1e-8]
    corr = model_df[keep].corr().abs()
    redundant: set[str] = set()
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if b not in redundant and corr.loc[a, b] > 0.98:
                redundant.add(b)
    keep = [c for c in keep if c not in redundant]

    model_df = model_df[["PID", "start", "stop", "event", *keep]]
    model_df[keep] = ((model_df[keep] - model_df[keep].mean()) /
                      model_df[keep].std(ddof=0))

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(model_df, id_col="PID", event_col="event",
            start_col="start", stop_col="stop", show_progress=False)
    ctv.dropped_covariates_ = sorted(set(cols) - set(keep))
    return ctv, model_df


# --------------------------------------------------------------------------------------
# Sequence (GRU) model
# --------------------------------------------------------------------------------------
class SequenceDataBuilder:
    """Reshape the landmark table into padded per-patient sequences for an RNN.

    The landmark matrix is already a regular per-patient time series (one step per
    landmark day), so a recurrent model can consume it directly and learn the
    temporal dependence between consecutive toxicity states instead of relying only
    on hand-engineered look-back windows.
    """

    def __init__(self, feature_cols: Sequence[str]):
        self.feature_cols = list(feature_cols)
        self.medians_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "SequenceDataBuilder":
        X = (df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
                                  .replace([np.inf, -np.inf], np.nan))
        # A column that is entirely missing in the training split would otherwise
        # propagate NaN through the network; impute it with a constant zero.
        self.medians_ = X.median().fillna(0.0)
        Xi = X.fillna(self.medians_)
        self.mean_ = Xi.mean().fillna(0.0)
        self.std_ = Xi.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame):
        """Return ``(X, y, mask, index)`` with ``X`` of shape (n_patients, T, n_features)."""
        d = df.sort_values(["PID", "landmark_day"])
        X = (d[self.feature_cols].apply(pd.to_numeric, errors="coerce")
                                 .replace([np.inf, -np.inf], np.nan))
        X = ((X.fillna(self.medians_) - self.mean_) / self.std_)
        X = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)

        pids = d["PID"].to_numpy()
        y = d["y"].to_numpy(dtype=np.float32)
        uniq, inverse = np.unique(pids, return_inverse=True)
        lengths = np.bincount(inverse)
        T = int(lengths.max())

        Xs = np.zeros((len(uniq), T, X.shape[1]), dtype=np.float32)
        ys = np.zeros((len(uniq), T), dtype=np.float32)
        mask = np.zeros((len(uniq), T), dtype=np.float32)
        row_index = np.full((len(uniq), T), -1, dtype=np.int64)

        pos = np.zeros(len(uniq), dtype=int)
        for r in range(len(d)):
            p = inverse[r]
            Xs[p, pos[p]] = X[r]
            ys[p, pos[p]] = y[r]
            mask[p, pos[p]] = 1.0
            row_index[p, pos[p]] = d.index[r]
            pos[p] += 1
        return Xs, ys, mask, row_index, uniq

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)


if HAS_TORCH:

    class GRURiskNet(nn.Module):
        """Single-layer GRU emitting a severe-AE probability at every landmark step."""

        def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(self.drop(h)).squeeze(-1)

else:  # pragma: no cover
    GRURiskNet = None  # type: ignore


def train_gru_model(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                    feature_cols: Sequence[str], cfg=CFG,
                    hidden: int = 48, epochs: int = 120, lr: float = 3e-3,
                    verbose: bool = True):
    """Train the GRU sequence model and return ``(model, builder, valid_predictions)``.

    ``valid_predictions`` is a Series aligned to ``valid_df.index`` so it can be
    scored with exactly the same evaluation code as the tabular models.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed; run `pip install torch`.")

    torch.manual_seed(cfg.random_state)
    builder = SequenceDataBuilder(feature_cols).fit(train_df)

    Xtr, ytr, mtr, _, _ = builder.transform(train_df)
    Xva, yva, mva, idx_va, _ = builder.transform(valid_df)

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr); mtr_t = torch.from_numpy(mtr)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva); mva_t = torch.from_numpy(mva)

    # Positive class weight = n_negative / n_positive over the (masked) training steps.
    n_pos = float((ytr * mtr).sum())
    n_neg = float(mtr.sum()) - n_pos
    pos_weight = torch.tensor([float(np.clip(n_neg / max(n_pos, 1.0), 1.0, 20.0))])

    model = GRURiskNet(Xtr.shape[2], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    best_state, best_loss, patience, bad = None, np.inf, 20, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = (loss_fn(logits, ytr_t) * mtr_t).sum() / mtr_t.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vlogits = model(Xva_t)
            vloss = ((loss_fn(vlogits, yva_t) * mva_t).sum() / mva_t.sum()).item()
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train={loss.item():.4f}  valid={vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xva_t)).numpy()

    flat_idx = idx_va.reshape(-1)
    flat_p = probs.reshape(-1)
    keep = flat_idx >= 0
    preds = pd.Series(flat_p[keep], index=flat_idx[keep]).reindex(valid_df.index)
    return model, builder, preds
