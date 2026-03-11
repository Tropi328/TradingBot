"""
train_v3 — Train a LightGBM model from shadow observer JSONL data.

Usage (CLI):
    python -m bot.ml.train_v3 --data reports/shadow_observe.jsonl --out models/score_v3.pkl

Pipeline:
    1. Load JSONL shadow records (one JSON object per line).
    2. Extract the 57 V3 features from each record's ``score_v3_features`` dict.
       Fall back to reconstructing features from ShadowCandidate fields when
       the embedded feature dict is absent (older data).
    3. Derive a binary label: **win** = shadow outcome profitable (R > 0).
    4. Train a LightGBM binary classifier with 5-fold stratified CV.
    5. Optionally calibrate probabilities with isotonic regression.
    6. Save in the format expected by ``TrainedScoreV3Model.load()``.

The module exposes pure functions so each step is independently testable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bot.strategy.score_v3 import FEATURE_NAMES, TrainedScoreV3Model

_LOG = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_shadow_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dicts.

    Blank lines and lines that fail JSON parsing are silently skipped.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                _LOG.debug("Skipped malformed JSON at line %d in %s", lineno, path)
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Feature + label extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_features_from_record(
    record: dict[str, Any],
    feature_names: list[str] | None = None,
) -> dict[str, float] | None:
    """Pull the V3 feature vector from a single shadow record.

    Looks for the ``score_v3_features`` sub-dict first (written by
    ``apply_score_v3``).  Returns *None* if the record lacks usable
    feature data so the caller can skip it.
    """
    names = feature_names or FEATURE_NAMES
    # Primary source: embedded feature dict from the scoring pipeline.
    feat_dict = record.get("score_v3_features") or record.get("raw_score_breakdown") or {}
    if not feat_dict:
        return None
    # Verify at least half the features are present — otherwise too sparse.
    present = sum(1 for n in names if n in feat_dict)
    if present < len(names) * 0.5:
        return None
    return {n: float(feat_dict.get(n, 0.0)) for n in names}


def derive_label(
    record: dict[str, Any],
    *,
    r_threshold: float = 0.0,
) -> int | None:
    """Derive a binary win/loss label from shadow outcome fields.

    Returns 1 (win) if shadow_r > *r_threshold*, 0 (loss) if filled and
    lost, or *None* if the candidate was never filled (unusable for
    supervised learning).
    """
    if not record.get("shadow_filled", False):
        return None
    shadow_r = record.get("shadow_r")
    if shadow_r is None:
        return None
    try:
        return 1 if float(shadow_r) > r_threshold else 0
    except (TypeError, ValueError):
        return None


def build_dataset(
    records: list[dict[str, Any]],
    *,
    feature_names: list[str] | None = None,
    r_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert shadow records into (X, y, feature_names) arrays.

    Skips records that lack features or were not filled.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with values 0 or 1
    names : list of feature name strings (column order)

    Raises
    ------
    ValueError
        If fewer than 20 usable samples remain after filtering.
    """
    names = feature_names or list(FEATURE_NAMES)
    rows: list[list[float]] = []
    labels: list[int] = []

    for rec in records:
        feat = extract_features_from_record(rec, names)
        if feat is None:
            continue
        label = derive_label(rec, r_threshold=r_threshold)
        if label is None:
            continue
        rows.append([feat[n] for n in names])
        labels.append(label)

    if len(rows) < 20:
        raise ValueError(
            f"Only {len(rows)} usable samples found (need >= 20). Collect more shadow data before training."
        )

    _LOG.info(
        "Built dataset: %d samples, %.1f%% wins",
        len(rows),
        100.0 * sum(labels) / len(labels),
    )
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32), names


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """Hyper-parameters for the LightGBM training pipeline."""

    n_folds: int = 5
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float | None = None  # auto-computed if None
    calibrate: bool = True
    random_state: int = 42


@dataclass
class TrainResult:
    """Outputs from a completed training run."""

    model: Any  # LGBMClassifier
    calibrator: Any | None  # IsotonicRegression or None
    feature_names: list[str]
    cv_auc_mean: float
    cv_auc_std: float
    cv_logloss_mean: float
    cv_logloss_std: float
    n_samples: int
    win_rate: float


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a LightGBM classifier with stratified K-fold CV.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) binary labels
    feature_names : column names matching X columns
    config : training hyper-parameters (defaults used if None)

    Returns
    -------
    TrainResult with the fitted model, optional calibrator, and CV metrics.

    Raises
    ------
    ImportError
        If ``lightgbm`` is not installed.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM is required for ML training. Install it with: pip install lightgbm") from exc
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cfg = config or TrainConfig()

    # Auto-compute class weight if not set
    pos_weight = cfg.scale_pos_weight
    if pos_weight is None:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

    clf = lgb.LGBMClassifier(
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=pos_weight,
        random_state=cfg.random_state,
        verbosity=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    scoring = {
        "auc": "roc_auc",
        "logloss": "neg_log_loss",
    }
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

    cv_auc_mean = float(np.mean(cv_results["test_auc"]))
    cv_auc_std = float(np.std(cv_results["test_auc"]))
    cv_ll_mean = float(-np.mean(cv_results["test_logloss"]))  # flip sign
    cv_ll_std = float(np.std(cv_results["test_logloss"]))

    _LOG.info(
        "CV results: AUC=%.4f +/- %.4f, LogLoss=%.4f +/- %.4f",
        cv_auc_mean,
        cv_auc_std,
        cv_ll_mean,
        cv_ll_std,
    )

    # Fit on full data
    clf.fit(X, y)

    # Optional calibration
    calibrator = None
    if cfg.calibrate and len(y) >= 100:
        _LOG.info("Calibrating probabilities with isotonic regression (3-fold)...")
        cal_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.random_state)
        cal_clf = CalibratedClassifierCV(clf, method="isotonic", cv=cal_cv)
        cal_clf.fit(X, y)
        calibrator = cal_clf

    win_rate = float(y.sum()) / len(y)

    return TrainResult(
        model=clf,
        calibrator=calibrator,
        feature_names=list(feature_names),
        cv_auc_mean=round(cv_auc_mean, 4),
        cv_auc_std=round(cv_auc_std, 4),
        cv_logloss_mean=round(cv_ll_mean, 4),
        cv_logloss_std=round(cv_ll_std, 4),
        n_samples=len(y),
        win_rate=round(win_rate, 4),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════════


def save_model(result: TrainResult, path: Path) -> Path:
    """Save trained model in TrainedScoreV3Model-compatible format.

    Writes the pickle + HMAC .sig sidecar via ``TrainedScoreV3Model.save()``.
    Returns the path written to.
    """
    wrapper = TrainedScoreV3Model(
        model=result.model,
        feature_names=result.feature_names,
        calibrator=result.calibrator,
    )
    wrapper.save(path)
    _LOG.info("Model saved to %s (+ .sig sidecar)", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m bot.ml.train_v3 --data <path> --out <path>``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train ScoreV3 LightGBM model from shadow observer data.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to shadow_observe.jsonl file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/score_v3.pkl",
        help="Output path for the trained model (default: models/score_v3.pkl).",
    )
    parser.add_argument(
        "--r-threshold",
        type=float,
        default=0.0,
        help="Minimum R for a shadow outcome to count as a win (default: 0.0).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip probability calibration.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    data_path = Path(args.data)
    if not data_path.exists():
        _LOG.error("Data file not found: %s", data_path)
        raise SystemExit(1)

    _LOG.info("Loading shadow data from %s ...", data_path)
    records = load_shadow_jsonl(data_path)
    _LOG.info("Loaded %d records.", len(records))

    X, y, names = build_dataset(records, r_threshold=args.r_threshold)

    cfg = TrainConfig(
        n_folds=args.folds,
        calibrate=not args.no_calibrate,
    )
    result = train_model(X, y, names, config=cfg)

    _LOG.info(
        "Training complete: %d samples, %.1f%% win rate, AUC=%.4f",
        result.n_samples,
        result.win_rate * 100,
        result.cv_auc_mean,
    )

    out_path = Path(args.out)
    save_model(result, out_path)
    _LOG.info("Done. Model written to %s", out_path)


if __name__ == "__main__":
    main()
