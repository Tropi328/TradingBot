"""Tests for the ML training pipeline (bot.ml.train_v3).

Sections:
    F1 — JSONL loading
    F2 — Feature extraction from records
    F3 — Label derivation
    F4 — Dataset building
    F5 — Model training (requires lightgbm — auto-skipped if missing)
    F6 — Save / load round-trip
    F7 — CLI entry point
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bot.strategy.score_v3 import FEATURE_NAMES

from bot.ml.train_v3 import (
    TrainConfig,
    build_dataset,
    derive_label,
    extract_features_from_record,
    load_shadow_jsonl,
    save_model,
    train_model,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

_NUM_FEATURES = len(FEATURE_NAMES)


def _make_feature_dict(value: float = 0.5) -> dict[str, float]:
    """Create a complete V3 feature dict with all features set to *value*."""
    return {name: value for name in FEATURE_NAMES}


def _make_record(
    *,
    features: dict[str, float] | None = None,
    shadow_filled: bool = True,
    shadow_r: float | None = 1.5,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a minimal shadow record dict."""
    rec: dict[str, Any] = {
        "score_v3_features": features if features is not None else _make_feature_dict(),
        "shadow_filled": shadow_filled,
    }
    if shadow_r is not None:
        rec["shadow_r"] = shadow_r
    if extra:
        rec.update(extra)
    return rec


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL to *path*."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  F1 — JSONL loading
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadShadowJsonl:
    """Section F1: load_shadow_jsonl()."""

    def test_load_valid_records(self, tmp_path: Path) -> None:
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, records)
        loaded = load_shadow_jsonl(p)
        assert len(loaded) == 3
        assert loaded[0] == {"a": 1}
        assert loaded[2] == {"c": 3}

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text('{"x": 1}\n\n\n{"y": 2}\n', encoding="utf-8")
        loaded = load_shadow_jsonl(p)
        assert len(loaded) == 2

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text('{"ok": 1}\nNOT JSON\n{"ok": 2}\n', encoding="utf-8")
        loaded = load_shadow_jsonl(p)
        assert len(loaded) == 2
        assert loaded[0] == {"ok": 1}
        assert loaded[1] == {"ok": 2}

    def test_non_dict_lines_skipped(self, tmp_path: Path) -> None:
        """JSON arrays or primitives should be skipped — only dicts count."""
        p = tmp_path / "data.jsonl"
        p.write_text('[1, 2]\n"hello"\n42\n{"valid": true}\n', encoding="utf-8")
        loaded = load_shadow_jsonl(p)
        assert len(loaded) == 1
        assert loaded[0] == {"valid": True}

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_shadow_jsonl(p) == []


# ═══════════════════════════════════════════════════════════════════════════
#  F2 — Feature extraction from records
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractFeaturesFromRecord:
    """Section F2: extract_features_from_record()."""

    def test_full_features_extracted(self) -> None:
        rec = _make_record()
        feat = extract_features_from_record(rec)
        assert feat is not None
        assert len(feat) == _NUM_FEATURES
        assert all(v == 0.5 for v in feat.values())

    def test_missing_features_filled_with_zero(self) -> None:
        """If > 50% present but some missing, missing ones become 0.0."""
        partial = {name: 1.0 for name in FEATURE_NAMES[:50]}  # 50 of 57
        rec = _make_record(features=partial)
        feat = extract_features_from_record(rec)
        assert feat is not None
        assert feat[FEATURE_NAMES[0]] == 1.0
        # Missing features should default to 0.0
        assert feat[FEATURE_NAMES[56]] == 0.0

    def test_too_sparse_returns_none(self) -> None:
        """If fewer than 50% of features present, returns None."""
        sparse = {name: 1.0 for name in FEATURE_NAMES[:10]}  # 10 of 57 < 50%
        rec = _make_record(features=sparse)
        assert extract_features_from_record(rec) is None

    def test_no_features_dict_returns_none(self) -> None:
        rec: dict[str, Any] = {"shadow_filled": True, "shadow_r": 1.0}
        assert extract_features_from_record(rec) is None

    def test_empty_features_dict_returns_none(self) -> None:
        rec = _make_record(features={})
        assert extract_features_from_record(rec) is None

    def test_fallback_to_raw_score_breakdown(self) -> None:
        """If score_v3_features missing, falls back to raw_score_breakdown."""
        feat_dict = _make_feature_dict(0.7)
        rec: dict[str, Any] = {
            "raw_score_breakdown": feat_dict,
            "shadow_filled": True,
            "shadow_r": 1.0,
        }
        feat = extract_features_from_record(rec)
        assert feat is not None
        assert feat[FEATURE_NAMES[0]] == pytest.approx(0.7)

    def test_custom_feature_names(self) -> None:
        """Works with a custom feature list (subset)."""
        custom_names = ["htf_bias_aligned", "fvg_present", "mss_confirmed"]
        rec: dict[str, Any] = {
            "score_v3_features": {
                "htf_bias_aligned": 1.0,
                "fvg_present": 1.0,
                "mss_confirmed": 0.0,
            },
        }
        feat = extract_features_from_record(rec, feature_names=custom_names)
        assert feat is not None
        assert len(feat) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  F3 — Label derivation
# ═══════════════════════════════════════════════════════════════════════════


class TestDeriveLabel:
    """Section F3: derive_label()."""

    def test_win(self) -> None:
        rec = _make_record(shadow_filled=True, shadow_r=1.5)
        assert derive_label(rec) == 1

    def test_loss(self) -> None:
        rec = _make_record(shadow_filled=True, shadow_r=-0.5)
        assert derive_label(rec) == 0

    def test_breakeven_is_loss_at_default_threshold(self) -> None:
        rec = _make_record(shadow_filled=True, shadow_r=0.0)
        assert derive_label(rec) == 0

    def test_custom_threshold(self) -> None:
        rec = _make_record(shadow_filled=True, shadow_r=0.5)
        # With threshold 1.0, R=0.5 is a loss
        assert derive_label(rec, r_threshold=1.0) == 0
        # With threshold 0.3, R=0.5 is a win
        assert derive_label(rec, r_threshold=0.3) == 1

    def test_unfilled_returns_none(self) -> None:
        rec = _make_record(shadow_filled=False, shadow_r=2.0)
        assert derive_label(rec) is None

    def test_missing_shadow_r_returns_none(self) -> None:
        rec: dict[str, Any] = {"shadow_filled": True}  # no shadow_r key at all
        rec["score_v3_features"] = _make_feature_dict()
        assert derive_label(rec) is None

    def test_shadow_r_none_value_returns_none(self) -> None:
        rec: dict[str, Any] = {"shadow_filled": True, "shadow_r": None}
        assert derive_label(rec) is None

    def test_non_numeric_shadow_r_returns_none(self) -> None:
        rec: dict[str, Any] = {"shadow_filled": True, "shadow_r": "bad"}
        assert derive_label(rec) is None

    def test_not_filled_by_default_returns_none(self) -> None:
        """If shadow_filled key is missing, treat as not filled."""
        rec: dict[str, Any] = {"shadow_r": 1.0}
        assert derive_label(rec) is None


# ═══════════════════════════════════════════════════════════════════════════
#  F4 — Dataset building
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildDataset:
    """Section F4: build_dataset()."""

    def _make_n_records(self, n: int, *, win_rate: float = 0.5) -> list[dict[str, Any]]:
        """Generate *n* shadow records with specified win rate."""
        n_win = int(n * win_rate)
        recs = []
        for i in range(n):
            r_val = 1.5 if i < n_win else -0.5
            recs.append(_make_record(shadow_r=r_val))
        return recs

    def test_happy_path(self) -> None:
        recs = self._make_n_records(30)
        X, y, names = build_dataset(recs)
        assert X.shape == (30, _NUM_FEATURES)
        assert y.shape == (30,)
        assert len(names) == _NUM_FEATURES
        assert set(np.unique(y)) == {0, 1}

    def test_too_few_samples_raises(self) -> None:
        recs = self._make_n_records(10)
        with pytest.raises(ValueError, match="Only 10 usable samples"):
            build_dataset(recs)

    def test_unfilled_records_skipped(self) -> None:
        """Unfilled records don't count towards sample count."""
        recs = self._make_n_records(25)
        # Add 50 unfilled records
        for _ in range(50):
            recs.append(_make_record(shadow_filled=False))
        X, y, names = build_dataset(recs)
        assert X.shape[0] == 25

    def test_featureless_records_skipped(self) -> None:
        """Records without features don't count."""
        recs = self._make_n_records(25)
        for _ in range(20):
            recs.append({"shadow_filled": True, "shadow_r": 1.0})
        X, y, _ = build_dataset(recs)
        assert X.shape[0] == 25

    def test_custom_r_threshold(self) -> None:
        """Custom R threshold changes label assignment."""
        recs = []
        for _ in range(25):
            recs.append(_make_record(shadow_r=0.5))  # R=0.5
        X, y, _ = build_dataset(recs, r_threshold=1.0)
        assert np.all(y == 0)  # All are losses at threshold 1.0

    def test_dtype_correctness(self) -> None:
        recs = self._make_n_records(30)
        X, y, _ = build_dataset(recs)
        assert X.dtype == np.float64
        assert y.dtype == np.int32


# ═══════════════════════════════════════════════════════════════════════════
#  F5 — Model training (requires lightgbm)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainModel:
    """Section F5: train_model() — skipped if lightgbm not installed."""

    @pytest.fixture(autouse=True)
    def _require_lightgbm(self) -> None:
        pytest.importorskip("lightgbm")

    def _make_data(self, n: int = 100, win_rate: float = 0.5) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Create synthetic training data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, _NUM_FEATURES))
        n_pos = int(n * win_rate)
        y = np.array([1] * n_pos + [0] * (n - n_pos), dtype=np.int32)
        return X, y, list(FEATURE_NAMES)

    def test_train_returns_result(self) -> None:
        X, y, names = self._make_data(100)
        cfg = TrainConfig(n_folds=3, n_estimators=10)  # fast
        result = train_model(X, y, names, config=cfg)
        assert 0.0 <= result.cv_auc_mean <= 1.0
        assert result.n_samples == 100
        assert 0.0 <= result.win_rate <= 1.0
        assert result.feature_names == names

    def test_no_calibration_when_disabled(self) -> None:
        X, y, names = self._make_data(100)
        cfg = TrainConfig(n_folds=3, n_estimators=10, calibrate=False)
        result = train_model(X, y, names, config=cfg)
        assert result.calibrator is None

    def test_no_calibration_when_small_dataset(self) -> None:
        """Calibration requires >= 100 samples."""
        X, y, names = self._make_data(50)
        cfg = TrainConfig(n_folds=3, n_estimators=10, calibrate=True)
        result = train_model(X, y, names, config=cfg)
        assert result.calibrator is None

    def test_calibration_when_large_enough(self) -> None:
        X, y, names = self._make_data(150)
        cfg = TrainConfig(n_folds=3, n_estimators=10, calibrate=True)
        result = train_model(X, y, names, config=cfg)
        assert result.calibrator is not None

    def test_auto_scale_pos_weight(self) -> None:
        """When scale_pos_weight is None, it auto-computes from class balance."""
        X, y, names = self._make_data(100, win_rate=0.3)
        cfg = TrainConfig(n_folds=3, n_estimators=10)
        result = train_model(X, y, names, config=cfg)
        # Should succeed without error
        assert result.n_samples == 100

    def test_custom_scale_pos_weight(self) -> None:
        X, y, names = self._make_data(100)
        cfg = TrainConfig(n_folds=3, n_estimators=10, scale_pos_weight=2.0)
        result = train_model(X, y, names, config=cfg)
        assert result.n_samples == 100


# ═══════════════════════════════════════════════════════════════════════════
#  F6 — Save / load round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveLoadRoundTrip:
    """Section F6: save_model() + TrainedScoreV3Model.load()."""

    @pytest.fixture(autouse=True)
    def _require_lightgbm(self) -> None:
        pytest.importorskip("lightgbm")

    def _train_quick(self) -> "TrainResult":
        from bot.ml.train_v3 import TrainResult

        rng = np.random.default_rng(99)
        X = rng.standard_normal((60, _NUM_FEATURES))
        y = np.array([1] * 30 + [0] * 30, dtype=np.int32)
        names = list(FEATURE_NAMES)
        cfg = TrainConfig(n_folds=3, n_estimators=10, calibrate=False)
        return train_model(X, y, names, config=cfg)

    def test_save_creates_files(self, tmp_path: Path) -> None:
        result = self._train_quick()
        out = tmp_path / "model.pkl"
        save_model(result, out)
        assert out.exists()
        assert out.with_suffix(".pkl.sig").exists()

    def test_load_matches_save(self, tmp_path: Path) -> None:
        from bot.strategy.score_v3 import TrainedScoreV3Model

        result = self._train_quick()
        out = tmp_path / "model.pkl"
        save_model(result, out)

        loaded = TrainedScoreV3Model.load(out)
        assert loaded.feature_names == result.feature_names
        assert loaded.model is not None

    def test_predict_after_round_trip(self, tmp_path: Path) -> None:
        from bot.strategy.score_v3 import TrainedScoreV3Model

        result = self._train_quick()
        out = tmp_path / "model.pkl"
        save_model(result, out)

        loaded = TrainedScoreV3Model.load(out)
        feat = _make_feature_dict(0.5)
        p_win, expected_r = loaded.predict(feat)
        assert 0.0 <= p_win <= 1.0
        assert isinstance(expected_r, float)

    def test_tampered_file_raises(self, tmp_path: Path) -> None:
        from bot.strategy.score_v3 import TrainedScoreV3Model

        result = self._train_quick()
        out = tmp_path / "model.pkl"
        save_model(result, out)

        # Tamper with the model file
        data = out.read_bytes()
        out.write_bytes(data[:-10] + b"\x00" * 10)

        with pytest.raises(ValueError, match="integrity check failed"):
            TrainedScoreV3Model.load(out)


# ═══════════════════════════════════════════════════════════════════════════
#  F7 — CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


class TestCli:
    """Section F7: main() CLI entrypoint."""

    @pytest.fixture(autouse=True)
    def _require_lightgbm(self) -> None:
        pytest.importorskip("lightgbm")

    def test_cli_missing_data_file(self, tmp_path: Path) -> None:
        from bot.ml.train_v3 import main

        with pytest.raises(SystemExit):
            main(["--data", str(tmp_path / "nonexistent.jsonl")])

    def test_cli_too_few_samples(self, tmp_path: Path) -> None:
        """CLI should raise ValueError when not enough samples."""
        from bot.ml.train_v3 import main

        p = tmp_path / "data.jsonl"
        # Only 5 records — below minimum of 20
        records = [_make_record(shadow_r=1.0) for _ in range(5)]
        _write_jsonl(p, records)

        with pytest.raises(ValueError, match="need >= 20"):
            main(["--data", str(p), "--out", str(tmp_path / "model.pkl")])

    def test_cli_full_run(self, tmp_path: Path) -> None:
        """End-to-end CLI run with enough data."""
        from bot.ml.train_v3 import main

        p = tmp_path / "data.jsonl"
        records = []
        for i in range(30):
            r_val = 1.5 if i < 15 else -0.5
            records.append(_make_record(shadow_r=r_val))
        _write_jsonl(p, records)

        out = tmp_path / "model.pkl"
        main(["--data", str(p), "--out", str(out), "--no-calibrate", "--folds", "3"])

        assert out.exists()
        assert out.with_suffix(".pkl.sig").exists()
