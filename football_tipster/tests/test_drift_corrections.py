"""
test_drift_corrections.py — Tests for the drift detector, isotonic corrections,
and their integration with the ml_calibrator.
"""
import csv
import json
import os
import pytest

import drift
import corrections
import ml_calibrator
import logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_log(path, rows, model_version="v1"):
    """Write a bets_log.csv at `path` using the full FIELDS schema."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=logger.FIELDS)
        w.writeheader()
        for r in rows:
            full = {field: "" for field in logger.FIELDS}
            full.update(r)
            full.setdefault("model_version", model_version)
            w.writerow(full)


def _bet(market="Over/Under", pick="Over 2.5", model_prob=0.65,
         result="W", model_version="v1"):
    """Convenience builder for a settled-row dict."""
    return {
        "match_id":     "x",
        "market":       market,
        "pick":         pick,
        "model_prob":   f"{model_prob:.3f}",
        "odds_taken":   "1.80",
        "edge":         "5.0",
        "result":       result,
        "roi":          "0.800" if result == "W" else "-1.000",
        "model_version": model_version,
    }


# ── drift.compute_drift ───────────────────────────────────────────────────────

class TestComputeDrift:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Force the logger to look at the in-tmpdir file
        yield

    def test_empty_log_returns_empty(self):
        assert drift.compute_drift(model_version="v1") == []

    def test_below_min_sample_skipped(self):
        rows = [_bet(model_prob=0.60, result="W") for _ in range(20)]
        _write_log("bets_log.csv", rows)
        assert drift.compute_drift(model_version="v1") == []

    def test_well_calibrated_is_green(self):
        # 30 bets, predicted 60%, actual 60% (18W/12L) → gap 0pp → green
        rows = (
            [_bet(model_prob=0.60, result="W") for _ in range(18)] +
            [_bet(model_prob=0.60, result="L") for _ in range(12)]
        )
        _write_log("bets_log.csv", rows)
        out = drift.compute_drift(model_version="v1")
        assert len(out) == 1
        assert out[0]["severity"] == "green"
        assert abs(out[0]["gap_pp"]) < 1.0

    def test_overconfident_is_red(self):
        # Predicted avg 70%, actual 50% → 20pp gap → red
        rows = (
            [_bet(model_prob=0.70, result="W") for _ in range(15)] +
            [_bet(model_prob=0.70, result="L") for _ in range(15)]
        )
        _write_log("bets_log.csv", rows)
        out = drift.compute_drift(model_version="v1")
        assert out[0]["severity"] == "red"
        assert out[0]["gap_pp"] == pytest.approx(20.0, abs=1.0)

    def test_underconfident_is_yellow_or_red(self):
        # Predicted 50%, actual 65% → gap -15pp → red (boundary)
        rows = (
            [_bet(model_prob=0.50, result="W") for _ in range(20)] +
            [_bet(model_prob=0.50, result="L") for _ in range(10)]
        )
        _write_log("bets_log.csv", rows)
        out = drift.compute_drift(model_version="v1")
        assert out[0]["gap_pp"] < 0           # under-confident
        assert out[0]["severity"] in ("yellow", "red")

    def test_only_current_cohort_counted(self):
        # Mix v1 and v0 rows — only v1 should appear
        v1_rows = [_bet(model_prob=0.60, result="W", model_version="v1") for _ in range(30)]
        v0_rows = [_bet(model_prob=0.60, result="L", model_version="v0") for _ in range(30)]
        _write_log("bets_log.csv", v1_rows + v0_rows)
        out = drift.compute_drift(model_version="v1")
        assert len(out) == 1
        assert out[0]["n"] == 30
        # Actual WR should reflect only v1 (all wins), not the mixed pool
        assert out[0]["actual_wr_pct"] == pytest.approx(100.0)

    def test_sort_red_before_yellow(self):
        # Mix two buckets: one red, one yellow
        rows = (
            # bucket A: red — predicted 75%, actual 50%
            [_bet(market="A", model_prob=0.75, result="W") for _ in range(15)] +
            [_bet(market="A", model_prob=0.75, result="L") for _ in range(15)] +
            # bucket B: yellow — predicted 60%, actual 48% (12pp)
            [_bet(market="B", model_prob=0.60, result="W") for _ in range(14)] +
            [_bet(market="B", model_prob=0.60, result="L") for _ in range(16)]
        )
        _write_log("bets_log.csv", rows)
        out = drift.compute_drift(model_version="v1")
        assert out[0]["severity"] == "red"
        assert out[1]["severity"] == "yellow"

    def test_severity_for_helper(self):
        assert drift.severity_for(0.0)   == "green"
        assert drift.severity_for(9.9)   == "green"
        assert drift.severity_for(10.0)  == "yellow"
        assert drift.severity_for(14.9)  == "yellow"
        assert drift.severity_for(15.0)  == "red"
        assert drift.severity_for(-20.0) == "red"

    def test_has_actionable(self):
        green = [{"severity": "green"}]
        mixed = [{"severity": "green"}, {"severity": "yellow"}]
        assert drift.has_actionable(green) is False
        assert drift.has_actionable(mixed) is True
        assert drift.has_actionable([]) is False


# ── corrections.fit / save / apply round-trip ─────────────────────────────────

class TestCorrections:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        corrections.invalidate_cache()
        yield
        corrections.invalidate_cache()

    def test_apply_when_no_store_is_identity(self):
        assert corrections.apply_correction(0.6, "Over/Under", "Over 2.5") == 0.6

    def test_apply_out_of_range_unchanged(self):
        assert corrections.apply_correction(None, "M", "P") is None
        assert corrections.apply_correction(-0.5, "M", "P") == -0.5
        assert corrections.apply_correction(1.5,  "M", "P") == 1.5

    def test_fit_returns_none_below_min_sample(self):
        rows = [_bet(model_prob=0.60, result="W") for _ in range(20)]
        _write_log("bets_log.csv", rows)
        assert corrections.fit_correction("Over/Under", "Over 2.5") is None

    def test_fit_save_apply_round_trip(self):
        # Build a clearly-biased bucket: model says 70%, reality is 50%
        rows = (
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.65, result="W") for _ in range(15)] +
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.65, result="L") for _ in range(15)] +
            # Also include a couple of higher-prob bets so the regression has range
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.75, result="W") for _ in range(8)] +
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.75, result="L") for _ in range(8)]
        )
        _write_log("bets_log.csv", rows)

        entry = corrections.fit_correction("Over/Under", "Over 2.5")
        if entry is None:
            pytest.skip("sklearn not installed")
        assert entry["n_samples"] == 46
        assert len(entry["raw_probs"]) == len(entry["calibrated"])

        corrections.save_correction(entry)
        # File should exist
        assert os.path.exists(corrections._STORE_FILE)
        with open(corrections._STORE_FILE) as f:
            data = json.load(f)
        assert "Over/Under||Over 2.5" in data

        # Applied correction should pull a 65%-input down toward ~50% (the actual)
        corrected = corrections.apply_correction(0.65, "Over/Under", "Over 2.5")
        assert corrected < 0.65    # reduced (model was over-confident)
        # And an unrelated bucket should be untouched
        assert corrections.apply_correction(0.65, "Other", "Pick") == 0.65

    def test_delete_correction_removes_entry(self):
        rows = (
            [_bet(market="X", pick="Y", model_prob=0.50, result="W") for _ in range(20)] +
            [_bet(market="X", pick="Y", model_prob=0.70, result="L") for _ in range(15)]
        )
        _write_log("bets_log.csv", rows)
        entry = corrections.fit_correction("X", "Y")
        if entry is None:
            pytest.skip("sklearn not installed")
        corrections.save_correction(entry)
        assert any(c["market"] == "X" for c in corrections.list_corrections())
        corrections.delete_correction("X", "Y")
        assert not any(c["market"] == "X" for c in corrections.list_corrections())


# ── ml_calibrator integration ─────────────────────────────────────────────────

class TestCalibratorWithCorrection:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        corrections.invalidate_cache()
        ml_calibrator.reset_calibrator()
        yield
        corrections.invalidate_cache()
        ml_calibrator.reset_calibrator()

    def test_calibrate_no_correction_no_history_is_identity(self):
        """With no corrections and empty bets_log, calibrate returns input."""
        cal = ml_calibrator.get_calibrator()
        out = cal.calibrate(0.65, "Over/Under", "PL", "Over 2.5")
        assert abs(out - 0.65) < 1e-6

    def test_calibrate_applies_saved_correction(self):
        # Build a biased bucket and save a correction
        rows = (
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.65, result="W") for _ in range(10)] +
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.65, result="L") for _ in range(20)] +
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.75, result="W") for _ in range(5)] +
            [_bet(market="Over/Under", pick="Over 2.5", model_prob=0.75, result="L") for _ in range(10)]
        )
        _write_log("bets_log.csv", rows)
        entry = corrections.fit_correction("Over/Under", "Over 2.5")
        if entry is None:
            pytest.skip("sklearn not installed")
        corrections.save_correction(entry)
        ml_calibrator.reset_calibrator()

        cal = ml_calibrator.get_calibrator()
        # Input 0.65 should be pulled down toward the actual WR (~0.33)
        # but clamped at most 15pp by the ±15pp safety rail.
        out = cal.calibrate(0.65, "Over/Under", "PL", "Over 2.5")
        assert out < 0.65            # reduced
        assert out >= 0.50           # clamped at 0.65 - 0.15
