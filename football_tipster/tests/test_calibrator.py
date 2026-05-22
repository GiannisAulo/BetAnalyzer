"""
test_calibrator.py — unit tests for the ML calibration module.
"""
import csv
import os
import tempfile
import pytest
import ml_calibrator
from ml_calibrator import CalibrationModel, _parse_result, _load_history


# ── Result parser ─────────────────────────────────────────────────────────────

class TestParseResult:
    @pytest.mark.parametrize("raw,expected", [
        ("W", 1), ("WIN", 1), ("1", 1), ("YES", 1),
        ("w", 1), ("win", 1),
        ("L", 0), ("LOSS", 0), ("0", 0), ("NO", 0),
        ("l", 0), ("loss", 0),
        ("",    None), ("TBD", None), ("  ", None), (None, None),
    ])
    def test_parse_result(self, raw, expected):
        assert _parse_result(raw) == expected


# ── Helpers to write a temp log ───────────────────────────────────────────────

def _write_log(rows, path):
    """Write sample rows to a CSV at path."""
    fieldnames = ["match_id", "date", "home", "away", "league", "market",
                  "pick", "model_prob", "edge", "result", "settle_attempts"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({
                "match_id":        r.get("match_id", str(i + 1)),
                "date":            "2026-01-01",
                "home":            r.get("home", "Team A"),
                "away":            r.get("away", "Team B"),
                "league":          r.get("league", "PL"),
                "market":          r.get("market", "1X2"),
                "pick":            r.get("pick", "Home Win"),
                "model_prob":      r.get("model_prob", 0.55),
                "edge":            r.get("edge", 5.0),
                "result":          r.get("result", ""),
                "settle_attempts": r.get("settle_attempts", 0),
            })


# ── CalibrationModel with no data ─────────────────────────────────────────────

class TestNoData:
    def test_is_not_active(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        cal = CalibrationModel()
        assert not cal.is_active

    def test_calibrate_returns_input_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        cal = CalibrationModel()
        assert cal.calibrate(0.58, "BTTS", "PL") == 0.58



# ── CalibrationModel with sample data ────────────────────────────────────────

class TestWithData:
    """
    Uses 199 samples total (< MIN_LR_SIZE=200) to test segment-only calibration.
    - 100 BTTS Yes wins (PL)  → exactly reaches MIN_SEGMENT_SIZE (100), win_rate=1.0
    - 99  Over/Under Over 2.5 losses (BL1) → just below MIN_SEGMENT_SIZE — segment inactive
    - 5   incomplete 1X2 rows → skipped
    Per-league LR is also skipped: PL (100, all wins) and BL1 (99, all losses)
    each have only one class, so LogisticRegression cannot be fitted.
    """
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()

        rows = (
            [{"market": "BTTS",        "pick": "BTTS Yes",  "league": "PL",  "model_prob": 0.60, "result": "W"}] * 100
            + [{"market": "Over/Under", "pick": "Over 2.5",  "league": "BL1", "model_prob": 0.58, "result": "L"}] * 99
            # incomplete rows should be skipped
            + [{"market": "1X2",        "pick": "Home Win",  "league": "PL",  "model_prob": 0.55, "result": ""}] * 5
        )
        _write_log(rows, tmp_path / "bets_log.csv")
        self.cal = CalibrationModel()

    def test_sample_count(self):
        """Only completed bets are counted (5 incomplete rows excluded)."""
        assert self.cal.sample_count == 199

    def test_is_active(self):
        assert self.cal.is_active

    def test_no_lr_below_min_lr_size(self):
        """With 199 completed samples (< MIN_LR_SIZE=200), LR should NOT be trained."""
        assert not self.cal.uses_lr

    def test_summary_shows_markets(self):
        rows = self.cal.summary()
        markets_in_summary = {r[0] for r in rows}
        assert "BTTS" in markets_in_summary

    def test_calibrate_shifts_toward_win_rate(self):
        """BTTS Yes segment win rate = 100% → calibrated prob should be ≥ raw."""
        raw = 0.58
        cal_prob = self.cal.calibrate(raw, "BTTS", "PL", "BTTS Yes")
        assert cal_prob >= raw

    def test_calibrate_shifts_down_for_losing_market(self):
        """Over 2.5 BL1 segment win rate = 0% → calibrated prob should be ≤ raw."""
        raw = 0.58
        cal_prob = self.cal.calibrate(raw, "Over/Under", "BL1", "Over 2.5")
        assert cal_prob <= raw

    def test_calibrate_clamps_to_valid_range(self):
        """Calibrated probability must stay in (0, 1)."""
        for raw in [0.01, 0.50, 0.99]:
            result = self.cal.calibrate(raw, "BTTS", "PL", "BTTS Yes")
            assert 0.0 < result < 1.0

    def test_calibrate_max_drift(self):
        """Calibration must never move model_prob by more than 15pp."""
        raw = 0.55
        result = self.cal.calibrate(raw, "BTTS", "PL", "BTTS Yes")
        assert abs(result - raw) <= 0.15 + 1e-9



class TestCalibratePenalty:
    """Over/Under 0% win rate → calibration lowers the probability."""
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        rows = [
            {"market": "Over/Under", "league": "PL", "model_prob": 0.58, "result": "L"}
        ] * 20
        _write_log(rows, tmp_path / "bets_log.csv")
        self.cal = CalibrationModel()

    def test_calibrate_shifts_down(self):
        """Global rate = 0%, so calibration lowers the probability."""
        raw = 0.58
        result = self.cal.calibrate(raw, "Over/Under", "PL")
        assert result < raw


# ── Singleton behaviour ───────────────────────────────────────────────────────

class TestSingleton:
    def test_get_calibrator_returns_same_instance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        a = ml_calibrator.get_calibrator()
        b = ml_calibrator.get_calibrator()
        assert a is b

    def test_reset_calibrator_creates_new_instance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        a = ml_calibrator.get_calibrator()
        ml_calibrator.reset_calibrator()
        b = ml_calibrator.get_calibrator()
        assert a is not b


# ── Load history helper ───────────────────────────────────────────────────────

class TestLoadHistory:
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        records = _load_history()
        assert records == []

    def test_skips_incomplete_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_log([
            {"result": "W",   "model_prob": 0.55},
            {"result": "",    "model_prob": 0.55},   # incomplete
            {"result": "TBD", "model_prob": 0.55},   # unrecognised
        ], tmp_path / "bets_log.csv")
        records = _load_history()
        assert len(records) == 1

    def test_parses_model_prob_correctly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_log([{"result": "W", "model_prob": 0.612}], tmp_path / "bets_log.csv")
        records = _load_history()
        assert abs(records[0]["model_prob"] - 0.612) < 1e-9

    def test_pick_field_loaded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_log([{"result": "W", "model_prob": 0.60, "pick": "Over 2.5"}],
                   tmp_path / "bets_log.csv")
        records = _load_history()
        assert records[0]["pick"] == "Over 2.5"


# ── Per-direction calibration buckets ────────────────────────────────────────

class TestPerDirectionBuckets:
    """Over 2.5 and Under 2.5 must have separate calibration segments.
    MIN_SEGMENT_SIZE=100. Use 100 Over wins (PL) + 99 Under losses (BL1).
    Leagues are split so each league has a single class → per-league LR is
    skipped for both, leaving segment + global calibration active.
    Under 2.5 has 99 samples (< 100) → no per-direction segment, but the
    global Over/Under rate is slightly above 50%, so Under still
    calibrates via global signal."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        rows = (
            [{"market": "Over/Under", "pick": "Over 2.5",  "league": "PL",
              "model_prob": 0.60, "result": "W"}] * 100
            + [{"market": "Over/Under", "pick": "Under 2.5", "league": "BL1",
                "model_prob": 0.60, "result": "L"}] * 99
        )
        # PL: 100 wins (single class) → per-league LR skipped.
        # BL1: 99 losses (single class) → per-league LR skipped.
        _write_log(rows, tmp_path / "bets_log.csv")
        self.cal = CalibrationModel()

    def test_over_has_separate_segment(self):
        """Over 2.5 with 100 samples reaches MIN_SEGMENT_SIZE=100."""
        key_over = ("Over/Under", "Over 2.5", "PL")
        assert key_over in self.cal._seg

    def test_under_below_segment_threshold(self):
        """Under 2.5 BL1 with 99 samples is below MIN_SEGMENT_SIZE=100 — no segment."""
        key_under = ("Over/Under", "Under 2.5", "BL1")
        assert key_under not in self.cal._seg

    def test_over_calibrated_up(self):
        """Over 2.5 PL segment win rate = 100% → calibrated prob should be ≥ raw."""
        raw = 0.55
        over_cal = self.cal.calibrate(raw, "Over/Under", "PL", "Over 2.5")
        assert over_cal >= raw

    def test_under_returns_identity_without_segment(self):
        """Under 2.5 BL1 has no segment (24 < 25 min) and global is neutral → identity."""
        raw = 0.55
        under_cal = self.cal.calibrate(raw, "Over/Under", "BL1", "Under 2.5")
        # Global win rate ≈ 51% (neutral), segment absent → no calibration applied
        assert abs(under_cal - raw) < 0.05


# ── Calibration uncertainty ───────────────────────────────────────────────────

class TestCalibrationUncertainty:
    def test_zero_samples_returns_zero(self):
        assert ml_calibrator.calibration_uncertainty(0.6, 0) == 0.0

    def test_large_sample_small_uncertainty(self):
        unc = ml_calibrator.calibration_uncertainty(0.6, 500)
        assert unc < 0.05   # < 5pp for 500 samples

    def test_small_sample_large_uncertainty(self):
        unc = ml_calibrator.calibration_uncertainty(0.6, 10)
        assert unc > 0.10   # > 10pp for only 10 samples

    def test_result_in_valid_range(self):
        for p in [0.1, 0.5, 0.9]:
            for n in [5, 20, 100]:
                unc = ml_calibrator.calibration_uncertainty(p, n)
                assert 0.0 <= unc <= 0.5


# ── Per-league LR (Tier 3) ────────────────────────────────────────────────────

class TestPerLeagueLR:
    """
    Tier 3: per-league LR activates when >= MIN_LR_LEAGUE_SIZE (75) bets
    exist for that league. Tests use two separate leagues to confirm
    isolation — PL data must not affect BL1 model and vice versa.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()

        # PL: 75 rows (both classes) → above MIN_LR_LEAGUE_SIZE (75)
        # BL1: 75 rows (both classes) → above MIN_LR_LEAGUE_SIZE (75)
        # SA: only 40 rows → below MIN_LR_LEAGUE_SIZE (75), no model
        pl_rows  = [{"league": "PL",  "market": "1X2",      "pick": "Home Win",
                     "model_prob": 0.60, "result": "W"}] * 45 + \
                   [{"league": "PL",  "market": "1X2",      "pick": "Home Win",
                     "model_prob": 0.50, "result": "L"}] * 30
        bl1_rows = [{"league": "BL1", "market": "Over/Under", "pick": "Over 2.5",
                     "model_prob": 0.58, "result": "W"}] * 38 + \
                   [{"league": "BL1", "market": "Over/Under", "pick": "Over 2.5",
                     "model_prob": 0.58, "result": "L"}] * 37
        sa_rows  = [{"league": "SA",  "market": "BTTS",     "pick": "BTTS Yes",
                     "model_prob": 0.55, "result": "W"}] * 40

        _write_log(pl_rows + bl1_rows + sa_rows, tmp_path / "bets_log.csv")
        self.cal = CalibrationModel()

    def test_pl_has_league_lr(self):
        assert "PL" in self.cal.league_lr_leagues

    def test_bl1_has_league_lr(self):
        assert "BL1" in self.cal.league_lr_leagues

    def test_sa_below_threshold_no_league_lr(self):
        """SA has only 40 bets (< 75) → no per-league model."""
        assert "SA" not in self.cal.league_lr_leagues

    def test_league_lr_sample_count_pl(self):
        assert self.cal.league_lr_sample_count("PL") == 75

    def test_league_lr_sample_count_unknown_returns_zero(self):
        assert self.cal.league_lr_sample_count("XYZ") == 0

    def test_calibrate_uses_per_league_model(self):
        """PL calibration should return a different result than identity."""
        raw = 0.58
        cal_prob = self.cal.calibrate(raw, "1X2", "PL", "Home Win")
        # Per-league LR is active — result should differ from raw
        assert cal_prob != pytest.approx(raw, abs=1e-6)

    def test_calibrate_stays_within_clamp(self):
        """Per-league LR output must still respect the ±15pp hard clamp."""
        for raw in [0.40, 0.60, 0.80]:
            result = self.cal.calibrate(raw, "1X2", "PL", "Home Win")
            assert abs(result - raw) <= 0.15 + 1e-9

    def test_calibrate_in_valid_range(self):
        """Output must always be a valid probability (0, 1)."""
        result = self.cal.calibrate(0.55, "1X2", "PL", "Home Win")
        assert 0.0 < result < 1.0

    def test_league_models_are_independent(self):
        """PL and BL1 calibrated values for same input must differ."""
        raw = 0.58
        pl_cal  = self.cal.calibrate(raw, "1X2",       "PL",  "Home Win")
        bl1_cal = self.cal.calibrate(raw, "Over/Under", "BL1", "Over 2.5")
        # Different leagues + different data → different outputs
        assert pl_cal != pytest.approx(bl1_cal, abs=1e-6)

    def test_unknown_league_falls_through_to_global_or_segment(self):
        """A league with no per-league model must not crash — returns valid prob."""
        result = self.cal.calibrate(0.55, "1X2", "CL", "Home Win")
        assert 0.0 < result < 1.0

    def test_league_lr_leagues_property_sorted(self):
        leagues = self.cal.league_lr_leagues
        assert leagues == sorted(leagues)
