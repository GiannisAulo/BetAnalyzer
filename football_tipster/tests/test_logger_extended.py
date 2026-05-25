"""
test_logger_extended.py — additional edge case tests for logger.py.
Covers: ROI summary, log_bets deduplication, evaluate_result edge cases.
"""
import csv
import os
import pytest
import logger
from logger import (
    _evaluate_result, _load_rows, log_bets, FIELDS,
    compute_roi_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({field: r.get(field, "") for field in FIELDS})


def _finished_match(home_goals, away_goals):
    winner = "HOME_TEAM" if home_goals > away_goals else (
        "AWAY_TEAM" if away_goals > home_goals else "DRAW"
    )
    return {
        "status": "FINISHED",
        "score": {"winner": winner, "fullTime": {"home": home_goals, "away": away_goals}},
    }


# ── _evaluate_result edge cases ───────────────────────────────────────────────

class TestEvaluateResultEdgeCases:
    def test_over15_wins_when_2_goals(self):
        assert _evaluate_result("Over 1.5", "Over/Under", _finished_match(1, 1)) == "W"

    def test_over15_loses_when_1_goal(self):
        assert _evaluate_result("Over 1.5", "Over/Under", _finished_match(1, 0)) == "L"

    def test_under15_wins_when_1_goal(self):
        assert _evaluate_result("Under 1.5", "Over/Under", _finished_match(1, 0)) == "W"

    def test_under15_loses_when_2_goals(self):
        assert _evaluate_result("Under 1.5", "Over/Under", _finished_match(1, 1)) == "L"

    def test_over35_wins_when_4_goals(self):
        assert _evaluate_result("Over 3.5", "Over/Under", _finished_match(2, 2)) == "W"

    def test_over35_loses_when_3_goals(self):
        assert _evaluate_result("Over 3.5", "Over/Under", _finished_match(2, 1)) == "L"

    def test_under35_wins_when_3_goals(self):
        assert _evaluate_result("Under 3.5", "Over/Under", _finished_match(2, 1)) == "W"

    def test_btts_yes_wins_when_both_score(self):
        assert _evaluate_result("BTTS Yes", "BTTS", _finished_match(1, 2)) == "W"

    def test_btts_yes_loses_when_clean_sheet(self):
        assert _evaluate_result("BTTS Yes", "BTTS", _finished_match(2, 0)) == "L"

    def test_btts_no_wins_when_clean_sheet(self):
        assert _evaluate_result("BTTS No", "BTTS", _finished_match(2, 0)) == "W"

    def test_btts_no_loses_when_both_score(self):
        assert _evaluate_result("BTTS No", "BTTS", _finished_match(1, 1)) == "L"

    def test_draw_result_pick(self):
        assert _evaluate_result("Draw", "1X2", _finished_match(1, 1)) == "W"

    def test_draw_pick_loses_on_home_win(self):
        assert _evaluate_result("Draw", "1X2", _finished_match(2, 1)) == "L"

    def test_1x_wins_on_draw(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(0, 0)) == "W"

    def test_1x_wins_on_home_win(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(2, 0)) == "W"

    def test_1x_loses_on_away_win(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(0, 2)) == "L"

    def test_x2_wins_on_draw(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(1, 1)) == "W"

    def test_x2_wins_on_away_win(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(0, 1)) == "W"

    def test_x2_loses_on_home_win(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(3, 0)) == "L"

    def test_12_wins_on_home_win(self):
        assert _evaluate_result("12 (Home or Away)", "Double Chance", _finished_match(2, 0)) == "W"

    def test_12_wins_on_away_win(self):
        assert _evaluate_result("12 (Home or Away)", "Double Chance", _finished_match(0, 1)) == "W"

    def test_12_loses_on_draw(self):
        assert _evaluate_result("12 (Home or Away)", "Double Chance", _finished_match(1, 1)) == "L"

    def test_unfinished_match_returns_none(self):
        match = {"status": "SCHEDULED", "score": {"fullTime": {"home": None, "away": None}}}
        assert _evaluate_result("Home Win", "1X2", match) is None

    def test_unknown_pick_returns_none(self):
        assert _evaluate_result("Mystery Pick", "Unknown", _finished_match(1, 0)) is None


# ── ROI summary ───────────────────────────────────────────────────────────────

class TestROISummary:
    def test_returns_none_when_fewer_than_5_bets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_csv(tmp_path / "bets_log.csv", [
            {"result": "W", "odds_taken": "2.00", "roi": "1.000"} for _ in range(4)
        ])
        assert compute_roi_summary() is None

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert compute_roi_summary() is None

    def test_counts_wins_and_losses(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = (
            [{"result": "W", "odds_taken": "2.00", "roi": "1.000"}] * 3 +
            [{"result": "L", "odds_taken": "2.00", "roi": "-1.000"}] * 3
        )
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary is not None
        assert summary["wins"] == 3
        assert summary["losses"] == 3
        assert summary["total"] == 6

    def test_roi_pct_calculated_correctly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = (
            [{"result": "W", "odds_taken": "2.00", "roi": "1.000"}] * 5 +
            [{"result": "L", "odds_taken": "2.00", "roi": "-1.000"}] * 5
        )
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary["roi_pct"] == pytest.approx(0.0, abs=0.01)

    def test_all_wins_positive_roi(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = [{"result": "W", "odds_taken": "2.00", "roi": "1.000"}] * 5
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary["roi_pct"] > 0

    def test_all_losses_negative_roi(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = [{"result": "L", "odds_taken": "2.00", "roi": "-1.000"}] * 5
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary["roi_pct"] < 0

    def test_skips_rows_without_roi(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = (
            [{"result": "W", "odds_taken": "2.00", "roi": "1.000"}] * 5 +
            [{"result": "W", "odds_taken": "",      "roi": ""}] * 10   # no roi
        )
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary["total"] == 5   # only those with roi counted

    def test_skips_unsettled_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = (
            [{"result": "W", "odds_taken": "2.00", "roi": "1.000"}] * 5 +
            [{"result": "",  "odds_taken": "2.00", "roi": ""}] * 5   # unsettled
        )
        _write_csv(tmp_path / "bets_log.csv", rows)
        summary = compute_roi_summary()
        assert summary["total"] == 5

    def test_filter_by_model_version(self, tmp_path, monkeypatch):
        """Only rows tagged with the requested model_version should be counted."""
        monkeypatch.chdir(tmp_path)
        rows = (
            # Old cohort: 5 losses
            [{"result": "L", "odds_taken": "2.00", "roi": "-1.000", "model_version": "v2026-01-01"}] * 5 +
            # New cohort: 6 wins
            [{"result": "W", "odds_taken": "2.00", "roi": "1.000",  "model_version": "v2026-05-24"}] * 6
        )
        _write_csv(tmp_path / "bets_log.csv", rows)

        # Lifetime sees all 11
        lifetime = compute_roi_summary()
        assert lifetime["total"] == 11
        assert lifetime["wins"] == 6 and lifetime["losses"] == 5

        # Filter to new cohort only
        current = compute_roi_summary(model_version="v2026-05-24")
        assert current["total"] == 6
        assert current["wins"] == 6 and current["losses"] == 0
        assert current["roi_pct"] > 0

        # Filter to old cohort only
        old = compute_roi_summary(model_version="v2026-01-01")
        assert old["total"] == 5
        assert old["losses"] == 5
        assert old["roi_pct"] < 0

    def test_filter_unknown_version_returns_none(self, tmp_path, monkeypatch):
        """Requesting a version with fewer than 5 matching bets → None."""
        monkeypatch.chdir(tmp_path)
        rows = [{"result": "W", "odds_taken": "2.00", "roi": "1.000",
                 "model_version": "v2026-05-24"}] * 6
        _write_csv(tmp_path / "bets_log.csv", rows)
        assert compute_roi_summary(model_version="v9999-99-99") is None


# ── log_bets deduplication ────────────────────────────────────────────────────

class TestLogBetsDeduplication:
    def _make_pick(self, match_id="123"):
        return {
            "match_id": match_id, "home": "Arsenal", "away": "Chelsea",
            "league": "PL", "market": "1X2", "pick": "Home Win",
            "model_prob": 0.62, "edge": 7.5,
        }

    def test_same_match_id_not_logged_twice(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pick = self._make_pick("999")
        log_bets([pick])
        log_bets([pick])
        rows = _load_rows()
        assert len(rows) == 1

    def test_different_match_ids_both_logged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        log_bets([self._make_pick("101")])
        log_bets([self._make_pick("102")])
        rows = _load_rows()
        assert len(rows) == 2

    def test_empty_match_id_logged_once(self, tmp_path, monkeypatch):
        """Pick with no match_id (empty string) must still be logged."""
        monkeypatch.chdir(tmp_path)
        pick = self._make_pick("")
        log_bets([pick])
        rows = _load_rows()
        assert len(rows) == 1

    def test_log_creates_file_if_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not os.path.exists("bets_log.csv")
        log_bets([self._make_pick("555")])
        assert os.path.exists("bets_log.csv")

    def test_logged_pick_has_correct_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        log_bets([self._make_pick("777")])
        rows = _load_rows()
        assert rows[0]["market"] == "1X2"
        assert rows[0]["pick"] == "Home Win"
        assert rows[0]["league"] == "PL"


