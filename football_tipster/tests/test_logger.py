"""
test_logger.py — unit tests for logger.py (log_bets, settle_bets, _evaluate_result).
"""
import csv
import pytest
import logger
from logger import _evaluate_result, _load_rows, log_bets, settle_bets, FIELDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finished_match(home_goals, away_goals):
    """Build a minimal finished-match API response dict."""
    if home_goals > away_goals:
        winner = "HOME_TEAM"
    elif away_goals > home_goals:
        winner = "AWAY_TEAM"
    else:
        winner = "DRAW"
    return {
        "status": "FINISHED",
        "score": {
            "winner": winner,
            "fullTime": {"home": home_goals, "away": away_goals},
        },
    }


def _scheduled_match():
    return {"status": "SCHEDULED", "score": {"fullTime": {"home": None, "away": None}}}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({field: r.get(field, "") for field in FIELDS})


# ── _evaluate_result ──────────────────────────────────────────────────────────

class TestEvaluateResult:
    # 1X2
    def test_1x2_home_win(self):
        assert _evaluate_result("Home Win", "1X2", _finished_match(2, 0)) == "W"

    def test_1x2_home_loss(self):
        assert _evaluate_result("Home Win", "1X2", _finished_match(0, 1)) == "L"

    def test_1x2_draw_win(self):
        assert _evaluate_result("Draw", "1X2", _finished_match(1, 1)) == "W"

    def test_1x2_draw_loss(self):
        assert _evaluate_result("Draw", "1X2", _finished_match(2, 0)) == "L"

    def test_1x2_away_win(self):
        assert _evaluate_result("Away Win", "1X2", _finished_match(0, 2)) == "W"

    def test_1x2_away_loss(self):
        assert _evaluate_result("Away Win", "1X2", _finished_match(1, 1)) == "L"

    # Double Chance
    def test_dc_1x_home(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(2, 0)) == "W"

    def test_dc_1x_draw(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(1, 1)) == "W"

    def test_dc_1x_away(self):
        assert _evaluate_result("1X (Home or Draw)", "Double Chance", _finished_match(0, 1)) == "L"

    def test_dc_x2_draw(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(0, 0)) == "W"

    def test_dc_x2_away(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(0, 2)) == "W"

    def test_dc_x2_home(self):
        assert _evaluate_result("X2 (Draw or Away)", "Double Chance", _finished_match(3, 0)) == "L"

    def test_dc_12_home(self):
        assert _evaluate_result("12 (Home or Away)", "Double Chance", _finished_match(1, 0)) == "W"

    def test_dc_12_draw(self):
        assert _evaluate_result("12 (Home or Away)", "Double Chance", _finished_match(1, 1)) == "L"

    # Over/Under
    def test_over25_win(self):
        assert _evaluate_result("Over 2.5", "Over/Under", _finished_match(2, 1)) == "W"

    def test_over25_loss(self):
        assert _evaluate_result("Over 2.5", "Over/Under", _finished_match(1, 1)) == "L"

    def test_under25_win(self):
        assert _evaluate_result("Under 2.5", "Over/Under", _finished_match(1, 0)) == "W"

    def test_under25_loss(self):
        assert _evaluate_result("Under 2.5", "Over/Under", _finished_match(2, 1)) == "L"

    def test_over35_win(self):
        assert _evaluate_result("Over 3.5", "Over/Under", _finished_match(2, 2)) == "W"

    def test_over35_loss(self):
        assert _evaluate_result("Over 3.5", "Over/Under", _finished_match(2, 1)) == "L"

    def test_under35_win(self):
        assert _evaluate_result("Under 3.5", "Over/Under", _finished_match(1, 1)) == "W"

    def test_under35_loss(self):
        assert _evaluate_result("Under 3.5", "Over/Under", _finished_match(2, 2)) == "L"

    # BTTS
    def test_btts_yes_win(self):
        assert _evaluate_result("BTTS Yes", "BTTS", _finished_match(1, 1)) == "W"

    def test_btts_yes_loss(self):
        assert _evaluate_result("BTTS Yes", "BTTS", _finished_match(1, 0)) == "L"

    def test_btts_no_win(self):
        assert _evaluate_result("BTTS No", "BTTS", _finished_match(2, 0)) == "W"

    def test_btts_no_loss(self):
        assert _evaluate_result("BTTS No", "BTTS", _finished_match(1, 2)) == "L"

    # Edge cases
    def test_unfinished_returns_none(self):
        assert _evaluate_result("Home Win", "1X2", _scheduled_match()) is None

    def test_missing_score_returns_none(self):
        match = {"status": "FINISHED", "score": {"fullTime": {"home": None, "away": None}}}
        assert _evaluate_result("Home Win", "1X2", match) is None

    def test_unknown_market_returns_none(self):
        assert _evaluate_result("Something", "Unknown", _finished_match(1, 0)) is None


# ── log_bets deduplication ────────────────────────────────────────────────────

class TestLogBets:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def _pick(self, mid="101"):
        return {
            "match_id": mid, "home": "A", "away": "B", "league": "PL",
            "market": "1X2", "pick": "Home Win", "model_prob": 0.55,
            "edge": None,
        }

    def test_creates_file_with_header(self):
        log_bets([self._pick()])
        rows = _load_rows()
        assert len(rows) == 1
        assert rows[0]["match_id"] == "101"

    def test_duplicate_skipped(self):
        log_bets([self._pick()])
        log_bets([self._pick()])   # second call — same match_id
        assert len(_load_rows()) == 1

    def test_different_ids_both_logged(self):
        log_bets([self._pick("101"), self._pick("102")])
        assert len(_load_rows()) == 2

    def test_no_profit_column(self):
        log_bets([self._pick()])
        rows = _load_rows()
        assert "profit" not in rows[0]

    def test_result_blank_on_write(self):
        log_bets([self._pick()])
        assert _load_rows()[0]["result"] == ""


# ── settle_bets ───────────────────────────────────────────────────────────────

class TestSettleBets:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_no_file_returns_zero(self):
        settled, failed = settle_bets()
        assert settled == 0 and failed == 0

    def test_already_settled_skipped(self, tmp_path):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "1", "market": "1X2", "pick": "Home Win", "result": "W"},
        ])
        settled, failed = settle_bets()
        assert settled == 0 and failed == 0

    def test_settle_writes_result(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win", "result": ""},
        ])
        # Patch fetcher.get_match to return a finished 2-0 home win
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: _finished_match(2, 0))
        settled, failed = settle_bets()
        assert settled == 1
        rows = _load_rows()
        assert rows[0]["result"] == "W"

    def test_settle_loss(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win", "result": ""},
        ])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: _finished_match(0, 2))
        settle_bets()
        assert _load_rows()[0]["result"] == "L"

    def test_unfinished_stays_blank(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win", "result": ""},
        ])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: _scheduled_match())
        settled, failed = settle_bets()
        assert settled == 0
        assert _load_rows()[0]["result"] == ""

    def test_api_error_counted_as_failed(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win", "result": ""},
        ])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: (_ for _ in ()).throw(Exception("timeout")))
        settled, failed = settle_bets()
        assert settled == 0 and failed == 1

    def test_settle_attempts_incremented_on_failure(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win",
             "result": "", "settle_attempts": "2"},
        ])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: (_ for _ in ()).throw(Exception("err")))
        settle_bets()
        rows = _load_rows()
        assert int(rows[0]["settle_attempts"]) == 3

    def test_row_skipped_after_max_attempts(self, tmp_path, monkeypatch):
        _write_csv(tmp_path / "bets_log.csv", [
            {"match_id": "999", "market": "1X2", "pick": "Home Win",
             "result": "", "settle_attempts": "10"},
        ])
        import fetcher
        call_count = {"n": 0}
        def fake_get(mid):
            call_count["n"] += 1
            return _finished_match(2, 0)
        monkeypatch.setattr(fetcher, "get_match", fake_get)
        settled, failed = settle_bets()
        assert call_count["n"] == 0   # skipped — never called API
        assert failed == 1

    def test_settle_attempts_zero_on_new_bet(self, tmp_path):
        from logger import log_bets
        pick = {
            "match_id": "101", "home": "A", "away": "B", "league": "PL",
            "market": "1X2", "pick": "Home Win", "model_prob": 0.55,
            "edge": None,
        }
        log_bets([pick])
        rows = _load_rows()
        assert rows[0].get("settle_attempts", "0") == "0"
