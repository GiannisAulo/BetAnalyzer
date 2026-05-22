"""Tests for match_store.py — SQLite persistence layer (E.3)."""

import pytest
import tempfile
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import match_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_db():
    """Return a Path to a fresh temporary SQLite database."""
    tmp = tempfile.mktemp(suffix=".db")
    return Path(tmp)


def _api_match(match_id, home_id, away_id, hg, ag, winner, utc_date,
               league="PL", stage="REGULAR_SEASON"):
    return {
        "id": match_id,
        "status": "FINISHED",
        "utcDate": utc_date,
        "stage": stage,
        "competition": {"code": league},
        "homeTeam": {"id": home_id},
        "awayTeam": {"id": away_id},
        "score": {
            "fullTime": {"home": hg, "away": ag},
            "winner": winner,
        },
    }


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_table(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        assert db.exists()
        count = match_store.match_count(db_path=db)
        assert count == 0

    def test_idempotent(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        match_store.init_db(db_path=db)   # should not raise
        assert match_store.match_count(db_path=db) == 0


# ---------------------------------------------------------------------------
# store_matches
# ---------------------------------------------------------------------------

class TestStoreMatches:
    def test_stores_finished_match(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        data = {"matches": [_api_match(1, 10, 20, 2, 1, "HOME_TEAM", "2025-10-01T15:00:00Z")]}
        match_store.store_matches(data, league="PL", db_path=db)
        assert match_store.match_count(db_path=db) == 1

    def test_ignores_unfinished_match(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _api_match(2, 10, 20, 0, 0, "DRAW", "2025-10-01T15:00:00Z")
        m["status"] = "SCHEDULED"
        data = {"matches": [m]}
        match_store.store_matches(data, league="PL", db_path=db)
        assert match_store.match_count(db_path=db) == 0

    def test_ignores_match_with_missing_score(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _api_match(3, 10, 20, None, None, "", "2025-10-01T15:00:00Z")
        m["score"]["fullTime"]["home"] = None
        data = {"matches": [m]}
        match_store.store_matches(data, league="PL", db_path=db)
        assert match_store.match_count(db_path=db) == 0

    def test_deduplicates_on_reingest(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _api_match(4, 10, 20, 1, 0, "HOME_TEAM", "2025-10-01T15:00:00Z")
        data = {"matches": [m]}
        match_store.store_matches(data, db_path=db)
        match_store.store_matches(data, db_path=db)   # second call — same match
        assert match_store.match_count(db_path=db) == 1

    def test_stores_multiple_matches(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        data = {"matches": [
            _api_match(5, 10, 20, 2, 0, "HOME_TEAM", "2025-10-01T15:00:00Z"),
            _api_match(6, 30, 40, 1, 1, "DRAW",      "2025-10-02T15:00:00Z"),
            _api_match(7, 50, 60, 0, 2, "AWAY_TEAM", "2025-10-03T15:00:00Z"),
        ]}
        match_store.store_matches(data, db_path=db)
        assert match_store.match_count(db_path=db) == 3

    def test_handles_empty_data(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        match_store.store_matches({}, db_path=db)      # no-op
        match_store.store_matches(None, db_path=db)    # no-op
        assert match_store.match_count(db_path=db) == 0


# ---------------------------------------------------------------------------
# get_team_match_history
# ---------------------------------------------------------------------------

class TestGetTeamMatchHistory:
    def _setup(self, db):
        match_store.init_db(db_path=db)
        data = {"matches": [
            _api_match(10, 1, 2, 3, 0, "HOME_TEAM", "2025-10-05T15:00:00Z", league="PL"),
            _api_match(11, 3, 1, 1, 1, "DRAW",      "2025-10-12T15:00:00Z", league="PL"),
            _api_match(12, 1, 4, 0, 2, "AWAY_TEAM", "2025-10-19T15:00:00Z", league="PL"),
            _api_match(13, 5, 6, 2, 2, "DRAW",      "2025-10-20T15:00:00Z", league="PL"),
        ]}
        match_store.store_matches(data, db_path=db)

    def test_returns_matches_for_team(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(1, db_path=db)
        assert len(results) == 3   # matches 10, 11, 12 involve team 1

    def test_excludes_unrelated_teams(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(6, db_path=db)
        assert len(results) == 1
        assert results[0]["id"] == 13

    def test_returns_newest_first(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(1, db_path=db)
        dates = [r["utcDate"] for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_respects_limit(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(1, limit=2, db_path=db)
        assert len(results) == 2

    def test_result_structure(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(1, db_path=db)
        r = results[0]
        assert "id" in r
        assert "utcDate" in r
        assert r["status"] == "FINISHED"
        assert "homeTeam" in r and "id" in r["homeTeam"]
        assert "awayTeam" in r and "id" in r["awayTeam"]
        assert "score" in r
        assert "fullTime" in r["score"]
        assert "home" in r["score"]["fullTime"]
        assert "away" in r["score"]["fullTime"]

    def test_empty_for_unknown_team(self):
        db = _tmp_db()
        self._setup(db)
        results = match_store.get_team_match_history(999, db_path=db)
        assert results == []


# ---------------------------------------------------------------------------
# merge_match_history
# ---------------------------------------------------------------------------

class TestMergeMatchHistory:
    def _make(self, match_id, utc_date):
        return {"id": match_id, "utcDate": utc_date}

    def test_api_matches_first(self):
        api = [self._make(1, "2025-11-01"), self._make(2, "2025-10-01")]
        db  = [self._make(3, "2025-09-01"), self._make(4, "2025-08-01")]
        merged = match_store.merge_match_history(api, db)
        assert [m["id"] for m in merged] == [1, 2, 3, 4]

    def test_deduplicates(self):
        api = [self._make(1, "2025-11-01"), self._make(2, "2025-10-01")]
        db  = [self._make(2, "2025-10-01"), self._make(3, "2025-09-01")]
        merged = match_store.merge_match_history(api, db)
        ids = [m["id"] for m in merged]
        assert ids.count(2) == 1
        assert len(merged) == 3

    def test_sorted_newest_first(self):
        api = [self._make(3, "2025-09-01")]
        db  = [self._make(1, "2025-11-01"), self._make(2, "2025-10-01")]
        merged = match_store.merge_match_history(api, db)
        dates = [m["utcDate"] for m in merged]
        assert dates == sorted(dates, reverse=True)

    def test_empty_api(self):
        db = [self._make(1, "2025-10-01")]
        merged = match_store.merge_match_history([], db)
        assert len(merged) == 1

    def test_empty_db(self):
        api = [self._make(1, "2025-10-01")]
        merged = match_store.merge_match_history(api, [])
        assert len(merged) == 1

    def test_both_empty(self):
        merged = match_store.merge_match_history([], [])
        assert merged == []
