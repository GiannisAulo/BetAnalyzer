"""
test_phase2_xg.py — Tests for Phase 2: xG proxy from shots-on-target.

Covers:
  2.2  SQLite schema migration (home_shots_on_target, away_shots_on_target)
  2.3  xG proxy computation in parse_team_history
  2.4  xG blend into Poisson lambda in compute_match_probabilities
  2.5  Shot stats persistence and round-trip through match_store

Test categories per feature:
  - Happy path: shot data flows through correctly
  - Edge cases: partial data, minimum thresholds, boundary values
  - Negative scenarios: no shot data, graceful fallback to goals
"""
import pytest
import tempfile
from pathlib import Path
import math
from datetime import datetime, timedelta, timezone

import analyzer
import match_store


def _days_ago(n: int) -> str:
    """Return an ISO 8601 UTC timestamp string for N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.strftime("%Y-%m-%dT12:00:00Z")


# ── Shared helpers ───────────────────────────────────────────────────────────

def _tmp_db():
    """Return a Path to a fresh temporary SQLite database."""
    return Path(tempfile.mktemp(suffix=".db"))


def _make_match(match_id, home_id, away_id, hg, ag, winner, utc_date,
                home_sot=None, away_sot=None, league="PL"):
    """Build a match dict matching the API response structure."""
    m = {
        "id": match_id,
        "status": "FINISHED",
        "utcDate": utc_date,
        "stage": "REGULAR_SEASON",
        "competition": {"code": league},
        "homeTeam": {"id": home_id},
        "awayTeam": {"id": away_id},
        "score": {
            "fullTime": {"home": hg, "away": ag},
            "winner": winner,
        },
    }
    if home_sot is not None:
        m["home_shots_on_target"] = home_sot
    if away_sot is not None:
        m["away_shots_on_target"] = away_sot
    return m


def _matches_data(matches):
    """Wrap a list of match dicts in the API response structure."""
    return {"matches": matches}


# =============================================================================
# 2.2  SQLite schema migration
# =============================================================================

class TestSchemaMigration:
    """Verify the schema includes shot stats columns after init_db."""

    def test_new_db_has_shot_columns(self):
        """A fresh database should have home_shots_on_target and away_shots_on_target."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        import sqlite3
        conn = sqlite3.connect(str(db))
        cursor = conn.execute("PRAGMA table_info(matches)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "home_shots_on_target" in columns
        assert "away_shots_on_target" in columns

    def test_migration_is_idempotent(self):
        """Running init_db twice doesn't fail on the ALTER TABLE migration."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        match_store.init_db(db_path=db)  # should not raise

    def test_store_match_with_shots(self):
        """A match with shot data should persist shots to the DB."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(100, 1, 2, 3, 1, "HOME_TEAM", "2026-01-15T15:00:00Z",
                        home_sot=7, away_sot=3)
        match_store.store_matches(_matches_data([m]), db_path=db)
        history = match_store.get_team_match_history(1, db_path=db)
        assert len(history) == 1
        assert history[0].get("home_shots_on_target") == 7
        assert history[0].get("away_shots_on_target") == 3

    def test_store_match_without_shots(self):
        """A match without shot data should store NULLs (keys absent in result)."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(101, 1, 2, 2, 0, "HOME_TEAM", "2026-01-15T15:00:00Z")
        match_store.store_matches(_matches_data([m]), db_path=db)
        history = match_store.get_team_match_history(1, db_path=db)
        assert len(history) == 1
        assert "home_shots_on_target" not in history[0]
        assert "away_shots_on_target" not in history[0]


# =============================================================================
# 2.2  update_match_stats backfill
# =============================================================================

class TestUpdateMatchStats:
    """Test backfilling shot stats for existing matches."""

    def test_backfill_sets_shot_data(self):
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(200, 1, 2, 1, 1, "DRAW", "2026-02-01T15:00:00Z")
        match_store.store_matches(_matches_data([m]), db_path=db)
        # Initially no shot data
        h = match_store.get_team_match_history(1, db_path=db)
        assert "home_shots_on_target" not in h[0]
        # Backfill
        match_store.update_match_stats(200, home_sot=5, away_sot=4, db_path=db)
        h = match_store.get_team_match_history(1, db_path=db)
        assert h[0]["home_shots_on_target"] == 5
        assert h[0]["away_shots_on_target"] == 4

    def test_backfill_does_not_overwrite_existing(self):
        """update_match_stats only fills NULLs — existing data is preserved."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(201, 1, 2, 2, 1, "HOME_TEAM", "2026-02-01T15:00:00Z",
                        home_sot=6, away_sot=3)
        match_store.store_matches(_matches_data([m]), db_path=db)
        # Try to overwrite with different values
        match_store.update_match_stats(201, home_sot=99, away_sot=99, db_path=db)
        h = match_store.get_team_match_history(1, db_path=db)
        # Original values preserved (WHERE ... IS NULL guard)
        assert h[0]["home_shots_on_target"] == 6
        assert h[0]["away_shots_on_target"] == 3

    def test_backfill_nonexistent_match_is_noop(self):
        """Backfilling a match_id that doesn't exist does nothing."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        match_store.update_match_stats(999, home_sot=5, away_sot=4, db_path=db)
        assert match_store.match_count(db_path=db) == 0


# =============================================================================
# 2.2  Shot stats extraction from different API formats
# =============================================================================

class TestShotStatsExtraction:
    """store_matches extracts shot data from multiple API response formats."""

    def test_pre_extracted_fields(self):
        """Shot data in top-level home_shots_on_target / away_shots_on_target."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(300, 1, 2, 2, 1, "HOME_TEAM", "2026-03-01T15:00:00Z",
                        home_sot=5, away_sot=2)
        match_store.store_matches(_matches_data([m]), db_path=db)
        h = match_store.get_team_match_history(1, db_path=db)
        assert h[0]["home_shots_on_target"] == 5

    def test_statistics_dict_format(self):
        """Shot data in statistics.home.shotsOnTarget / statistics.away.shotsOnTarget."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(301, 1, 2, 1, 0, "HOME_TEAM", "2026-03-01T15:00:00Z")
        m["statistics"] = {
            "home": {"shotsOnTarget": 8},
            "away": {"shotsOnTarget": 4},
        }
        match_store.store_matches(_matches_data([m]), db_path=db)
        h = match_store.get_team_match_history(1, db_path=db)
        assert h[0]["home_shots_on_target"] == 8
        assert h[0]["away_shots_on_target"] == 4

    def test_team_nested_statistics(self):
        """Shot data in homeTeam.statistics.shotsOnTarget."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        m = _make_match(302, 1, 2, 0, 0, "DRAW", "2026-03-01T15:00:00Z")
        m["homeTeam"]["statistics"] = {"shotsOnTarget": 3}
        m["awayTeam"]["statistics"] = {"shotsOnTarget": 2}
        match_store.store_matches(_matches_data([m]), db_path=db)
        h = match_store.get_team_match_history(1, db_path=db)
        assert h[0]["home_shots_on_target"] == 3
        assert h[0]["away_shots_on_target"] == 2


# =============================================================================
# 2.3  xG proxy in parse_team_history
# =============================================================================

class TestXgProxy:
    """parse_team_history computes xG proxy from shots-on-target."""

    @pytest.fixture
    def team_with_shots(self):
        """6 home matches for team 1, all with shot data."""
        team_id = 1
        matches = []
        for i in range(6):
            matches.append(_make_match(
                400 + i, team_id, 2, 2, 1, "HOME_TEAM",
                f"2026-04-{10+i:02d}T15:00:00Z",
                home_sot=5, away_sot=3,
            ))
        return _matches_data(matches), team_id

    @pytest.fixture
    def team_without_shots(self):
        """6 home matches for team 1, no shot data."""
        team_id = 1
        matches = []
        for i in range(6):
            matches.append(_make_match(
                500 + i, team_id, 2, 2, 1, "HOME_TEAM",
                f"2026-04-{10+i:02d}T15:00:00Z",
            ))
        return _matches_data(matches), team_id

    # -- Happy path --

    def test_xg_scored_home_computed(self, team_with_shots):
        """With 6 home matches with shot data, xg_scored_home should be populated."""
        data, tid = team_with_shots
        result = analyzer.parse_team_history(data, tid)
        assert result["xg_scored_home"] is not None
        # Each match: home_sot=5, xG = 5 * 0.33 = 1.65
        assert result["xg_scored_home"] == pytest.approx(1.65, abs=0.1)

    def test_xg_conceded_home_computed(self, team_with_shots):
        """xg_conceded_home = away_sot * 0.33 for each match."""
        data, tid = team_with_shots
        result = analyzer.parse_team_history(data, tid)
        assert result["xg_conceded_home"] is not None
        # Each match: away_sot=3, xG = 3 * 0.33 = 0.99
        assert result["xg_conceded_home"] == pytest.approx(0.99, abs=0.1)

    # -- Negative: no shot data --

    def test_no_shots_returns_none(self, team_without_shots):
        """Without shot data, all xG fields should be None."""
        data, tid = team_without_shots
        result = analyzer.parse_team_history(data, tid)
        assert result["xg_scored_home"] is None
        assert result["xg_conceded_home"] is None
        assert result["xg_scored_away"] is None
        assert result["xg_conceded_away"] is None

    # -- Edge case: insufficient data --

    def test_single_match_returns_none(self):
        """1 match yields weight ~0.73 (19 days old) — below the 2.0 threshold."""
        team_id = 1
        matches = [_make_match(
            600, team_id, 2, 1, 0, "HOME_TEAM",
            f"2026-04-{10:02d}T15:00:00Z",
            home_sot=4, away_sot=2,
        )]
        result = analyzer.parse_team_history(_matches_data(matches), team_id)
        assert result["xg_scored_home"] is None

    def test_exactly_4_matches_activates_xg(self):
        """Exactly 4 matches with shot data should activate xG proxy.
        Matches must be very recent (<=7 days) so weight sum exceeds the 2.0 threshold."""
        team_id = 1
        matches = []
        for i in range(4):
            matches.append(_make_match(
                700 + i, team_id, 2, 1, 0, "HOME_TEAM",
                _days_ago(i + 1),   # 1–4 days ago — always within the 7-day window
                home_sot=6, away_sot=2,
            ))
        result = analyzer.parse_team_history(_matches_data(matches), team_id)
        assert result["xg_scored_home"] is not None
        # xG = 6 * 0.33 = 1.98
        assert result["xg_scored_home"] == pytest.approx(1.98, abs=0.15)

    # -- Edge case: mixed matches (some with shots, some without) --

    def test_mixed_shot_data(self):
        """Matches with and without shots — only shot-data matches contribute to xG."""
        team_id = 1
        matches = []
        # 5 matches with shots
        for i in range(5):
            matches.append(_make_match(
                800 + i, team_id, 2, 2, 1, "HOME_TEAM",
                f"2026-04-{10+i:02d}T15:00:00Z",
                home_sot=4, away_sot=3,
            ))
        # 3 matches without shots
        for i in range(3):
            matches.append(_make_match(
                810 + i, team_id, 2, 3, 0, "HOME_TEAM",
                f"2026-03-{10+i:02d}T15:00:00Z",
            ))
        result = analyzer.parse_team_history(_matches_data(matches), team_id)
        # xG computed from 5 shot-data matches only
        assert result["xg_scored_home"] is not None
        assert result["xg_scored_home"] == pytest.approx(4 * 0.33, abs=0.15)
        # Goals average uses all 8 matches
        assert result["avg_scored_home"] is not None

    # -- Away matches --

    def test_xg_scored_away(self):
        """xG for away matches uses away team's shots."""
        team_id = 1
        matches = []
        for i in range(5):
            matches.append(_make_match(
                900 + i, 2, team_id, 1, 2, "AWAY_TEAM",
                f"2026-04-{10+i:02d}T15:00:00Z",
                home_sot=3, away_sot=7,   # team 1 is away, has 7 SoT
            ))
        result = analyzer.parse_team_history(_matches_data(matches), team_id)
        assert result["xg_scored_away"] is not None
        # Away team's xG scored = away_sot * 0.33 = 7 * 0.33 = 2.31
        assert result["xg_scored_away"] == pytest.approx(7 * 0.33, abs=0.15)
        # Away team's xG conceded = home_sot * 0.33 = 3 * 0.33 = 0.99
        assert result["xg_conceded_away"] == pytest.approx(3 * 0.33, abs=0.15)


# =============================================================================
# 2.3  default_team_stats includes xG fields
# =============================================================================

class TestDefaultTeamStats:
    """_default_team_stats includes xG fields as None."""

    def test_xg_fields_present(self):
        defaults = analyzer._default_team_stats()
        assert "xg_scored_home" in defaults
        assert "xg_conceded_home" in defaults
        assert "xg_scored_away" in defaults
        assert "xg_conceded_away" in defaults

    def test_xg_fields_are_none(self):
        defaults = analyzer._default_team_stats()
        assert defaults["xg_scored_home"] is None
        assert defaults["xg_conceded_home"] is None


# =============================================================================
# 2.4  xG blend into Poisson lambda
# =============================================================================

class TestXgBlend:
    """compute_match_probabilities blends xG proxy into expected goals."""

    def _standings(self, team_id, avg_scored=1.5, avg_conceded=1.2):
        return {
            "id": team_id, "form_score": 0.6, "position": 5,
            "avg_scored": avg_scored, "avg_conceded": avg_conceded,
            "played": 20, "points": 35, "league": "PL",
        }

    def _history_with_xg(self, xg_scored, xg_conceded, avg_scored=1.5, avg_conceded=1.2):
        h = analyzer._default_team_stats()
        h.update({
            "avg_scored_home": avg_scored,
            "avg_conceded_home": avg_conceded,
            "avg_scored_away": avg_scored,
            "avg_conceded_away": avg_conceded,
            "xg_scored_home": xg_scored,
            "xg_conceded_home": xg_conceded,
            "xg_scored_away": xg_scored,
            "xg_conceded_away": xg_conceded,
            "home_games": 10,
            "away_games": 10,
        })
        return h

    def _history_no_xg(self, avg_scored=1.5, avg_conceded=1.2):
        h = analyzer._default_team_stats()
        h.update({
            "avg_scored_home": avg_scored,
            "avg_conceded_home": avg_conceded,
            "avg_scored_away": avg_scored,
            "avg_conceded_away": avg_conceded,
            "home_games": 10,
            "away_games": 10,
        })
        return h

    def test_xg_changes_expected_goals(self):
        """When xG is available, expected goals should differ from goals-only."""
        home_s = self._standings(1, avg_scored=2.0, avg_conceded=1.0)
        away_s = self._standings(2, avg_scored=1.0, avg_conceded=1.5)

        # Goals-only: avg_scored=2.0
        hist_no_xg = self._history_no_xg(avg_scored=2.0, avg_conceded=1.0)
        result_goals = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_no_xg, hist_no_xg, {"meetings": 0},
        )

        # With xG: xg_scored=1.2 (lower than goals-based 2.0)
        hist_xg = self._history_with_xg(1.2, 1.0, avg_scored=2.0, avg_conceded=1.0)
        result_xg = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_xg, hist_xg, {"meetings": 0},
        )

        # xG is lower than goals -> expected total should be lower with xG blend
        assert result_xg["expected_total"] < result_goals["expected_total"]

    def test_no_xg_uses_goals_only(self):
        """Without xG data, goals averages are used unchanged."""
        home_s = self._standings(1)
        away_s = self._standings(2)
        hist = self._history_no_xg(avg_scored=1.5, avg_conceded=1.2)

        result = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist, hist, {"meetings": 0},
        )
        # Should produce valid results without error
        assert result["expected_total"] > 0
        assert 0 < result["home"] < 1

    def test_xg_blend_ratio(self):
        """Verify the 60/40 xG/goals blend ratio."""
        # If goals avg = 2.0 and xG avg = 1.0, blended should be:
        # 0.60 * 1.0 + 0.40 * 2.0 = 1.40
        home_s = self._standings(1, avg_scored=2.0, avg_conceded=1.0)
        away_s = self._standings(2, avg_scored=2.0, avg_conceded=1.0)

        # Team with goals=2.0 but xG=1.0 (overperforming)
        hist = self._history_with_xg(1.0, 1.0, avg_scored=2.0, avg_conceded=1.0)

        # The internal home_avg_scored should be 0.6*1.0 + 0.4*2.0 = 1.40
        # We can verify indirectly: total xG should be between goals-only and xG-only
        result_xg = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist, hist, {"meetings": 0},
        )
        hist_goals = self._history_no_xg(avg_scored=2.0, avg_conceded=1.0)
        result_goals = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_goals, hist_goals, {"meetings": 0},
        )
        hist_xg_only = self._history_no_xg(avg_scored=1.0, avg_conceded=1.0)
        result_xg_only = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_xg_only, hist_xg_only, {"meetings": 0},
        )

        # Blended should be between the two extremes
        assert result_xg_only["expected_total"] < result_xg["expected_total"] < result_goals["expected_total"]

    def test_xg_improves_over_under_accuracy(self):
        """A team overperforming goals vs xG should have lower Over 2.5 with xG blend."""
        home_s = self._standings(1, avg_scored=2.5, avg_conceded=0.8)
        away_s = self._standings(2, avg_scored=1.0, avg_conceded=1.5)

        # Goals say team scores 2.5, but xG says only 1.5 (lucky finisher)
        hist_xg = self._history_with_xg(1.5, 0.8, avg_scored=2.5, avg_conceded=0.8)
        hist_goals = self._history_no_xg(avg_scored=2.5, avg_conceded=0.8)

        result_xg = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_xg, self._history_no_xg(), {"meetings": 0},
        )
        result_goals = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist_goals, self._history_no_xg(), {"meetings": 0},
        )

        # Over 2.5 should be lower with xG (team is overperforming)
        assert result_xg["over_2_5"] < result_goals["over_2_5"]


# =============================================================================
# 2.5  Round-trip: store → retrieve → parse_team_history
# =============================================================================

class TestXgRoundTrip:
    """Shot data flows through SQLite and back into parse_team_history."""

    def test_full_round_trip(self):
        """Store matches with shots → retrieve → parse → xG fields populated."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        team_id = 1
        matches = []
        for i in range(6):
            matches.append(_make_match(
                1000 + i, team_id, 2, 1, 1, "DRAW",
                f"2026-04-{10+i:02d}T15:00:00Z",
                home_sot=5, away_sot=4,
            ))
        match_store.store_matches(_matches_data(matches), db_path=db)

        # Retrieve from DB
        db_history = match_store.get_team_match_history(team_id, db_path=db)
        assert len(db_history) == 6
        assert all("home_shots_on_target" in m for m in db_history)

        # Parse into team stats
        result = analyzer.parse_team_history({"matches": db_history}, team_id)
        assert result["xg_scored_home"] is not None
        assert result["xg_scored_home"] == pytest.approx(5 * 0.33, abs=0.15)

    def test_round_trip_mixed_old_and_new(self):
        """Old matches without shots + new matches with shots → xG computed from new only."""
        db = _tmp_db()
        match_store.init_db(db_path=db)
        team_id = 1

        # Old matches (no shots)
        old = [_make_match(1100 + i, team_id, 2, 2, 0, "HOME_TEAM",
                           f"2025-12-{10+i:02d}T15:00:00Z")
               for i in range(5)]
        match_store.store_matches(_matches_data(old), db_path=db)

        # New matches (with shots)
        new = [_make_match(1200 + i, team_id, 2, 1, 1, "DRAW",
                           f"2026-04-{10+i:02d}T15:00:00Z",
                           home_sot=6, away_sot=3)
               for i in range(5)]
        match_store.store_matches(_matches_data(new), db_path=db)

        db_history = match_store.get_team_match_history(team_id, db_path=db)
        assert len(db_history) == 10

        result = analyzer.parse_team_history({"matches": db_history}, team_id)
        # xG computed from 5 new matches only (old ones lack shot data)
        assert result["xg_scored_home"] is not None
        # Goals avg uses all 10 matches
        assert result["avg_scored_home"] is not None


# =============================================================================
# Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing code that doesn't provide shot data still works."""

    def test_parse_team_history_no_shot_fields(self):
        """Legacy match data without shot fields works fine."""
        matches = [_make_match(1300 + i, 1, 2, 2, 1, "HOME_TEAM",
                               f"2026-04-{10+i:02d}T15:00:00Z")
                   for i in range(6)]
        result = analyzer.parse_team_history(_matches_data(matches), 1)
        # Goals-based stats still work
        assert result["avg_scored_home"] > 0
        # xG fields are None
        assert result["xg_scored_home"] is None

    def test_compute_match_probs_no_xg(self):
        """compute_match_probabilities works without xG fields in history."""
        home_s = {"id": 1, "form_score": 0.5, "position": 8,
                  "avg_scored": 1.3, "avg_conceded": 1.3,
                  "played": 15, "points": 20, "league": "PL"}
        away_s = {"id": 2, "form_score": 0.5, "position": 12,
                  "avg_scored": 1.1, "avg_conceded": 1.5,
                  "played": 15, "points": 15, "league": "PL"}
        hist = analyzer._default_team_stats()
        hist["home_games"] = 10
        hist["away_games"] = 10

        result = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, hist, hist, {"meetings": 0},
        )
        assert 0 < result["home"] < 1
        assert result["expected_total"] > 0

    def test_merge_old_db_matches_no_shots(self):
        """merge_match_history works with DB matches that have no shot fields."""
        api = [{"id": 1, "utcDate": "2026-04-01T15:00:00Z"}]
        db = [{"id": 2, "utcDate": "2026-03-01T15:00:00Z"}]  # no shot fields
        merged = match_store.merge_match_history(api, db)
        assert len(merged) == 2
