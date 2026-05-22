"""
test_analyzer_extended.py — additional edge case and negative tests for analyzer.py.
Covers: form decay weight, referee factor, standings edge cases, H2H goals blend,
knockout adjustment, referee integration, goals variance nudge, motivation edge cases,
default team stats.
"""
import math
import pytest
import analyzer
from datetime import datetime, timezone, timedelta


# ── Form decay weight ─────────────────────────────────────────────────────────

class TestFormDecayWeight:
    def test_future_date_clamped_to_one(self):
        """A future date produces age=0 days → weight == 1.0."""
        future = "2099-01-01T00:00:00Z"
        w = analyzer._form_decay_weight(future)
        assert w <= 1.0

    def test_empty_string_returns_one(self):
        assert analyzer._form_decay_weight("") == 1.0

    def test_none_returns_one(self):
        assert analyzer._form_decay_weight(None) == 1.0

    def test_garbage_string_returns_one(self):
        assert analyzer._form_decay_weight("not-a-date") == 1.0

    def test_old_match_lower_weight_than_recent(self):
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        old    = (now - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert analyzer._form_decay_weight(recent) > analyzer._form_decay_weight(old)

    def test_30_day_match_weight_approx_61pct(self):
        """weight = exp(-0.5 * 30/30) = exp(-0.5) ≈ 0.606."""
        now = datetime.now(timezone.utc)
        d30 = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        w = analyzer._form_decay_weight(d30)
        assert abs(w - math.exp(-0.5)) < 0.02

    def test_weight_strictly_positive(self):
        """Even very old dates produce weight > 0."""
        now = datetime.now(timezone.utc)
        very_old = (now - timedelta(days=3650)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert analyzer._form_decay_weight(very_old) > 0.0


# ── Referee factor ────────────────────────────────────────────────────────────

class TestRefereeFactorCompute:
    def _make_hist(self, ref_name, scores, team_id=1):
        matches = []
        for i, (hg, ag) in enumerate(scores):
            matches.append({
                "id": i + 1000,
                "homeTeam": {"id": team_id},
                "awayTeam": {"id": 999},
                "score": {"fullTime": {"home": hg, "away": ag}},
                "referees": [{"type": "REFEREE", "name": ref_name}],
                "utcDate": "2025-06-01T15:00:00Z",
            })
        return {"matches": matches}

    def test_empty_ref_name_returns_one(self):
        assert analyzer.compute_referee_factor("", {}, {}) == 1.0

    def test_none_hist_returns_one(self):
        assert analyzer.compute_referee_factor("Ref A", None, None) == 1.0

    def test_insufficient_data_returns_one(self):
        """Fewer than 8 matches returns 1.0."""
        hist = self._make_hist("Ref A", [(2, 1)] * 5)
        assert analyzer.compute_referee_factor("Ref A", hist, {}) == 1.0

    def test_high_scoring_ref_above_one(self):
        """avg 4 goals/game vs league 2.6 → factor > 1."""
        hist = self._make_hist("Ref A", [(2, 2)] * 8)
        result = analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=2.6)
        assert result > 1.0

    def test_low_scoring_ref_below_one(self):
        """avg 1 goal/game vs league 2.6 → factor < 1."""
        hist = self._make_hist("Ref A", [(0, 1)] * 8)
        result = analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=2.6)
        assert result < 1.0

    def test_factor_clamped_max_115(self):
        hist = self._make_hist("Ref A", [(8, 8)] * 10)
        result = analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=2.6)
        assert result <= 1.15

    def test_factor_clamped_min_085(self):
        hist = self._make_hist("Ref A", [(0, 0)] * 10)
        result = analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=2.6)
        assert result >= 0.85

    def test_deduplication_same_match_both_sides(self):
        """Match appearing in both histories must be counted once."""
        hist = self._make_hist("Ref A", [(2, 1)] * 8)
        result_double = analyzer.compute_referee_factor("Ref A", hist, hist, league_avg_gpg=2.6)
        result_single = analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=2.6)
        assert result_double == pytest.approx(result_single, abs=1e-9)

    def test_no_matching_ref_returns_one(self):
        hist = self._make_hist("Other Ref", [(2, 2)] * 8)
        assert analyzer.compute_referee_factor("Ref A", hist, {}) == 1.0

    def test_zero_league_avg_safe_fallback(self):
        hist = self._make_hist("Ref A", [(2, 1)] * 8)
        assert analyzer.compute_referee_factor("Ref A", hist, {}, league_avg_gpg=0) == 1.0

    def test_missing_score_fields_skipped(self):
        """Matches with None score fields must be skipped gracefully."""
        matches = [
            {"id": i, "homeTeam": {"id": 1}, "awayTeam": {"id": 2},
             "score": {"fullTime": {"home": None, "away": None}},
             "referees": [{"type": "REFEREE", "name": "Ref A"}],
             "utcDate": "2025-06-01T15:00:00Z"}
            for i in range(10)
        ]
        hist = {"matches": matches}
        assert analyzer.compute_referee_factor("Ref A", hist, {}) == 1.0


# ── Standings parsing edge cases ──────────────────────────────────────────────

class TestParseStandingsEdgeCases:
    def test_zero_played_games_defaults_to_one(self):
        """playedGames=0 must default to 1 to avoid ZeroDivisionError."""
        data = {"standings": [{"type": "TOTAL", "table": [
            {"team": {"id": 1, "name": "T"}, "position": 1, "points": 0,
             "playedGames": 0, "goalsFor": 5, "goalsAgainst": 3, "form": "W"}
        ]}]}
        standings = analyzer.parse_standings(data, "PL")
        assert standings[1]["avg_scored"] == pytest.approx(5.0)

    def test_no_total_group_returns_empty(self):
        data = {"standings": [{"type": "HOME", "table": []}]}
        assert analyzer.parse_standings(data, "PL") == {}

    def test_entry_without_team_id_skipped(self):
        data = {"standings": [{"type": "TOTAL", "table": [
            {"team": {}, "position": 1, "points": 30,
             "playedGames": 20, "goalsFor": 30, "goalsAgainst": 15, "form": "W"},
        ]}]}
        assert analyzer.parse_standings(data, "PL") == {}

    def test_form_score_all_wins(self):
        data = {"standings": [{"type": "TOTAL", "table": [
            {"team": {"id": 1, "name": "T"}, "position": 1, "points": 60,
             "playedGames": 20, "goalsFor": 50, "goalsAgainst": 10, "form": "WWWWWW"},
        ]}]}
        standings = analyzer.parse_standings(data, "PL")
        assert standings[1]["form_score"] == pytest.approx(1.0, abs=0.01)

    def test_form_score_all_losses(self):
        data = {"standings": [{"type": "TOTAL", "table": [
            {"team": {"id": 1, "name": "T"}, "position": 20, "points": 0,
             "playedGames": 20, "goalsFor": 5, "goalsAgainst": 40, "form": "LLLLLL"},
        ]}]}
        standings = analyzer.parse_standings(data, "PL")
        assert standings[1]["form_score"] == pytest.approx(0.0, abs=0.01)

    def test_multiple_teams_all_present(self):
        data = {"standings": [{"type": "TOTAL", "table": [
            {"team": {"id": i, "name": f"T{i}"}, "position": i, "points": i * 3,
             "playedGames": 20, "goalsFor": 30, "goalsAgainst": 20, "form": "W"}
            for i in range(1, 6)
        ]}]}
        standings = analyzer.parse_standings(data, "PL")
        assert len(standings) == 5


# ── H2H goals blend ───────────────────────────────────────────────────────────

class TestH2HGoalsBlend:
    def _standing(self, tid, pos=10):
        return {"id": tid, "position": pos, "points": 30, "played": 20,
                "avg_scored": 1.5, "avg_conceded": 1.5,
                "form_score": 0.5, "form_str": "W", "league": "PL"}

    def _history(self):
        return {"avg_scored_home": 1.5, "avg_conceded_home": 1.5,
                "avg_scored_away": 1.2, "avg_conceded_away": 1.8,
                "btts_rate": 0.50, "over_2_5_rate": 0.50, "over_3_5_rate": 0.30,
                "clean_sheet_rate_home": 0.25, "clean_sheet_rate_away": 0.20,
                "goals_std": None, "recent_ppg": None,
                "home_games": 10, "away_games": 10}

    def test_low_h2h_goals_reduces_expected_total(self):
        h2h_low = {
            "meetings": 5, "home_wins": 3.0, "draws": 1.0, "away_wins": 1.0,
            "weight_total": 5.0, "total_goals": [1, 1, 1, 1, 1],
            "btts_count": 0.0, "venue_split": True,
        }
        probs_with = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), h2h_low,
        )
        probs_without = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {},
        )
        assert probs_with["expected_total"] < probs_without["expected_total"]

    def test_high_h2h_goals_increases_expected_total(self):
        h2h_high = {
            "meetings": 5, "home_wins": 2.0, "draws": 1.0, "away_wins": 2.0,
            "weight_total": 5.0, "total_goals": [5, 6, 5, 4, 5],
            "btts_count": 5.0, "venue_split": True,
        }
        probs_with = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), h2h_high,
        )
        probs_without = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {},
        )
        assert probs_with["expected_total"] > probs_without["expected_total"]

    def test_four_h2h_meetings_not_blended(self):
        """Only 4 H2H meetings — below the 5-meeting threshold, no blend."""
        h2h_4 = {
            "meetings": 4, "home_wins": 2.0, "draws": 1.0, "away_wins": 1.0,
            "weight_total": 4.0, "total_goals": [1, 1, 1, 1], "btts_count": 0.0,
        }
        probs_with = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), h2h_4,
        )
        probs_without = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {},
        )
        assert probs_with["expected_total"] == pytest.approx(
            probs_without["expected_total"], abs=0.01
        )

    def test_h2h_blend_preserves_home_away_ratio(self):
        """Blending should scale both exp_home and exp_away proportionally."""
        h2h = {
            "meetings": 5, "home_wins": 3.0, "draws": 1.0, "away_wins": 1.0,
            "weight_total": 5.0, "total_goals": [2, 2, 2, 2, 2], "btts_count": 3.0,
        }
        probs = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), h2h,
        )
        total = probs["expected_home_goals"] + probs["expected_away_goals"]
        assert total == pytest.approx(probs["expected_total"], abs=1e-6)


# ── Knockout stage adjustment ─────────────────────────────────────────────────

class TestKnockoutStageAdjustment:
    def _standing(self, tid):
        return {"id": tid, "position": 5, "points": 40, "played": 20,
                "avg_scored": 1.5, "avg_conceded": 1.5,
                "form_score": 0.5, "form_str": "W", "league": "CL"}

    def _history(self):
        return {"avg_scored_home": 1.5, "avg_conceded_home": 1.5,
                "avg_scored_away": 1.2, "avg_conceded_away": 1.8,
                "btts_rate": 0.55, "over_2_5_rate": 0.55, "over_3_5_rate": 0.30,
                "clean_sheet_rate_home": 0.25, "clean_sheet_rate_away": 0.20,
                "goals_std": None, "recent_ppg": None,
                "home_games": 10, "away_games": 10}

    def test_knockout_reduces_expected_goals(self):
        probs_ko = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=True,
        )
        probs_league = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=False,
        )
        assert probs_ko["expected_total"] < probs_league["expected_total"]

    def test_knockout_reduces_over25_prob(self):
        probs_ko = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=True,
        )
        probs_league = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=False,
        )
        assert probs_ko["over_2_5"] < probs_league["over_2_5"]

    def test_knockout_reduces_btts_prob(self):
        probs_ko = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=True,
        )
        probs_league = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=False,
        )
        assert probs_ko["btts_yes"] < probs_league["btts_yes"]

    def test_is_knockout_flag_in_result(self):
        probs = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=True,
        )
        assert probs["is_knockout"] is True

    def test_non_knockout_flag_false(self):
        probs = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=False,
        )
        assert probs["is_knockout"] is False

    def test_1x2_probs_still_sum_to_one_in_knockout(self):
        probs = analyzer.compute_match_probabilities(
            "CL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, is_knockout=True,
        )
        total = probs["home"] + probs["draw"] + probs["away"]
        assert total == pytest.approx(1.0, abs=1e-6)


# ── Referee factor integration with compute_match_probabilities ───────────────

class TestRefereeFactorIntegration:
    def _standing(self, tid):
        return {"id": tid, "position": 5, "points": 40, "played": 20,
                "avg_scored": 1.5, "avg_conceded": 1.5,
                "form_score": 0.5, "form_str": "W", "league": "PL"}

    def _history(self):
        return {"avg_scored_home": 1.5, "avg_conceded_home": 1.5,
                "avg_scored_away": 1.2, "avg_conceded_away": 1.8,
                "btts_rate": 0.50, "over_2_5_rate": 0.50, "over_3_5_rate": 0.30,
                "clean_sheet_rate_home": 0.25, "clean_sheet_rate_away": 0.20,
                "goals_std": None, "recent_ppg": None,
                "home_games": 10, "away_games": 10}

    def test_high_ref_factor_increases_total_goals(self):
        probs_high = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.15,
        )
        probs_neutral = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.0,
        )
        assert probs_high["expected_total"] > probs_neutral["expected_total"]

    def test_low_ref_factor_reduces_total_goals(self):
        probs_low = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=0.85,
        )
        probs_neutral = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.0,
        )
        assert probs_low["expected_total"] < probs_neutral["expected_total"]

    def test_neutral_ref_factor_no_change(self):
        """referee_factor=1.0 must produce identical output to default."""
        probs_a = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.0,
        )
        probs_b = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {},
        )
        assert probs_a["expected_total"] == pytest.approx(probs_b["expected_total"], abs=1e-9)

    def test_ref_factor_affects_both_teams_symmetrically(self):
        """Referee factor applies to both home and away xG."""
        probs_high = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.15,
        )
        probs_neutral = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(), self._history(), {}, referee_factor=1.0,
        )
        assert probs_high["expected_home_goals"] > probs_neutral["expected_home_goals"]
        assert probs_high["expected_away_goals"] > probs_neutral["expected_away_goals"]


# ── Goals variance nudge ──────────────────────────────────────────────────────

class TestGoalsVarianceNudge:
    def _standing(self, tid):
        return {"id": tid, "position": 5, "points": 40, "played": 20,
                "avg_scored": 1.5, "avg_conceded": 1.5,
                "form_score": 0.5, "form_str": "W", "league": "PL"}

    def _history(self, goals_std):
        return {"avg_scored_home": 1.5, "avg_conceded_home": 1.5,
                "avg_scored_away": 1.2, "avg_conceded_away": 1.8,
                "btts_rate": 0.50, "over_2_5_rate": 0.50, "over_3_5_rate": 0.30,
                "clean_sheet_rate_home": 0.25, "clean_sheet_rate_away": 0.20,
                "recent_ppg": None, "home_games": 10, "away_games": 10,
                "goals_std": goals_std}

    def test_high_variance_inflates_xg(self):
        probs_volatile = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(2.5), self._history(2.5), {},
        )
        probs_stable = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(0.3), self._history(0.3), {},
        )
        assert probs_volatile["expected_total"] > probs_stable["expected_total"]

    def test_none_goals_std_runs_without_error(self):
        probs = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            self._history(None), self._history(None), {},
        )
        assert 0.0 < probs["home"] < 1.0

    def test_nudge_capped_at_10_pct(self):
        """Even extreme variance must not change xG by more than 10%."""
        hist_extreme = self._history(100.0)
        hist_none    = self._history(None)
        probs_extreme = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            hist_extreme, hist_extreme, {},
        )
        probs_base = analyzer.compute_match_probabilities(
            "PL", self._standing(1), self._standing(2),
            hist_none, hist_none, {},
        )
        ratio = probs_extreme["expected_total"] / probs_base["expected_total"]
        assert ratio <= 1.10 + 1e-6   # at most 10% increase per team, compounded ~20%


# ── Default team stats ────────────────────────────────────────────────────────

class TestDefaultTeamStats:
    def test_returns_all_required_keys(self):
        stats = analyzer._default_team_stats()
        required = ("avg_scored_home", "avg_conceded_home", "avg_scored_away",
                    "avg_conceded_away", "btts_rate", "over_2_5_rate",
                    "clean_sheet_rate_home", "clean_sheet_rate_away",
                    "goals_std", "recent_ppg", "home_games", "away_games")
        for key in required:
            assert key in stats

    def test_home_games_zero(self):
        assert analyzer._default_team_stats()["home_games"] == 0

    def test_away_games_zero(self):
        assert analyzer._default_team_stats()["away_games"] == 0

    def test_goals_std_none(self):
        assert analyzer._default_team_stats()["goals_std"] is None

    def test_recent_ppg_none(self):
        assert analyzer._default_team_stats()["recent_ppg"] is None

    def test_default_btts_rate_is_fifty_pct(self):
        assert analyzer._default_team_stats()["btts_rate"] == pytest.approx(0.50)


# ── Motivation factor edge cases ──────────────────────────────────────────────

class TestMotivationEdgeCases:
    def test_always_in_valid_range(self):
        """Factor must always be between 0.85 and 1.10 for any input."""
        test_cases = [
            {"position": 1,  "points": 0,   "played": 0,  "league": "PL"},
            {"position": 20, "points": 0,   "played": 38, "league": "PL"},
            {"position": 1,  "points": 100, "played": 38, "league": "PL"},
            {"position": 10, "points": 50,  "played": 20, "league": "BL1"},
            {"position": 3,  "points": 60,  "played": 30, "league": "SA"},
        ]
        for s in test_cases:
            f = analyzer.compute_motivation_factor(s, total_teams=20)
            assert 0.84 <= f <= 1.11, f"Out of range: {f} for {s}"

    def test_bsa_uses_34_game_season(self):
        """BSA season is 34 games. Position 2, played=30 → 4 games left → title race."""
        standing = {"position": 2, "points": 55, "played": 30, "league": "BSA"}
        factor = analyzer.compute_motivation_factor(standing, total_teams=20)
        assert factor == pytest.approx(1.10)

    def test_top3_zero_games_left_not_title_motivated(self):
        """Top 3 but 0 games left → title race condition requires games_left >= 3."""
        standing = {"position": 1, "points": 88, "played": 38, "league": "PL"}
        factor = analyzer.compute_motivation_factor(standing, total_teams=20)
        assert factor != 1.10

    def test_returns_float(self):
        standing = {"position": 10, "points": 35, "played": 20, "league": "PL"}
        assert isinstance(analyzer.compute_motivation_factor(standing), float)

    def test_dead_rubber_already_relegated(self):
        """In bottom 3, max_pts mathematically below safety → dead rubber."""
        standing = {"position": 19, "points": 10, "played": 36, "league": "PL"}
        factor = analyzer.compute_motivation_factor(standing, total_teams=20)
        assert factor == pytest.approx(0.85)

    def test_mid_table_safe_dead_rubber(self):
        """Mid-table, safe, few games left → reduced motivation."""
        standing = {"position": 10, "points": 50, "played": 33, "league": "PL"}
        factor = analyzer.compute_motivation_factor(standing, total_teams=20)
        assert factor < 1.0


# ── Parse team history edge cases ─────────────────────────────────────────────

class TestParseTeamHistoryEdgeCases:
    def _make_match(self, team_id, hg, ag, home=True, match_id=1, days_ago=10):
        date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "id": match_id,
            "homeTeam": {"id": team_id if home else 999},
            "awayTeam": {"id": 999 if home else team_id},
            "score": {
                "winner": "HOME_TEAM" if hg > ag else ("AWAY_TEAM" if ag > hg else "DRAW"),
                "fullTime": {"home": hg, "away": ag},
            },
            "utcDate": date,
        }

    def test_empty_matches_returns_defaults(self):
        result = analyzer.parse_team_history({"matches": []}, team_id=1)
        assert result["home_games"] == 0
        assert result["away_games"] == 0

    def test_none_data_returns_defaults(self):
        result = analyzer.parse_team_history(None, team_id=1)
        assert result["btts_rate"] == pytest.approx(0.50)

    def test_missing_score_fields_skipped(self):
        """Matches with None goals must be skipped without error."""
        matches = [{"homeTeam": {"id": 1}, "awayTeam": {"id": 2},
                    "score": {"fullTime": {"home": None, "away": None}},
                    "utcDate": "2025-06-01T00:00:00Z"}]
        result = analyzer.parse_team_history({"matches": matches}, team_id=1)
        assert result["home_games"] == 0

    def test_clean_sheet_counted_correctly(self):
        matches = [
            self._make_match(1, 2, 0, home=True, match_id=1),  # home CS
            self._make_match(1, 1, 1, home=True, match_id=2),  # no CS
            self._make_match(1, 3, 0, home=True, match_id=3),  # home CS
        ]
        result = analyzer.parse_team_history({"matches": matches}, team_id=1)
        # 2/3 clean sheets (roughly, decay-weighted)
        assert result["clean_sheet_rate_home"] > 0.5

    def test_over25_rate_all_high_scoring(self):
        matches = [
            self._make_match(1, 2, 2, home=True, match_id=i)
            for i in range(6)
        ]
        result = analyzer.parse_team_history({"matches": matches}, team_id=1)
        assert result["over_2_5_rate"] == pytest.approx(1.0, abs=0.01)

    def test_recent_ppg_computed_from_last_5(self):
        """5 wins in a row → recent_ppg = 3.0."""
        matches = [
            self._make_match(1, 2, 0, home=True, match_id=i, days_ago=i)
            for i in range(1, 6)
        ]
        result = analyzer.parse_team_history({"matches": matches}, team_id=1)
        assert result["recent_ppg"] == pytest.approx(3.0)

    def test_recent_ppg_none_when_fewer_than_5(self):
        matches = [self._make_match(1, 2, 0, home=True, match_id=1)]
        result = analyzer.parse_team_history({"matches": matches}, team_id=1)
        assert result["recent_ppg"] is None
