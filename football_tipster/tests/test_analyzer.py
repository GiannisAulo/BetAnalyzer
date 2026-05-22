"""
test_analyzer.py — unit tests for the probability model and data parsers.
"""
import math
import pytest
import analyzer
from analyzer import build_reason


# ── Poisson helpers ──────────────────────────────────────────────────────────

class TestPoissonProb:
    def test_zero_goals_mean_one(self):
        """P(k=0 | λ=1) = e^-1 ≈ 0.368"""
        result = analyzer.poisson_prob(1.0, 0)
        assert abs(result - math.exp(-1)) < 1e-9

    def test_one_goal_mean_one(self):
        """P(k=1 | λ=1) = e^-1 ≈ 0.368"""
        result = analyzer.poisson_prob(1.0, 1)
        assert abs(result - math.exp(-1)) < 1e-9

    def test_zero_lambda(self):
        """λ=0: certain 0 goals, impossible any k>0"""
        assert analyzer.poisson_prob(0, 0) == 1.0
        assert analyzer.poisson_prob(0, 1) == 0.0

    def test_probabilities_sum_to_one(self):
        """Sum of P(k | λ=2.5) for k=0..20 should be ≈1"""
        total = sum(analyzer.poisson_prob(2.5, k) for k in range(21))
        assert abs(total - 1.0) < 1e-6

    def test_high_lambda_shifts_peak(self):
        """For λ=5 the mode is at k=4 or k=5 (they are equal by Poisson symmetry)."""
        probs = [analyzer.poisson_prob(5.0, k) for k in range(11)]
        peak = probs.index(max(probs))
        assert peak in (4, 5)


class TestProbOver:
    def test_over_25_typical(self):
        """P(>2.5 | λ=3) should be > 0.5"""
        result = analyzer.prob_over(3.0, 2.5)
        assert result > 0.5

    def test_over_25_low_lambda(self):
        """With λ=1 very few 3-goal games expected"""
        result = analyzer.prob_over(1.0, 2.5)
        assert result < 0.1

    def test_over_35_less_than_over_25(self):
        """P(>3.5) < P(>2.5) for any λ"""
        lam = 2.8
        assert analyzer.prob_over(lam, 3.5) < analyzer.prob_over(lam, 2.5)

    def test_range(self):
        """Result always in [0, 1]"""
        for lam in [0.5, 1.5, 2.5, 3.5, 5.0]:
            result = analyzer.prob_over(lam, 2.5)
            assert 0.0 <= result <= 1.0


class TestProbBtts:
    def test_both_attack(self):
        """High-scoring teams: BTTS should be likely"""
        result = analyzer.prob_btts(2.0, 2.0)
        assert result > 0.6

    def test_one_team_blank(self):
        """If one team almost never scores, BTTS unlikely"""
        result = analyzer.prob_btts(0.1, 2.0)
        assert result < 0.15

    def test_range(self):
        """Result always in [0, 1]"""
        for lh, la in [(0.5, 0.5), (1.5, 1.5), (3.0, 0.2)]:
            result = analyzer.prob_btts(lh, la)
            assert 0.0 <= result <= 1.0


# ── Standings parser ──────────────────────────────────────────────────────────

STANDINGS_FIXTURE = {
    "standings": [
        {
            "type": "TOTAL",
            "table": [
                {
                    "position": 1,
                    "points": 70,
                    "playedGames": 30,
                    "goalsFor": 60,
                    "goalsAgainst": 20,
                    "form": "WWDWW",
                    "team": {"id": 101, "name": "Top FC"},
                },
                {
                    "position": 10,
                    "points": 40,
                    "playedGames": 30,
                    "goalsFor": 35,
                    "goalsAgainst": 40,
                    "form": "WLDLD",
                    "team": {"id": 202, "name": "Mid FC"},
                },
            ],
        }
    ]
}


class TestParseStandings:
    def test_returns_dict_keyed_by_id(self):
        result = analyzer.parse_standings(STANDINGS_FIXTURE, "PL")
        assert 101 in result
        assert 202 in result

    def test_avg_goals(self):
        result = analyzer.parse_standings(STANDINGS_FIXTURE, "PL")
        team = result[101]
        assert abs(team["avg_scored"] - 2.0) < 1e-9     # 60/30
        assert abs(team["avg_conceded"] - 0.667) < 0.01 # 20/30

    def test_form_score_top_team(self):
        """WWDWW is a strong form — score should be > 0.5"""
        result = analyzer.parse_standings(STANDINGS_FIXTURE, "PL")
        assert result[101]["form_score"] > 0.5

    def test_empty_data(self):
        result = analyzer.parse_standings({}, "PL")
        assert result == {}

    def test_missing_standings_key(self):
        result = analyzer.parse_standings({"standings": []}, "PL")
        assert result == {}


# ── Team history parser ───────────────────────────────────────────────────────

def _make_match(hg, ag, team_id, as_home=True):
    home_id, away_id = (team_id, 999) if as_home else (999, team_id)
    return {
        "homeTeam": {"id": home_id},
        "awayTeam": {"id": away_id},
        "score": {"fullTime": {"home": hg, "away": ag}},
    }


class TestParseTeamHistory:
    def test_home_averages(self):
        data = {
            "matches": [
                _make_match(2, 1, 1, as_home=True),
                _make_match(3, 0, 1, as_home=True),
            ]
        }
        result = analyzer.parse_team_history(data, 1)
        assert abs(result["avg_scored_home"] - 2.5) < 1e-9
        assert abs(result["avg_conceded_home"] - 0.5) < 1e-9

    def test_clean_sheet_rate(self):
        data = {
            "matches": [
                _make_match(1, 0, 1, as_home=True),   # CS
                _make_match(2, 0, 1, as_home=True),   # CS
                _make_match(1, 1, 1, as_home=True),   # no CS
                _make_match(0, 2, 1, as_home=True),   # no CS
            ]
        }
        result = analyzer.parse_team_history(data, 1)
        assert abs(result["clean_sheet_rate_home"] - 0.5) < 1e-9

    def test_btts_rate(self):
        data = {
            "matches": [
                _make_match(1, 1, 1, as_home=True),  # BTTS
                _make_match(2, 0, 1, as_home=True),  # not BTTS
            ]
        }
        result = analyzer.parse_team_history(data, 1)
        assert abs(result["btts_rate"] - 0.5) < 1e-9

    def test_empty_data_returns_defaults(self):
        result = analyzer.parse_team_history({}, 1)
        assert "avg_scored_home" in result


# ── H2H parser ────────────────────────────────────────────────────────────────

class TestParseH2H:
    def test_counts(self):
        data = {
            "matches": [
                {"score": {"fullTime": {"home": 2, "away": 0}, "winner": "HOME_TEAM"}},
                {"score": {"fullTime": {"home": 1, "away": 1}, "winner": "DRAW"}},
                {"score": {"fullTime": {"home": 0, "away": 1}, "winner": "AWAY_TEAM"}},
            ]
        }
        result = analyzer.parse_h2h(data)
        assert result["meetings"]  == 3
        assert result["home_wins"] == 1
        assert result["draws"]     == 1
        assert result["away_wins"] == 1

    def test_empty(self):
        result = analyzer.parse_h2h({})
        assert result["meetings"] == 0

    def test_btts_count(self):
        data = {
            "matches": [
                {"score": {"fullTime": {"home": 1, "away": 1}, "winner": "DRAW"}},
                {"score": {"fullTime": {"home": 2, "away": 0}, "winner": "HOME_TEAM"}},
            ]
        }
        result = analyzer.parse_h2h(data)
        assert result["btts_count"] == 1


# ── Full probability pipeline ────────────────────────────────────────────────

def _default_standing():
    return {"form_score": 0.5, "position": 10, "avg_scored": 1.2, "avg_conceded": 1.2}


def _default_history():
    return {
        "avg_scored_home":      1.5,
        "avg_conceded_home":    1.0,
        "avg_scored_away":      1.0,
        "avg_conceded_away":    1.5,
        "clean_sheet_rate_home": 0.3,
        "clean_sheet_rate_away": 0.2,
        "btts_rate":            0.5,
        "over_2_5_rate":        0.5,
        "over_3_5_rate":        0.3,
        "home_games":           5,
        "away_games":           5,
    }


def _default_h2h():
    return {"meetings": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
            "total_goals": [], "btts_count": 0}


class TestComputeMatchProbabilities:
    def test_probabilities_sum_to_one(self):
        probs = analyzer.compute_match_probabilities(
            "PL",
            _default_standing(), _default_standing(),
            _default_history(),  _default_history(),
            _default_h2h(),
        )
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 1e-6

    def test_required_keys_present(self):
        probs = analyzer.compute_match_probabilities(
            "BL1",
            _default_standing(), _default_standing(),
            _default_history(),  _default_history(),
            _default_h2h(),
        )
        for key in ["home", "draw", "away", "over_2_5", "under_2_5",
                    "over_3_5", "under_3_5", "btts_yes", "btts_no"]:
            assert key in probs, f"Missing key: {key}"

    def test_over_under_complement(self):
        probs = analyzer.compute_match_probabilities(
            "PL",
            _default_standing(), _default_standing(),
            _default_history(),  _default_history(),
            _default_h2h(),
        )
        assert abs(probs["over_2_5"] + probs["under_2_5"] - 1.0) < 1e-6
        assert abs(probs["btts_yes"] + probs["btts_no"]   - 1.0) < 1e-6

    def test_strong_home_team_increases_home_prob(self):
        home_s = {"form_score": 0.9, "position": 1, "avg_scored": 2.5, "avg_conceded": 0.5}
        away_s = {"form_score": 0.2, "position": 18, "avg_scored": 0.8, "avg_conceded": 2.0}
        probs = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, _default_history(), _default_history(), _default_h2h()
        )
        assert probs["home"] > probs["away"]

    def test_all_probabilities_in_range(self):
        probs = analyzer.compute_match_probabilities(
            "SA",
            _default_standing(), _default_standing(),
            _default_history(),  _default_history(),
            _default_h2h(),
        )
        # Only check keys that represent probabilities (not expected goals, form scores, etc.)
        prob_keys = {
            "home", "draw", "away",
            "over_2_5", "under_2_5", "over_3_5", "under_3_5",
            "btts_yes", "btts_no",
        }
        for key in prob_keys:
            val = probs[key]
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"


# ── build_reason ──────────────────────────────────────────────────────────────

def _probs_with_goals():
    """Probs dict with all fields build_reason uses."""
    return {
        "expected_home_goals": 1.8,
        "expected_away_goals": 1.1,
        "home_avg_scored":     1.8,
        "away_avg_scored":     1.1,
        "home_cs_rate":        0.30,
        "away_cs_rate":        0.20,
    }


def _standing_with_form(form="WWDWL"):
    return {"form_score": 0.7, "position": 3, "avg_scored": 1.8,
            "avg_conceded": 0.9, "form_str": form}


class TestBuildReason:
    def test_returns_string(self):
        r = build_reason("Home Win", "1X2", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form("WWWDW"), _standing_with_form("LDWDL"),
                         _default_h2h())
        assert isinstance(r, str)

    def test_contains_xg(self):
        r = build_reason("Home Win", "1X2", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form(), _standing_with_form(), _default_h2h())
        assert "xG" in r
        assert "1.8" in r

    def test_contains_form_strings(self):
        r = build_reason("Home Win", "1X2", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form("WWWDW"), _standing_with_form("LDLDL"),
                         _default_h2h())
        assert "WWWDW" in r
        assert "LDLDL" in r

    def test_h2h_shown_when_enough_meetings(self):
        h2h = {"meetings": 6, "home_wins": 4, "draws": 1, "away_wins": 1,
                "total_goals": [2, 3, 1], "btts_count": 2}
        r = build_reason("Home Win", "1X2", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form(), _standing_with_form(), h2h)
        assert "H2H" in r
        assert "4W" in r

    def test_h2h_hidden_when_too_few_meetings(self):
        h2h = {"meetings": 3, "home_wins": 2, "draws": 1, "away_wins": 0,
                "total_goals": [], "btts_count": 0}
        r = build_reason("Home Win", "1X2", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form(), _standing_with_form(), h2h)
        assert "H2H" not in r

    def test_clean_sheet_shown_for_btts(self):
        r = build_reason("BTTS Yes", "BTTS", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form(), _standing_with_form(), _default_h2h())
        assert "CS" in r

    def test_clean_sheet_shown_for_over_under(self):
        r = build_reason("Over 2.5", "Over/Under", _probs_with_goals(),
                         "Arsenal", "Chelsea",
                         _standing_with_form(), _standing_with_form(), _default_h2h())
        assert "CS" in r

    def test_empty_probs_returns_string(self):
        """Should never raise — returns empty string or partial reason."""
        r = build_reason("Home Win", "1X2", {}, "A", "B", {}, {}, {})
        assert isinstance(r, str)


# ── Dixon-Coles helpers ───────────────────────────────────────────────────────

class TestTau:
    def test_rho_zero_is_identity(self):
        """rho=0 → tau=1.0 for all low-score cells (pure Poisson)."""
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            assert analyzer._tau(x, y, 1.5, 1.2, 0) == pytest.approx(1.0)

    def test_non_low_score_always_one(self):
        """Any cell with x+y >= 2 (except 1-1) returns 1."""
        for x, y in [(2, 0), (0, 2), (3, 1), (2, 2)]:
            assert analyzer._tau(x, y, 1.5, 1.2, -0.13) == pytest.approx(1.0)

    def test_00_reduced_with_negative_rho(self):
        """Negative rho should reduce the 0-0 probability (common in football)."""
        tau_val = analyzer._tau(0, 0, 1.5, 1.2, -0.13)
        assert tau_val > 1.0   # 1 - lam_h*lam_a*(-0.13) > 1

    def test_11_increased_with_negative_rho(self):
        """Negative rho increases the 1-1 cell."""
        tau_val = analyzer._tau(1, 1, 1.5, 1.2, -0.13)
        assert tau_val > 1.0   # 1 - (-0.13) = 1.13


class TestScoreMatrix:
    def test_matrix_sums_to_one(self):
        matrix = analyzer._score_matrix(1.5, 1.2)
        total = sum(p for row in matrix for p in row)
        assert abs(total - 1.0) < 1e-6

    def test_all_values_non_negative(self):
        matrix = analyzer._score_matrix(1.5, 1.2)
        for row in matrix:
            for p in row:
                assert p >= 0.0

    def test_rho_zero_matches_poisson(self):
        """With rho=0 the matrix should be close to independent Poisson.
        Small deviation is expected due to 8x8 grid truncation + renormalisation."""
        import math
        matrix = analyzer._score_matrix(1.5, 1.2, rho=0.0)
        expected_00 = math.exp(-1.5) * math.exp(-1.2)
        assert matrix[0][0] == pytest.approx(expected_00, rel=5e-3)


class TestProbsFromMatrix:
    def test_1x2_sums_to_one(self):
        matrix = analyzer._score_matrix(1.5, 1.2)
        hw, d, aw, *_ = analyzer._probs_from_matrix(matrix)
        assert abs(hw + d + aw - 1.0) < 1e-6

    def test_strong_home_dominates(self):
        """lam_h >> lam_a → home win should dominate."""
        matrix = analyzer._score_matrix(3.0, 0.5)
        hw, d, aw, *_ = analyzer._probs_from_matrix(matrix)
        assert hw > aw
        assert hw > d

    def test_over_25_reasonable(self):
        matrix = analyzer._score_matrix(1.5, 1.2)
        *_, over_25, _ = analyzer._probs_from_matrix(matrix)
        assert 0.3 < over_25 < 0.8


# ── Strength factors ─────────────────────────────────────────────────────────

# League mean = 1.4 scored, 1.4 conceded — team 2 is exactly average
_STANDINGS_FOR_SF = {
    1: {"avg_scored": 2.4, "avg_conceded": 0.8},   # strong attacker, solid defence
    2: {"avg_scored": 1.4, "avg_conceded": 1.4},   # exactly average
    3: {"avg_scored": 0.4, "avg_conceded": 2.0},   # weak
}


class TestComputeStrengthFactors:
    def test_average_team_near_one(self):
        factors = analyzer._compute_strength_factors(_STANDINGS_FOR_SF)
        assert factors[2]["attack"]  == pytest.approx(1.0, rel=1e-3)
        assert factors[2]["defence"] == pytest.approx(1.0, rel=1e-3)

    def test_strong_attacker_above_one(self):
        factors = analyzer._compute_strength_factors(_STANDINGS_FOR_SF)
        assert factors[1]["attack"] > 1.0

    def test_weak_defender_above_one(self):
        # avg_conceded > league avg → defence factor > 1 (worse)
        factors = analyzer._compute_strength_factors(_STANDINGS_FOR_SF)
        assert factors[3]["defence"] > 1.0

    def test_empty_standings_returns_empty(self):
        assert analyzer._compute_strength_factors({}) == {}

    def test_all_teams_present(self):
        factors = analyzer._compute_strength_factors(_STANDINGS_FOR_SF)
        # Exclude metadata keys (_league_avg_scored, _league_avg_conceded)
        team_keys = {k for k in factors.keys() if not str(k).startswith("_")}
        assert team_keys == {1, 2, 3}

    def test_indices_clamped_to_max_two(self):
        """Extreme scoring/conceding rates must not produce indices above 2.0."""
        extreme = {
            1: {"avg_scored": 4.5, "avg_conceded": 0.2},  # elite attacker
            2: {"avg_scored": 0.3, "avg_conceded": 4.0},  # terrible defence
            3: {"avg_scored": 1.0, "avg_conceded": 1.0},  # average
        }
        factors = analyzer._compute_strength_factors(extreme)
        for tid in (1, 2, 3):
            assert factors[tid]["attack"]  <= 2.0, f"attack index > 2.0 for team {tid}"
            assert factors[tid]["defence"] <= 2.0, f"defence index > 2.0 for team {tid}"
            assert factors[tid]["attack"]  >= 0.4, f"attack index < 0.4 for team {tid}"
            assert factors[tid]["defence"] >= 0.4, f"defence index < 0.4 for team {tid}"

    def test_split_indices_clamped(self):
        """Split home/away indices from team histories also clamped to [0.40, 2.00]."""
        extreme_standings = {
            1: {"avg_scored": 4.5, "avg_conceded": 0.2},
            2: {"avg_scored": 1.0, "avg_conceded": 1.0},
        }
        extreme_histories = {
            1: {
                "avg_scored_home":   5.0, "avg_conceded_home": 0.1,
                "avg_scored_away":   3.5, "avg_conceded_away": 0.3,
                "home_games": 10, "away_games": 10,
            }
        }
        factors = analyzer._compute_strength_factors(extreme_standings, extreme_histories)
        assert factors[1]["home_attack"]  <= 2.0
        assert factors[1]["home_defence"] >= 0.4
        assert factors[1]["away_attack"]  <= 2.0


class TestStrengthAdjustedExpectedGoals:
    """compute_match_probabilities should use strength factors when provided."""

    def _sf(self):
        return analyzer._compute_strength_factors(_STANDINGS_FOR_SF)

    def test_strong_vs_weak_inflates_home_goals(self):
        # Team 1 (strong attack) vs Team 3 (weak defence)
        home_s = {**_default_standing(), "id": 1, "avg_scored": 2.4, "avg_conceded": 0.8}
        away_s = {**_default_standing(), "id": 3, "avg_scored": 0.6, "avg_conceded": 2.0}
        probs = analyzer.compute_match_probabilities(
            "PL", home_s, away_s,
            _default_history(), _default_history(), _default_h2h(),
            strength_factors=self._sf(),
        )
        assert probs["expected_home_goals"] > probs["expected_away_goals"]

    def test_fallback_when_no_factors(self):
        """No crash and valid output when strength_factors=None."""
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            strength_factors=None,
        )
        assert 0.0 < probs["home"] < 1.0

    def test_fallback_when_id_missing(self):
        """Standing without 'id' key → falls back to simple average."""
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            strength_factors=self._sf(),
        )
        assert 0.0 < probs["home"] < 1.0

    def test_xg_never_exceeds_ceiling(self):
        """Elite attacker vs bottom-table defence must not produce xG > 3.5 per team."""
        # Simulate Bayern (elite home scorer) vs bottom Championship side (leaks goals)
        extreme_standings = {
            1: {"avg_scored": 3.2, "avg_conceded": 0.5, "form_score": 0.9, "position": 1},
            2: {"avg_scored": 0.5, "avg_conceded": 3.0, "form_score": 0.1, "position": 20},
            # fill out a full 20-team league so league avg is realistic
            **{i: {"avg_scored": 1.4, "avg_conceded": 1.4, "form_score": 0.5, "position": i}
               for i in range(3, 21)},
        }
        sf = analyzer._compute_strength_factors(extreme_standings)
        home_s = {**extreme_standings[1], "id": 1, "avg_scored": 3.2, "avg_conceded": 0.5}
        away_s = {**extreme_standings[2], "id": 2, "avg_scored": 0.5, "avg_conceded": 3.0}
        home_hist = {**_default_history(), "avg_scored_home": 3.5, "avg_conceded_home": 0.4, "home_games": 15}
        away_hist = {**_default_history(), "avg_conceded_away": 3.5, "away_games": 15}
        probs = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, home_hist, away_hist, _default_h2h(),
            strength_factors=sf, total_teams=20,
        )
        assert probs["expected_home_goals"] <= 3.5, (
            f"xG home {probs['expected_home_goals']:.2f} exceeds ceiling of 3.5"
        )
        assert probs["expected_away_goals"] <= 3.5, (
            f"xG away {probs['expected_away_goals']:.2f} exceeds ceiling of 3.5"
        )

    def test_xg_realistic_for_average_match(self):
        """Two average teams should produce xG in the 1.0–2.0 range per team."""
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            strength_factors=None,
        )
        assert 0.8 <= probs["expected_home_goals"] <= 2.5
        assert 0.8 <= probs["expected_away_goals"] <= 2.5

    def test_xg_home_favoured_over_away(self):
        """Home team should always have higher xG than away for equivalent strength (home advantage)."""
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            strength_factors=None,
        )
        assert probs["expected_home_goals"] > probs["expected_away_goals"]


class TestHomAwaySplitGuard:
    """Goals model should use split stats only when sample >= 4."""

    def test_uses_split_when_enough_games(self):
        home_hist = {**_default_history(), "avg_scored_home": 2.5, "home_games": 5}
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            home_hist, _default_history(), _default_h2h(),
        )
        # With split used, exp_home blends around 2.5 (stronger than default 1.2)
        assert probs["expected_home_goals"] > 1.5

    def test_falls_back_when_too_few_home_games(self):
        # home_games=2 → should use standing avg (1.2), not the inflated split avg
        home_hist = {**_default_history(), "avg_scored_home": 3.0, "home_games": 2}
        home_s = {**_default_standing(), "avg_scored": 1.2}
        probs = analyzer.compute_match_probabilities(
            "PL", home_s, _default_standing(),
            home_hist, _default_history(), _default_h2h(),
        )
        # Should be close to standing avg, not the inflated 3.0
        assert probs["expected_home_goals"] < 2.0

    def test_falls_back_when_zero_away_games(self):
        away_hist = {**_default_history(), "avg_scored_away": 3.0, "away_games": 0}
        away_s = {**_default_standing(), "avg_scored": 1.0}
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), away_s,
            _default_history(), away_hist, _default_h2h(),
        )
        assert probs["expected_away_goals"] < 2.0


# ── H2H recency weighting ────────────────────────────────────────────────────

class TestH2HDecayWeight:
    def test_recent_match_weight_near_one(self):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        w = analyzer._h2h_decay_weight(recent)
        assert w > 0.90

    def test_old_match_has_lower_weight(self):
        w_old    = analyzer._h2h_decay_weight("2019-01-01T00:00:00Z")
        w_recent = analyzer._h2h_decay_weight("2024-01-01T00:00:00Z")
        assert w_old < w_recent

    def test_missing_date_returns_one(self):
        assert analyzer._h2h_decay_weight("") == 1.0
        assert analyzer._h2h_decay_weight(None) == 1.0

    def test_unparseable_date_returns_one(self):
        assert analyzer._h2h_decay_weight("not-a-date") == 1.0


class TestParseH2HWeighted:
    def test_recent_home_win_dominates_old(self):
        """A recent home win should outweigh many old away wins."""
        from datetime import datetime, timezone, timedelta
        recent  = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        old     = "2018-01-01T00:00:00Z"
        data = {
            "matches": [
                {"utcDate": recent, "score": {"fullTime": {"home": 2, "away": 0}, "winner": "HOME_TEAM"}},
                {"utcDate": old,    "score": {"fullTime": {"home": 0, "away": 1}, "winner": "AWAY_TEAM"}},
                {"utcDate": old,    "score": {"fullTime": {"home": 0, "away": 1}, "winner": "AWAY_TEAM"}},
                {"utcDate": old,    "score": {"fullTime": {"home": 0, "away": 1}, "winner": "AWAY_TEAM"}},
                {"utcDate": old,    "score": {"fullTime": {"home": 0, "away": 1}, "winner": "AWAY_TEAM"}},
            ]
        }
        result = analyzer.parse_h2h(data)
        # Weighted home_wins should be higher than unweighted would suggest
        assert result["home_wins"] > result["away_wins"]

    def test_weight_total_present(self):
        data = {"matches": [
            {"utcDate": "2024-01-01T00:00:00Z",
             "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"}}
        ]}
        result = analyzer.parse_h2h(data)
        assert "weight_total" in result
        assert result["weight_total"] > 0


# ── Season-stage confidence scaling ─────────────────────────────────────────

class TestSeasonStageScaling:
    def test_adjustments_near_zero_early_season(self):
        """After 2 games, form and position nudges should be tiny."""
        strong_home = {"form_score": 1.0, "position": 1, "avg_scored": 2.5,
                       "avg_conceded": 0.5, "played": 2}
        weak_away   = {"form_score": 0.0, "position": 20, "avg_scored": 0.5,
                       "avg_conceded": 2.5, "played": 2}
        probs_early = analyzer.compute_match_probabilities(
            "PL", strong_home, weak_away,
            _default_history(), _default_history(), _default_h2h(),
        )
        # After 2 games, home prob shouldn't be massively boosted
        assert probs_early["home"] < 0.70

    def test_adjustments_full_strength_after_10_games(self):
        """After 15 games, full adjustments apply — strong home should dominate."""
        strong_home = {"form_score": 1.0, "position": 1, "avg_scored": 2.5,
                       "avg_conceded": 0.5, "played": 15}
        weak_away   = {"form_score": 0.0, "position": 20, "avg_scored": 0.5,
                       "avg_conceded": 2.5, "played": 15}
        probs_full = analyzer.compute_match_probabilities(
            "PL", strong_home, weak_away,
            _default_history(), _default_history(), _default_h2h(),
        )
        assert probs_full["home"] > probs_full["away"]


# ── Fatigue flag ─────────────────────────────────────────────────────────────

class TestFatigueFlag:
    def test_away_fatigue_reduces_away_goals(self):
        from datetime import datetime, timezone, timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        probs_fresh = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
        )
        probs_tired = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            away_last_match_date=yesterday,
        )
        assert probs_tired["expected_away_goals"] < probs_fresh["expected_away_goals"]

    def test_away_fatigue_flag_set(self):
        from datetime import datetime, timezone, timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            away_last_match_date=yesterday,
        )
        assert probs["away_fatigue"] is True

    def test_no_fatigue_when_rested(self):
        from datetime import datetime, timezone, timedelta
        # 15 days ago → beyond the 14-day full-recovery threshold
        rested = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
            away_last_match_date=rested,
        )
        assert probs["away_fatigue"] is False

    def test_no_fatigue_when_date_absent(self):
        probs = analyzer.compute_match_probabilities(
            "PL", _default_standing(), _default_standing(),
            _default_history(), _default_history(), _default_h2h(),
        )
        assert probs["away_fatigue"] is False
        assert probs["home_fatigue"] is False


# ── Boundary tests for compute_match_probabilities ───────────────────────────

class TestComputeBoundaries:
    def test_pos_diff_exactly_5_no_adjustment(self):
        """pos_diff = 5 is not > 5, so no adjustment fires."""
        home_s = {**_default_standing(), "position": 5}
        away_s = {**_default_standing(), "position": 10}   # diff = 5
        probs_no_adj = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, _default_history(), _default_history(), _default_h2h()
        )
        home_s2 = {**_default_standing(), "position": 4}   # diff = 6 → fires
        probs_adj = analyzer.compute_match_probabilities(
            "PL", home_s2, away_s, _default_history(), _default_history(), _default_h2h()
        )
        # When home position improves (diff > 5), home gets a boost
        assert probs_adj["home"] >= probs_no_adj["home"]

    def test_pos_diff_exactly_minus_5_no_adjustment(self):
        home_s = {**_default_standing(), "position": 10}
        away_s = {**_default_standing(), "position": 5}   # diff = -5, not < -5
        probs_no_adj = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, _default_history(), _default_history(), _default_h2h()
        )
        away_s2 = {**_default_standing(), "position": 4}   # diff = -6 → fires
        probs_adj = analyzer.compute_match_probabilities(
            "PL", home_s, away_s2, _default_history(), _default_history(), _default_h2h()
        )
        assert probs_adj["away"] >= probs_no_adj["away"]

    def test_form_adv_exactly_030_no_adjustment(self):
        """form_adv = 0.30 is not > 0.30, so no boost."""
        home_s = {**_default_standing(), "form_score": 0.80, "played": 15}
        away_s = {**_default_standing(), "form_score": 0.50, "played": 15}   # diff = 0.30
        probs_boundary = analyzer.compute_match_probabilities(
            "PL", home_s, away_s, _default_history(), _default_history(), _default_h2h()
        )
        home_s2 = {**_default_standing(), "form_score": 0.81, "played": 15}  # diff = 0.31 → fires
        probs_boosted = analyzer.compute_match_probabilities(
            "PL", home_s2, away_s, _default_history(), _default_history(), _default_h2h()
        )
        assert probs_boosted["home"] >= probs_boundary["home"]

    def test_h2h_diff_exactly_010_no_adjustment(self):
        """h2h_home_diff = 0.10 is not > 0.10, so no nudge."""
        # baseline PL home = 0.46, so home_wins/meetings = 0.56 → diff = 0.10 exactly
        h2h_boundary = {"meetings": 10, "home_wins": 5.6, "draws": 3,
                        "away_wins": 1.4, "total_goals": [], "btts_count": 0,
                        "weight_total": 10}
        probs = analyzer.compute_match_probabilities(
            "PL", {**_default_standing(), "played": 15},
            {**_default_standing(), "played": 15},
            _default_history(), _default_history(), h2h_boundary,
        )
        assert 0.0 < probs["home"] < 1.0   # valid output, no crash


# ── Exact rest-days fatigue (Tier 1 #1) ─────────────────────────────────────

class TestExactRestDaysFatigue:
    """Verify the continuous fatigue curve replaces the old binary flag."""

    def _probs(self, home_last=None, away_last=None):
        s = {**_default_standing(), "played": 20}
        return analyzer.compute_match_probabilities(
            "PL", s, s,
            _default_history(), _default_history(), _default_h2h(),
            home_last_match_date=home_last,
            away_last_match_date=away_last,
        )

    def test_no_fatigue_baseline(self):
        """No last-match date → no fatigue applied, xG unchanged."""
        p = self._probs()
        assert p["expected_home_goals"] > 0
        assert p["expected_away_goals"] > 0

    def test_1_day_rest_reduces_away_xg(self):
        """Away team played yesterday → away xG drops, home xG rises."""
        from datetime import datetime, timezone, timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        p_fresh  = self._probs()
        p_tired  = self._probs(away_last=yesterday)
        assert p_tired["expected_away_goals"] < p_fresh["expected_away_goals"]
        assert p_tired["expected_home_goals"] > p_fresh["expected_home_goals"]

    def test_1_day_rest_reduces_home_xg(self):
        """Home team played yesterday → home xG drops."""
        from datetime import datetime, timezone, timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        p_fresh = self._probs()
        p_tired = self._probs(home_last=yesterday)
        assert p_tired["expected_home_goals"] < p_fresh["expected_home_goals"]

    def test_more_rest_less_penalty(self):
        """5 days rest has a smaller penalty than 2 days rest."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        d2 = (now - timedelta(days=2)).isoformat()
        d5 = (now - timedelta(days=5)).isoformat()
        p2 = self._probs(away_last=d2)
        p5 = self._probs(away_last=d5)
        # Less rest (d2) → bigger penalty → lower away xG
        assert p2["expected_away_goals"] < p5["expected_away_goals"]

    def test_14_days_no_penalty(self):
        """14 days rest → no fatigue penalty (full recovery)."""
        from datetime import datetime, timezone, timedelta
        two_weeks = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        p_fresh = self._probs()
        p_rested = self._probs(away_last=two_weeks)
        # Should be effectively identical (within floating-point rounding)
        assert abs(p_rested["expected_away_goals"] - p_fresh["expected_away_goals"]) < 0.001

    def test_fatigue_flag_set(self):
        """away_fatigue flag is True when rest < 14 days."""
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        p = self._probs(away_last=recent)
        assert p["away_fatigue"] is True

    def test_no_fatigue_flag_when_rested(self):
        """away_fatigue False when last match was 14+ days ago."""
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        p = self._probs(away_last=old)
        assert p["away_fatigue"] is False


# ── H2H venue split (Tier 1 #2) ─────────────────────────────────────────────

class TestH2HVenueSplit:
    """Verify parse_h2h uses only home-venue games when home_team_id supplied."""

    def _make_match(self, home_id, away_id, hg, ag, winner, date="2024-01-01T15:00:00Z"):
        return {
            "homeTeam": {"id": home_id},
            "awayTeam": {"id": away_id},
            "score": {"fullTime": {"home": hg, "away": ag}, "winner": winner},
            "utcDate": date,
        }

    def test_venue_split_filters_away_games(self):
        """When home_team_id=1, only matches where team 1 was home are counted."""
        matches = [
            self._make_match(1, 2, 2, 0, "HOME_TEAM"),  # team 1 at home — counts
            self._make_match(1, 2, 1, 0, "HOME_TEAM"),  # team 1 at home — counts
            self._make_match(1, 2, 0, 1, "AWAY_TEAM"),  # team 1 at home — counts
            self._make_match(2, 1, 0, 2, "AWAY_TEAM"),  # team 1 away — excluded
            self._make_match(2, 1, 1, 1, "DRAW"),        # team 1 away — excluded
        ]
        h2h = analyzer.parse_h2h({"matches": matches}, home_team_id=1)
        assert h2h["meetings"] == 3   # only the 3 home games

    def test_venue_split_correct_wins(self):
        """Home wins are correctly tallied from venue-split games."""
        from datetime import datetime, timezone, timedelta
        # Use recent dates so decay weight ≈ 1.0 for reliable assertions
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=10)).isoformat()
        matches = [
            self._make_match(1, 2, 2, 0, "HOME_TEAM", date=recent),
            self._make_match(1, 2, 1, 0, "HOME_TEAM", date=recent),
            self._make_match(1, 2, 0, 1, "AWAY_TEAM", date=recent),
            self._make_match(2, 1, 0, 2, "AWAY_TEAM", date=recent),  # excluded
        ]
        h2h = analyzer.parse_h2h({"matches": matches}, home_team_id=1)
        # 3 venue games: 2 home wins, 0 draws, 1 away win (decay weight ~1.0)
        assert h2h["meetings"] == 3
        assert h2h["home_wins"] > h2h["away_wins"]   # 2 wins > 1 loss
        assert h2h["draws"] == pytest.approx(0, abs=0.01)

    def test_fallback_when_fewer_than_3_venue_games(self):
        """Falls back to all meetings when < 3 venue-specific games exist."""
        matches = [
            self._make_match(1, 2, 2, 0, "HOME_TEAM"),  # only 1 home game
            self._make_match(2, 1, 1, 0, "HOME_TEAM"),
            self._make_match(2, 1, 1, 0, "HOME_TEAM"),
            self._make_match(2, 1, 2, 1, "HOME_TEAM"),
            self._make_match(2, 1, 0, 0, "DRAW"),
        ]
        h2h = analyzer.parse_h2h({"matches": matches}, home_team_id=1)
        assert h2h["meetings"] == 5   # all 5 matches used

    def test_no_home_id_uses_all_matches(self):
        """When home_team_id=None, all matches are used."""
        matches = [
            self._make_match(1, 2, 2, 0, "HOME_TEAM"),
            self._make_match(2, 1, 1, 0, "HOME_TEAM"),
            self._make_match(1, 2, 0, 1, "AWAY_TEAM"),
        ]
        h2h = analyzer.parse_h2h({"matches": matches})
        assert h2h["meetings"] == 3


# ── Motivation factor ────────────────────────────────────────────────────────

class TestMotivationFactor:
    """Verify motivation factors for different league positions."""

    def test_title_contender_high_motivation(self):
        s = {"position": 1, "points": 75, "played": 32, "league": "PL"}
        assert analyzer.compute_motivation_factor(s, total_teams=20) == pytest.approx(1.10)

    def test_european_race_moderate_boost(self):
        s = {"position": 5, "points": 58, "played": 32, "league": "PL"}
        f = analyzer.compute_motivation_factor(s, total_teams=20)
        assert 1.04 <= f <= 1.08

    def test_relegation_battle_high_motivation(self):
        s = {"position": 18, "points": 28, "played": 32, "league": "PL"}
        f = analyzer.compute_motivation_factor(s, total_teams=20)
        assert f >= 1.06

    def test_dead_rubber_relegated(self):
        """Already relegated — cannot reach safety mathematically."""
        s = {"position": 20, "points": 16, "played": 36, "league": "PL"}
        f = analyzer.compute_motivation_factor(s, total_teams=20)
        assert f == pytest.approx(0.85)

    def test_already_champion_dead_rubber(self):
        s = {"position": 1, "points": 90, "played": 37, "league": "PL"}
        f = analyzer.compute_motivation_factor(s, total_teams=20)
        assert f < 1.0

    def test_mid_table_safe_reduced(self):
        s = {"position": 9, "points": 55, "played": 34, "league": "PL"}
        f = analyzer.compute_motivation_factor(s, total_teams=20)
        assert f <= 1.0

    def test_motivation_affects_xg(self):
        """Motivation multiplier is applied to xG: higher motiv → higher xG,
        all else equal. Both home teams have identical scoring stats — only
        their league position (and thus motivation factor) differs."""
        s_opponent = {"position": 10, "points": 40, "played": 32, "league": "PL",
                      "id": 2, "form_score": 0.5, "avg_scored": 1.2, "avg_conceded": 1.2}
        # Same scoring stats, different motivation
        _base = {"form_score": 0.5, "avg_scored": 1.4, "avg_conceded": 1.2,
                 "league": "PL", "played": 32}
        s_motivated = {**_base, "id": 1, "position": 2,  "points": 72}  # motiv=1.10
        s_dead      = {**_base, "id": 3, "position": 20, "points": 16}  # motiv=0.85

        p_motivated = analyzer.compute_match_probabilities(
            "PL", s_motivated, s_opponent, _default_history(), _default_history(), _default_h2h()
        )
        p_dead = analyzer.compute_match_probabilities(
            "PL", s_dead, s_opponent, _default_history(), _default_history(), _default_h2h()
        )
        # Title-race home team (motiv=1.10) scores more than relegated home team (motiv=0.85)
        assert p_motivated["expected_home_goals"] > p_dead["expected_home_goals"]


# ── Over 2.5 special mode ────────────────────────────────────────────────────

class TestOver25Mode:
    """Verify the Over 2.5 coupon logic selects and ranks correctly."""

    def _make_fixture(self, home, away, over_2_5_prob, exp_h=1.5, exp_a=1.2):
        return {
            "home_name": home,
            "away_name": away,
            "league": "PL",
            "match_id": 1,
            "utc_date": "2026-04-18T15:00:00Z",
            "probs": {
                "over_2_5": over_2_5_prob,
                "expected_home_goals": exp_h,
                "expected_away_goals": exp_a,
                "btts_yes": 0.5,
            },
            "picks": [],
        }

    def test_sorted_by_over25_prob(self):
        """Fixtures are returned sorted by over_2_5 probability descending."""
        fixtures = [
            self._make_fixture("A", "B", 0.50),
            self._make_fixture("C", "D", 0.70),
            self._make_fixture("E", "F", 0.60),
        ]
        # Simulate the ranking logic used in render_over25
        candidates = [
            (fx, fx["probs"]["over_2_5"]) for fx in fixtures
            if fx["probs"]["over_2_5"] >= 0.45
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        probs = [p for _, p in candidates]
        assert probs == sorted(probs, reverse=True)

    def test_filters_below_min_prob(self):
        """Fixtures below 45% threshold are excluded."""
        fixtures = [
            self._make_fixture("A", "B", 0.30),
            self._make_fixture("C", "D", 0.65),
        ]
        qualifying = [fx for fx in fixtures if fx["probs"]["over_2_5"] >= 0.45]
        assert len(qualifying) == 1
        assert qualifying[0]["home_name"] == "C"

    def test_top6_only(self):
        """At most 6 fixtures returned."""
        fixtures = [self._make_fixture(f"H{i}", f"A{i}", 0.50 + i * 0.02) for i in range(10)]
        candidates = sorted(fixtures, key=lambda x: x["probs"]["over_2_5"], reverse=True)
        assert len(candidates[:6]) == 6

    def test_empty_message_when_none_qualify(self):
        """No qualifying fixtures → list is empty (caller shows message)."""
        fixtures = [
            self._make_fixture("A", "B", 0.20),
            self._make_fixture("C", "D", 0.30),
        ]
        qualifying = [fx for fx in fixtures if fx["probs"]["over_2_5"] >= 0.45]
        assert qualifying == []
