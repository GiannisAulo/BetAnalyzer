"""
test_phase1_filters.py — Tests for Phase 1 accuracy improvements.

Covers:
  1.1+1.2  Per-market minimum probability floors (MIN_PROB)
  1.3      Expected total contextual gates for Over/Under
  1.4      Tightened CL knockout suppression constants

Test categories per feature:
  - Happy path: valid picks pass through
  - Edge cases: boundary values, exact thresholds
  - Negative scenarios: invalid picks correctly rejected
"""
import pytest
import markets
from markets import MIN_PROB
from config import KNOCKOUT_GOALS_FACTOR, KNOCKOUT_OVER_PENALTY, KNOCKOUT_BTTS_PENALTY
from analyzer import compute_match_probabilities


# ── Shared fixture builders ──────────────────────────────────────────────────

def _base_probs(**overrides):
    """Build a probs dict with sensible defaults, overriding specific keys."""
    defaults = {
        "home": 0.45, "draw": 0.27, "away": 0.28,
        "over_1_5": 0.70, "under_1_5": 0.30,
        "over_2_5": 0.55, "under_2_5": 0.45,
        "over_3_5": 0.30, "under_3_5": 0.70,
        "btts_yes": 0.50, "btts_no": 0.50,
        "home_cs_rate": 0.25, "away_cs_rate": 0.20,
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# 1.1+1.2  MIN_PROB — Per-market minimum probability floors
# =============================================================================

class TestMinProbConstant:
    """Verify the MIN_PROB dict is defined with expected values."""

    def test_home_win_floor(self):
        assert MIN_PROB["Home Win"] == 0.62   # raised: live WR 44% at avg 0.55 — need stronger conviction

    def test_away_win_floor(self):
        assert MIN_PROB["Away Win"] == 0.65   # raised: 0W/6L live at 0.52 — no edge below 0.65

    def test_draw_floor(self):
        assert MIN_PROB["Draw"] == 0.45

    def test_over25_floor(self):
        assert MIN_PROB["Over 2.5"] == 0.65   # raised: 52% live WR at avg 0.64 — 12pp calibration gap

    def test_under25_floor(self):
        assert MIN_PROB["Under 2.5"] == 0.65  # raised: 46% live WR at avg 0.61

    def test_over35_floor(self):
        assert MIN_PROB["Over 3.5"] == 0.65   # raised: backtest 44% WR at 0.45

    def test_under35_floor(self):
        assert MIN_PROB["Under 3.5"] == 0.50


# ── 1X2 MIN_PROB enforcement ────────────────────────────────────────────────

class TestMinProb1X2:
    """MIN_PROB blocks low-probability 1X2 picks on both odds and no-odds paths."""

    # -- Happy path --

    def test_home_win_above_floor_with_odds(self):
        """Home Win at 0.65 (above 0.62 floor) with positive edge -> pick shown."""
        probs = _base_probs(home=0.65)
        picks = markets.evaluate_1x2(probs, odds_home=2.00, min_edge=3.0)
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home) == 1

    def test_away_win_above_floor_no_odds(self):
        """Away Win at 0.65 (above 0.50 floor and 0.65 no-odds threshold) -> pick shown."""
        probs = _base_probs(away=0.65)
        picks = markets.evaluate_1x2(probs)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert len(away) == 1

    def test_draw_above_floor_no_odds(self):
        """Draw at 0.65 (above 0.45 floor and 0.65 no-odds threshold) -> pick shown."""
        probs = _base_probs(draw=0.65)
        picks = markets.evaluate_1x2(probs)
        draw = [p for p in picks if p["pick"] == "Draw"]
        assert len(draw) == 1

    # -- Edge cases: exactly at MIN_PROB boundary --

    def test_away_win_exactly_at_floor_with_edge(self):
        """Away Win at 0.65 exactly (current floor boundary) with positive edge -> pick shown."""
        probs = _base_probs(away=0.65)
        # odds 2.50 -> implied 0.40, edge = (0.65 - 0.40)*100 = 25.0
        picks = markets.evaluate_1x2(probs, odds_away=2.50, min_edge=3.0)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert len(away) == 1
        assert away[0]["model_prob"] == 0.65

    def test_home_win_exactly_at_floor(self):
        """Home Win at 0.62 exactly (current floor) -> passes MIN_PROB (>= not >)."""
        probs = _base_probs(home=0.62)
        picks = markets.evaluate_1x2(probs, odds_home=2.00, min_edge=3.0)
        # implied = 0.50, edge = (0.62-0.50)*100 = 12.0
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home) == 1

    # -- Negative: below MIN_PROB --

    def test_away_win_below_floor_rejected_with_odds(self):
        """Away Win at 0.11 (below 0.25 floor) -> rejected even with huge odds edge.
        This is the Inter Cagliari scenario (row 14 in bets_log)."""
        probs = _base_probs(away=0.112)
        picks = markets.evaluate_1x2(probs, odds_away=15.50, min_edge=3.0)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert away == []

    def test_home_win_below_floor_rejected_with_odds(self):
        """Home Win at 0.25 (below 0.30 floor) -> rejected."""
        probs = _base_probs(home=0.25)
        picks = markets.evaluate_1x2(probs, odds_home=5.00, min_edge=3.0)
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert home == []

    def test_away_win_just_below_floor(self):
        """Away Win at 0.249 -> just below 0.25 floor, rejected."""
        probs = _base_probs(away=0.249)
        picks = markets.evaluate_1x2(probs, odds_away=5.00, min_edge=3.0)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert away == []

    def test_draw_below_floor_rejected(self):
        """Draw at 0.19 (below 0.20 floor) -> rejected even with huge odds."""
        probs = _base_probs(draw=0.19)
        picks = markets.evaluate_1x2(probs, odds_draw=8.00, min_edge=3.0)
        draw = [p for p in picks if p["pick"] == "Draw"]
        assert draw == []

    def test_floor_does_not_affect_no_odds_threshold(self):
        """Home Win at 0.45: fails MIN_PROB (0.55) floor — rejected before reaching no-odds threshold.
        Should NOT generate a pick without odds."""
        probs = _base_probs(home=0.45)
        picks = markets.evaluate_1x2(probs)
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert home == []


# ── Over/Under MIN_PROB enforcement ─────────────────────────────────────────

class TestMinProbOverUnder:
    """MIN_PROB blocks low-probability Over/Under picks."""

    def test_over25_below_floor_rejected_with_odds(self):
        """Over 2.5 at 0.50 (below 0.55 floor) -> rejected even with positive edge."""
        probs = _base_probs(over_2_5=0.50)
        picks = markets.evaluate_over_under(probs, odds_over25=2.20, min_edge=3.0)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_at_floor_accepted_with_odds(self):
        """Over 2.5 at 0.65 (current floor) with positive edge -> pick shown."""
        probs = _base_probs(over_2_5=0.65)
        # implied = 1/1.80 = 0.556, edge = (0.65-0.556)*100 = 9.4
        picks = markets.evaluate_over_under(probs, odds_over25=1.80, min_edge=3.0)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_under25_below_floor_rejected(self):
        """Under 2.5 at 0.48 (below 0.50 floor) -> rejected."""
        probs = _base_probs(under_2_5=0.48)
        picks = markets.evaluate_over_under(probs, odds_under25=2.20, min_edge=3.0)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert u25 == []

    def test_over35_below_floor_rejected(self):
        """Over 3.5 at 0.40 (below 0.45 floor) -> rejected."""
        probs = _base_probs(over_3_5=0.40)
        picks = markets.evaluate_over_under(probs, odds_over35=3.00, min_edge=3.0)
        o35 = [p for p in picks if p["pick"] == "Over 3.5"]
        assert o35 == []


# ── Over 2.5 raised no-odds threshold ──────────────────────────────────────

class TestOver25RaisedThreshold:
    """Over 2.5 no-odds threshold raised from 0.58 to 0.65."""

    def test_over25_at_old_threshold_rejected(self):
        """Over 2.5 at 0.58 (old threshold) -> no longer shown without odds."""
        probs = _base_probs(over_2_5=0.58)
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_at_064_rejected(self):
        """Over 2.5 at 0.64 -> still below new 0.65 threshold."""
        probs = _base_probs(over_2_5=0.64)
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_at_new_threshold_shown(self):
        """Over 2.5 at 0.67 -> exactly at no-odds threshold, shown."""
        probs = _base_probs(over_2_5=0.67)
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_over25_above_new_threshold_shown(self):
        """Over 2.5 at 0.72 -> well above threshold, shown."""
        probs = _base_probs(over_2_5=0.72)
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1
        assert o25[0]["model_prob"] >= 0.70

    def test_under25_threshold_unchanged(self):
        """Under 2.5 at 0.67 -> exactly at no-odds threshold, shown."""
        probs = _base_probs(under_2_5=0.67)
        picks = markets.evaluate_over_under(probs)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert len(u25) == 1

    def test_under35_threshold_unchanged(self):
        """Under 3.5 at 0.58 -> still uses 0.58 threshold."""
        probs = _base_probs(under_3_5=0.58)
        picks = markets.evaluate_over_under(probs)
        u35 = [p for p in picks if p["pick"] == "Under 3.5"]
        assert len(u35) == 1

    def test_over25_with_odds_ignores_no_odds_threshold(self):
        """Over 2.5 at 0.65 with odds -> edge logic, MIN_PROB (0.62) still applies."""
        probs = _base_probs(over_2_5=0.65)
        # implied = 1/1.80 = 0.556, edge = (0.65-0.556)*100 = 9.4
        picks = markets.evaluate_over_under(probs, odds_over25=1.80, min_edge=3.0)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1


# =============================================================================
# 1.3  Expected total contextual gates for Over/Under
# =============================================================================

class TestExpectedTotalGates:
    """expected_total parameter gates Over/Under picks based on the match's xG total."""

    # -- Happy path: expected_total supports the pick --

    def test_over25_high_xg_passes(self):
        """Over 2.5 with expected_total=3.2 (>= 2.8 gate) -> allowed."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=3.2)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_under25_low_xg_passes(self):
        """Under 2.5 with expected_total=2.0 (<= 2.50 cap) -> allowed."""
        # No-odds floor for Under 2.5 is 0.67; use that to clear the gate.
        probs = _base_probs(under_2_5=0.67)
        picks = markets.evaluate_over_under(probs, expected_total=2.0)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert len(u25) == 1

    def test_over35_high_xg_passes(self):
        """Over 3.5 with expected_total=4.0 (>= 3.5 gate) and prob >= 0.65 -> allowed."""
        probs = _base_probs(over_3_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=4.0)
        o35 = [p for p in picks if p["pick"] == "Over 3.5"]
        assert len(o35) == 1

    def test_under35_low_xg_passes(self):
        """Under 3.5 with expected_total=3.5 (<= 4.0 cap) -> allowed."""
        probs = _base_probs(under_3_5=0.65)
        picks = markets.evaluate_over_under(probs, expected_total=3.5)
        u35 = [p for p in picks if p["pick"] == "Under 3.5"]
        assert len(u35) == 1

    # -- Edge cases: exactly at gate boundary --

    def test_over25_exactly_at_gate(self):
        """Over 2.5 with expected_total=2.90 exactly -> passes (>= gate)."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=2.90)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_under25_exactly_at_cap(self):
        """Under 2.5 with expected_total=2.50 exactly -> passes (<= cap)."""
        probs = _base_probs(under_2_5=0.67)   # at no-odds floor
        picks = markets.evaluate_over_under(probs, expected_total=2.50)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert len(u25) == 1

    def test_over35_exactly_at_gate(self):
        """Over 3.5 with expected_total=3.50 exactly and prob >= 0.65 -> passes."""
        probs = _base_probs(over_3_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=3.50)
        o35 = [p for p in picks if p["pick"] == "Over 3.5"]
        assert len(o35) == 1

    def test_under35_exactly_at_cap(self):
        """Under 3.5 with expected_total=4.00 exactly -> passes."""
        probs = _base_probs(under_3_5=0.65)
        picks = markets.evaluate_over_under(probs, expected_total=4.00)
        u35 = [p for p in picks if p["pick"] == "Under 3.5"]
        assert len(u35) == 1

    # -- Negative: expected_total contradicts pick direction --

    def test_over25_low_xg_rejected(self):
        """Over 2.5 with expected_total=2.1 (< 2.8 gate) -> rejected.
        This blocks the model from recommending Over 2.5 in low-scoring fixtures."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=2.1)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_low_xg_rejected_with_odds(self):
        """Over 2.5 with odds AND low expected_total -> still rejected.
        The contextual gate fires before the edge check."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, odds_over25=1.80, min_edge=3.0,
                                            expected_total=2.1)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_under25_high_xg_rejected(self):
        """Under 2.5 with expected_total=3.0 (> 2.6 cap) -> rejected."""
        probs = _base_probs(under_2_5=0.65)
        picks = markets.evaluate_over_under(probs, expected_total=3.0)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert u25 == []

    def test_over35_low_xg_rejected(self):
        """Over 3.5 with expected_total=3.0 (< 3.5 gate) -> rejected."""
        probs = _base_probs(over_3_5=0.58)
        picks = markets.evaluate_over_under(probs, expected_total=3.0)
        o35 = [p for p in picks if p["pick"] == "Over 3.5"]
        assert o35 == []

    def test_under35_high_xg_rejected(self):
        """Under 3.5 with expected_total=4.5 (> 4.0 cap) -> rejected."""
        probs = _base_probs(under_3_5=0.65)
        picks = markets.evaluate_over_under(probs, expected_total=4.5)
        u35 = [p for p in picks if p["pick"] == "Under 3.5"]
        assert u35 == []

    # -- No expected_total: backward compatibility --

    def test_no_expected_total_skips_gate(self):
        """When expected_total is None (not passed), gates don't apply."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=None)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_default_arg_is_none(self):
        """Calling without expected_total= at all works (backward compat)."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    # -- Over/Under 1.5 has no gate --

    def test_over15_not_gated(self):
        """Over 1.5 has no expected_total gate — unaffected by low xG."""
        probs = _base_probs(over_1_5=0.80)
        picks = markets.evaluate_over_under(probs, expected_total=1.5)
        o15 = [p for p in picks if p["pick"] == "Over 1.5"]
        assert len(o15) == 1

    def test_under15_not_gated(self):
        """Under 1.5 has no expected_total gate."""
        probs = _base_probs(under_1_5=0.75)
        picks = markets.evaluate_over_under(probs, expected_total=4.0)
        u15 = [p for p in picks if p["pick"] == "Under 1.5"]
        assert len(u15) == 1

    # -- Just below gate boundary --

    def test_over25_just_below_gate(self):
        """Over 2.5 with expected_total=2.79 (just below 2.80) -> rejected."""
        probs = _base_probs(over_2_5=0.70)
        picks = markets.evaluate_over_under(probs, expected_total=2.79)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_under25_just_above_cap(self):
        """Under 2.5 with expected_total=2.61 (just above 2.60) -> rejected."""
        probs = _base_probs(under_2_5=0.65)
        picks = markets.evaluate_over_under(probs, expected_total=2.61)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert u25 == []


# =============================================================================
# 1.4  CL knockout suppression constants
# =============================================================================

class TestKnockoutConstants:
    """Knockout suppression constants are set from empirical data, not bet samples."""

    def test_goals_factor_near_neutral(self):
        """KNOCKOUT_GOALS_FACTOR should be near-neutral (0.90–1.00).
        CL 2023/24 empirical: knockout avg 3.35 vs group avg 3.08 (ratio 1.08).
        We use 0.93 — mild suppression pending more seasons of data."""
        assert 0.88 <= KNOCKOUT_GOALS_FACTOR <= 1.00

    def test_over_penalty_mild(self):
        """KNOCKOUT_OVER_PENALTY should be mild (>= 0.90)."""
        assert KNOCKOUT_OVER_PENALTY >= 0.90

    def test_btts_penalty_mild(self):
        """KNOCKOUT_BTTS_PENALTY should be mild (>= 0.90)."""
        assert KNOCKOUT_BTTS_PENALTY >= 0.90


class TestKnockoutProbabilities:
    """Knockout suppression produces lower probabilities than league games."""

    @pytest.fixture
    def match_args(self):
        """Shared arguments for compute_match_probabilities in league vs knockout."""
        home_standing = {
            "id": 1, "form_score": 0.7, "position": 3, "avg_scored": 1.8,
            "avg_conceded": 0.9, "played": 20, "points": 45, "league": "CL",
        }
        away_standing = {
            "id": 2, "form_score": 0.6, "position": 5, "avg_scored": 1.5,
            "avg_conceded": 1.1, "played": 20, "points": 38, "league": "CL",
        }
        from analyzer import _default_team_stats
        home_hist = _default_team_stats()
        away_hist = _default_team_stats()
        h2h = {"meetings": 0}
        return {
            "league_code": "CL",
            "home_standing": home_standing,
            "away_standing": away_standing,
            "home_history": home_hist,
            "away_history": away_hist,
            "h2h": h2h,
        }

    def test_knockout_reduces_expected_goals(self, match_args):
        """Knockout matches have lower expected goals than league games."""
        league = compute_match_probabilities(**match_args, is_knockout=False)
        knockout = compute_match_probabilities(**match_args, is_knockout=True)
        assert knockout["expected_total"] < league["expected_total"]

    def test_knockout_reduces_over25(self, match_args):
        """Over 2.5 probability is lower in knockout matches."""
        league = compute_match_probabilities(**match_args, is_knockout=False)
        knockout = compute_match_probabilities(**match_args, is_knockout=True)
        assert knockout["over_2_5"] < league["over_2_5"]

    def test_knockout_reduces_btts(self, match_args):
        """BTTS Yes probability is lower in knockout matches."""
        league = compute_match_probabilities(**match_args, is_knockout=False)
        knockout = compute_match_probabilities(**match_args, is_knockout=True)
        assert knockout["btts_yes"] < league["btts_yes"]

    def test_knockout_xg_reduction_matches_factor(self, match_args):
        """xG ratio should match KNOCKOUT_GOALS_FACTOR (currently 0.93 — mild suppression)."""
        league = compute_match_probabilities(**match_args, is_knockout=False)
        knockout = compute_match_probabilities(**match_args, is_knockout=True)
        ratio = knockout["expected_total"] / league["expected_total"]
        assert ratio == pytest.approx(KNOCKOUT_GOALS_FACTOR, abs=0.02)


# =============================================================================
# Combined filter interaction tests
# =============================================================================

class TestFilterInteractions:
    """Verify filters interact correctly — MIN_PROB + expected_total + threshold."""

    def test_over25_passes_min_prob_but_fails_gate(self):
        """Over 2.5 at 0.60 passes MIN_PROB (0.55) but fails gate (xG 2.3 < 2.8)."""
        probs = _base_probs(over_2_5=0.60)
        picks = markets.evaluate_over_under(probs, odds_over25=1.80, min_edge=3.0,
                                            expected_total=2.3)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_passes_gate_but_fails_min_prob(self):
        """Over 2.5 at 0.50 passes gate (xG 3.0) but fails MIN_PROB (0.55)."""
        probs = _base_probs(over_2_5=0.50)
        picks = markets.evaluate_over_under(probs, odds_over25=2.50, min_edge=3.0,
                                            expected_total=3.0)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []

    def test_over25_passes_all_filters(self):
        """Over 2.5 at 0.65 with xG 3.0 and positive edge -> passes all filters."""
        probs = _base_probs(over_2_5=0.65)
        # implied = 1/1.80 = 0.556, edge = (0.65-0.556)*100 = 9.4
        picks = markets.evaluate_over_under(probs, odds_over25=1.80, min_edge=3.0,
                                            expected_total=3.0)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_under25_passes_all_filters_no_odds(self):
        """Under 2.5 at 0.67 (no-odds floor) with xG 2.1 -> passes MIN_PROB, gate, threshold."""
        probs = _base_probs(under_2_5=0.67)
        picks = markets.evaluate_over_under(probs, expected_total=2.1)
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert len(u25) == 1

    def test_multiple_picks_filtered_independently(self):
        """Over 2.5 blocked by gate but Under 2.5 passes independently."""
        probs = _base_probs(over_2_5=0.70, under_2_5=0.67)
        picks = markets.evaluate_over_under(probs, expected_total=2.3)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        u25 = [p for p in picks if p["pick"] == "Under 2.5"]
        assert o25 == []      # blocked by Over 2.5 gate (2.3 < 2.90)
        assert len(u25) == 1  # passes Under 2.5 cap (2.3 <= 2.50)


# =============================================================================
# Regression tests — real scenarios from bets_log.csv
# =============================================================================

class TestRealScenarios:
    """Reproduce real losing scenarios from the log and verify they'd now be filtered."""

    def test_inter_away_win_011_rejected(self):
        """Row 14: Inter vs Cagliari Away Win at 0.112, odds 15.50 -> lost.
        MIN_PROB floor (0.25) should reject this."""
        probs = _base_probs(away=0.112)
        picks = markets.evaluate_1x2(probs, odds_away=15.50, min_edge=3.0)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert away == []

    def test_werder_away_win_032_rejected(self):
        """Row 39: Werder Bremen Away Win at 0.321, odds 4.70 -> lost.
        Now rejected by MIN_PROB (0.50) — the raised floor catches it."""
        probs = _base_probs(away=0.321)
        picks = markets.evaluate_1x2(probs, odds_away=4.70, min_edge=5.0)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert away == []

    def test_over25_low_xg_match_rejected(self):
        """Simulate a match like Casa Pia vs Santa Clara (row 21) with
        Over 2.5 at 0.643, odds 2.52 -> lost. If expected_total was ~2.4,
        the gate should block it."""
        probs = _base_probs(over_2_5=0.643)
        picks = markets.evaluate_over_under(probs, odds_over25=2.52, min_edge=3.0,
                                            expected_total=2.4)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert o25 == []
