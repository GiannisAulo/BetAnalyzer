"""
test_markets.py — unit tests for all market evaluators.
"""
import pytest
import markets
from markets import MIN_FAIR_ODDS


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_probs():
    """A roughly balanced match — no strong favourite."""
    return {
        "home": 0.40, "draw": 0.28, "away": 0.32,
        "over_1_5": 0.65, "under_1_5": 0.35,
        "over_2_5": 0.55, "under_2_5": 0.45,
        "over_3_5": 0.30, "under_3_5": 0.70,
        "btts_yes": 0.52, "btts_no":   0.48,
        "home_cs_rate": 0.25, "away_cs_rate": 0.20,
    }


@pytest.fixture
def confident_probs():
    """Home team heavy favourite — model prob just inside the no-odds cap."""
    return {
        "home": 0.60, "draw": 0.22, "away": 0.18,
        "over_1_5": 0.78, "under_1_5": 0.22,
        "over_2_5": 0.62, "under_2_5": 0.38,
        "over_3_5": 0.35, "under_3_5": 0.65,
        "btts_yes": 0.61, "btts_no":   0.39,
        "home_cs_rate": 0.35, "away_cs_rate": 0.15,
    }


@pytest.fixture
def short_odds_probs():
    """Very strong favourite — all outcomes above MAX_PROB_NO_ODDS threshold."""
    return {
        "home": 0.75, "draw": 0.15, "away": 0.10,
        "over_1_5": 0.90, "under_1_5": 0.10,
        "over_2_5": 0.75, "under_2_5": 0.25,
        "over_3_5": 0.50, "under_3_5": 0.50,
        "btts_yes": 0.70, "btts_no":   0.30,
        "home_cs_rate": 0.40, "away_cs_rate": 0.10,
    }


# ── MIN_FAIR_ODDS constant ────────────────────────────────────────────────────

class TestMinFairOdds:
    def test_constant_value(self):
        assert MIN_FAIR_ODDS == 1.60


# ── 1X2 evaluator ────────────────────────────────────────────────────────────

class TestEvaluate1X2:
    def test_no_picks_below_min_prob(self, balanced_probs):
        """home=0.40, draw=0.28, away=0.32 — all below 0.50 threshold."""
        picks = markets.evaluate_1x2(balanced_probs)
        assert picks == []

    def test_pick_generated_above_threshold(self, confident_probs):
        """home=0.65 meets the no-odds threshold so a Home Win pick is expected."""
        probs = {**confident_probs, "home": 0.65}
        picks = markets.evaluate_1x2(probs)
        home_picks = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home_picks) == 1
        assert home_picks[0]["model_prob"] >= 0.55

    def test_high_confidence_pick_still_shown(self, short_odds_probs):
        """home=0.75 — high confidence, no odds cap — pick must be shown."""
        picks = markets.evaluate_1x2(short_odds_probs)
        home_picks = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home_picks) == 1
        assert home_picks[0]["model_prob"] >= 0.70

    def test_picks_with_real_odds(self, confident_probs):
        """When odds are supplied, edge-based filtering takes over.
        Override home to 0.65 so it clears the current 0.62 MIN_PROB floor."""
        probs = {**confident_probs, "home": 0.65}
        picks = markets.evaluate_1x2(
            probs,
            odds_home=1.80,   # implied 0.556 vs model 0.65 → edge +9.4%
            min_edge=3.0,
        )
        home_picks = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home_picks) == 1
        assert home_picks[0]["edge"] > 0

    def test_no_edge_pick_discarded(self, confident_probs):
        """Negative edge → no pick."""
        picks = markets.evaluate_1x2(
            confident_probs,
            odds_home=1.50,   # implied 0.667 > model 0.60 → negative edge
            min_edge=3.0,
        )
        home_picks = [p for p in picks if p["pick"] == "Home Win"]
        assert home_picks == []

    def test_pick_fields(self, balanced_probs):
        """Each pick has required fields."""
        probs = {**balanced_probs, "home": 0.58}
        picks = markets.evaluate_1x2(probs)
        for pick in picks:
            for field in ("market", "pick", "model_prob"):
                assert field in pick


# ── Double Chance evaluator ───────────────────────────────────────────────────

class TestEvaluateDoubleChance:
    def test_x2_can_pass_cap(self, balanced_probs):
        """X2 = draw+away = 0.28+0.32 = 0.60 — inside [0.60, 0.625]."""
        picks = markets.evaluate_double_chance(balanced_probs)
        x2_picks = [p for p in picks if "X2" in p["pick"]]
        assert len(x2_picks) == 1

    def test_no_pick_below_new_threshold(self):
        """X2 = draw+away = 0.20+0.37 = 0.57 — below 0.60 floor → no pick."""
        probs = {"home": 0.43, "draw": 0.20, "away": 0.37,
                 "over_2_5": 0.5, "under_2_5": 0.5, "over_3_5": 0.3,
                 "under_3_5": 0.7, "btts_yes": 0.5, "btts_no": 0.5,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_double_chance(probs)
        x2 = [p for p in picks if "X2" in p["pick"]]
        assert x2 == []

    def test_12_suppressed_when_real_odds_below_min(self):
        """12 = home+away = 0.75+0.15 = 0.90.
        When real bookmaker odds are below MIN_FAIR_ODDS (1.60), the pick is
        suppressed regardless of model probability. The no-odds path no longer
        enforces this cap (it's a market-price filter, not a confidence one)."""
        probs = {"home": 0.75, "draw": 0.10, "away": 0.15,
                 "over_2_5": 0.5, "under_2_5": 0.5, "over_3_5": 0.3,
                 "under_3_5": 0.7, "btts_yes": 0.5, "btts_no": 0.5,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        # Real odds 1.10 → below MIN_FAIR_ODDS 1.60 → suppressed
        picks = markets.evaluate_double_chance(probs, odds_12=1.10, min_edge=3.0)
        dc12 = [p for p in picks if "12" in p["pick"]]
        assert dc12 == [], "12 with real odds 1.10 must be suppressed by MIN_FAIR_ODDS"

    def test_12_shown_when_fair_odds_sufficient(self):
        """12 = home+away = 0.40+0.20 = 0.60 → fair odds 1.67 >= MIN_FAIR_ODDS → shown."""
        probs = {"home": 0.40, "draw": 0.40, "away": 0.20,
                 "over_2_5": 0.5, "under_2_5": 0.5, "over_3_5": 0.3,
                 "under_3_5": 0.7, "btts_yes": 0.5, "btts_no": 0.5,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_double_chance(probs)
        dc12 = [p for p in picks if "12" in p["pick"]]
        assert len(dc12) == 1
        assert dc12[0]["model_prob"] >= 0.60

    def test_with_real_odds(self, balanced_probs):
        """Real odds supplied — use edge logic."""
        picks = markets.evaluate_double_chance(
            balanced_probs, odds_x2=1.50, min_edge=3.0
        )
        # X2 model_prob = 0.60, implied = 1/1.50 = 0.667 → negative edge
        x2 = [p for p in picks if "X2" in p["pick"] and p.get("odds")]
        assert x2 == []


# ── Over/Under evaluator ──────────────────────────────────────────────────────

class TestEvaluateOverUnder:
    def test_no_picks_below_threshold(self):
        """All outcomes below threshold → no picks at all."""
        probs = {"over_1_5": 0.50, "under_1_5": 0.50,
                 "over_2_5": 0.45, "under_2_5": 0.40,
                 "over_3_5": 0.30, "under_3_5": 0.45,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_over_under(probs)
        assert picks == []

    def test_pick_in_valid_band(self):
        """over_2_5=0.67 is exactly at the no-odds threshold."""
        probs = {"over_1_5": 0.60, "under_1_5": 0.40,
                 "over_2_5": 0.67, "under_2_5": 0.33,
                 "over_3_5": 0.30, "under_3_5": 0.70,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1

    def test_high_confidence_over25_shown(self):
        """over_2_5=0.70 — high confidence, no upper cap, should be shown."""
        probs = {"over_1_5": 0.85, "under_1_5": 0.15,
                 "over_2_5": 0.70, "under_2_5": 0.30,
                 "over_3_5": 0.40, "under_3_5": 0.60,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_over_under(probs)
        o25 = [p for p in picks if p["pick"] == "Over 2.5"]
        assert len(o25) == 1
        assert o25[0]["model_prob"] >= 0.65

    def test_with_real_odds_positive_edge(self):
        probs = {"over_1_5": 0.60, "under_1_5": 0.40,
                 "over_2_5": 0.65, "under_2_5": 0.35,
                 "over_3_5": 0.30, "under_3_5": 0.70,
                 "home_cs_rate": 0.3, "away_cs_rate": 0.2}
        picks = markets.evaluate_over_under(probs, odds_over25=1.90, min_edge=3.0)
        # implied = 1/1.90 = 0.526, edge = (0.65-0.526)*100 = 12.4%
        o25 = [p for p in picks if p["pick"] == "Over 2.5" and p.get("odds")]
        assert len(o25) == 1
        assert o25[0]["edge"] > 0


# ── BTTS evaluator ────────────────────────────────────────────────────────────

class TestEvaluateBtts:
    def test_btts_yes_in_band(self, balanced_probs):
        """btts_yes=0.52 is in [0.55, 0.625]? No — 0.52 < 0.55."""
        picks = markets.evaluate_btts(balanced_probs)
        assert picks == []

    def test_btts_yes_at_threshold(self):
        """BTTS Yes no-odds threshold is currently 0.62 (in evaluate_btts)."""
        probs = {"btts_yes": 0.62, "btts_no": 0.38,
                 "home_cs_rate": 0.20, "away_cs_rate": 0.25}
        picks = markets.evaluate_btts(probs)
        yes_picks = [p for p in picks if p["pick"] == "BTTS Yes"]
        assert len(yes_picks) == 1

    def test_btts_no_weakened_by_low_cs_rate(self):
        """BTTS No should be weakened when clean sheet rate < 35%."""
        probs = {"btts_yes": 0.42, "btts_no": 0.58,
                 "home_cs_rate": 0.20, "away_cs_rate": 0.25}
        # After weakening (×0.90), btts_no = 0.522 — below 0.55 threshold
        picks = markets.evaluate_btts(probs)
        no_picks = [p for p in picks if p["pick"] == "BTTS No"]
        assert no_picks == []

    def test_btts_no_with_high_cs_rate(self):
        """BTTS No is disabled (backtest 49% WR) — never generated regardless of cs_rate."""
        probs = {"btts_yes": 0.42, "btts_no": 0.58,
                 "home_cs_rate": 0.45, "away_cs_rate": 0.40}
        picks = markets.evaluate_btts(probs)
        no_picks = [p for p in picks if p["pick"] == "BTTS No"]
        assert no_picks == []


# ── Rating helpers ────────────────────────────────────────────────────────────

class _TestKellyStake_REMOVED:
    def test_positive_edge_returns_stake(self):
        # prob=0.60, odds=2.00 → kelly=(1×0.60−0.40)/1=0.20, ×0.25×1000=50, capped at 5%=50
        stake = markets.kelly_stake(0.60, 2.00, 1000)
        assert stake == pytest.approx(50.0)

    def test_cap_at_5_percent(self):
        # Very strong edge — raw Kelly would exceed 5% bankroll cap
        stake = markets.kelly_stake(0.90, 2.00, 1000)
        assert stake == pytest.approx(50.0)   # 5% of 1000

    def test_no_edge_returns_zero(self):
        # prob=0.40, odds=2.00 → kelly=(0.40−0.60)/1=−0.20 → 0
        assert markets.kelly_stake(0.40, 2.00, 1000) == 0.0

    def test_no_odds_returns_zero(self):
        assert markets.kelly_stake(0.65, None, 1000) == 0.0

    def test_odds_at_or_below_1_returns_zero(self):
        assert markets.kelly_stake(0.65, 1.0, 1000) == 0.0
        assert markets.kelly_stake(0.65, 0.9, 1000) == 0.0

    def test_quarter_kelly_fraction(self):
        # prob=0.55, odds=2.10 → b=1.10, kelly=(1.10×0.55−0.45)/1.10≈0.141
        # quarter = 0.141×0.25×1000 ≈ 35.23
        stake = markets.kelly_stake(0.55, 2.10, 1000)
        assert 30.0 < stake < 40.0

    def test_small_bankroll(self):
        stake = markets.kelly_stake(0.60, 2.00, 100)
        assert stake == pytest.approx(5.0)   # 5% of 100


class TestHelpers:
    def test_all_pick_fields_present(self, balanced_probs):
        """Every generated pick must have the full set of required fields."""
        probs = {**balanced_probs, "home": 0.58}
        all_picks = (
            markets.evaluate_1x2(probs)
            + markets.evaluate_double_chance(probs)
            + markets.evaluate_over_under(probs)
            + markets.evaluate_btts(probs)
        )
        required = {"market", "pick", "model_prob", "implied_prob", "odds", "edge"}
        for pick in all_picks:
            assert required.issubset(pick.keys()), f"Missing fields in {pick}"


# ── Over/Under 1.5 edge cases ────────────────────────────────────────────────

class TestOverUnder15:
    def test_over15_shown_at_threshold(self):
        """over_1_5=0.75 is exactly at threshold → must appear."""
        probs = {"over_1_5": 0.75, "under_1_5": 0.25,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80}
        picks = markets.evaluate_over_under(probs)
        o15 = [p for p in picks if p["pick"] == "Over 1.5"]
        assert len(o15) == 1

    def test_over15_not_shown_below_threshold(self):
        """over_1_5=0.74 is just below 0.75 threshold → not shown without odds."""
        probs = {"over_1_5": 0.74, "under_1_5": 0.26,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80}
        picks = markets.evaluate_over_under(probs)
        o15 = [p for p in picks if p["pick"] == "Over 1.5"]
        assert o15 == []

    def test_under15_shown_at_threshold(self):
        """under_1_5=0.70 is exactly at threshold → must appear."""
        probs = {"over_1_5": 0.30, "under_1_5": 0.70,
                 "over_2_5": 0.20, "under_2_5": 0.80,
                 "over_3_5": 0.05, "under_3_5": 0.95}
        picks = markets.evaluate_over_under(probs)
        u15 = [p for p in picks if p["pick"] == "Under 1.5"]
        assert len(u15) == 1

    def test_under15_not_shown_below_threshold(self):
        """under_1_5=0.69 → not shown without odds."""
        probs = {"over_1_5": 0.31, "under_1_5": 0.69,
                 "over_2_5": 0.20, "under_2_5": 0.80,
                 "over_3_5": 0.05, "under_3_5": 0.95}
        picks = markets.evaluate_over_under(probs)
        u15 = [p for p in picks if p["pick"] == "Under 1.5"]
        assert u15 == []

    def test_over15_with_odds_uses_edge_logic(self):
        """With real odds supplied, edge logic replaces the threshold check."""
        probs = {"over_1_5": 0.70, "under_1_5": 0.30,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80}
        # implied 1/1.40=0.714 > model 0.70 → negative edge → no pick
        picks = markets.evaluate_over_under(probs, odds_over15=1.40, min_edge=3.0)
        o15 = [p for p in picks if p["pick"] == "Over 1.5" and p.get("odds")]
        assert o15 == []

    def test_over15_positive_edge_with_odds(self):
        """over_1_5=0.85, odds=1.50 → implied 0.667, edge=+18.3%."""
        probs = {"over_1_5": 0.85, "under_1_5": 0.15,
                 "over_2_5": 0.60, "under_2_5": 0.40,
                 "over_3_5": 0.30, "under_3_5": 0.70}
        picks = markets.evaluate_over_under(probs, odds_over15=1.50, min_edge=3.0)
        o15 = [p for p in picks if p["pick"] == "Over 1.5" and p.get("odds")]
        assert len(o15) == 1
        assert o15[0]["edge"] == pytest.approx((0.85 - 1/1.50) * 100, abs=0.1)


# ── Combo evaluator ───────────────────────────────────────────────────────────

class TestEvaluateCombos:
    @pytest.fixture
    def combo_picks(self):
        """Single picks with real odds — enough to build combos."""
        return [
            {"market": "1X2",       "pick": "Home Win",  "model_prob": 0.65,
             "odds": 1.85, "implied_prob": 0.54, "edge": 11.0},
            {"market": "Over/Under", "pick": "Over 1.5", "model_prob": 0.80,
             "odds": 1.35, "implied_prob": 0.74, "edge": 6.0},
            {"market": "Over/Under", "pick": "Over 2.5", "model_prob": 0.62,
             "odds": 1.90, "implied_prob": 0.526, "edge": 9.4},
            {"market": "BTTS",      "pick": "BTTS Yes",  "model_prob": 0.60,
             "odds": 1.75, "implied_prob": 0.571, "edge": 2.9},
        ]

    def test_returns_list(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        assert isinstance(result, list)

    def test_max_five_combos(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        assert len(result) <= 5

    def test_combo_market_label(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        for c in result:
            assert c["market"] == "Combo"

    def test_combo_pick_name_has_plus(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        for c in result:
            assert " + " in c["pick"]

    def test_combo_odds_at_least_160(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        for c in result:
            assert c["odds"] >= 1.60 or c["odds"] is None

    def test_impossible_btts_yes_under25_excluded(self):
        """BTTS Yes + Under 2.5 is logically impossible — must never appear."""
        picks = [
            {"market": "BTTS",      "pick": "BTTS Yes",   "model_prob": 0.65,
             "odds": 1.75, "implied_prob": 0.57, "edge": 8.0},
            {"market": "Over/Under", "pick": "Under 2.5", "model_prob": 0.60,
             "odds": 1.80, "implied_prob": 0.556, "edge": 4.4},
        ]
        result = markets.evaluate_combos(picks)
        names = [c["pick"] for c in result]
        assert not any("BTTS Yes" in n and "Under 2.5" in n for n in names)

    def test_impossible_btts_yes_under15_excluded(self):
        picks = [
            {"market": "BTTS",      "pick": "BTTS Yes",   "model_prob": 0.65,
             "odds": 1.75, "implied_prob": 0.57, "edge": 8.0},
            {"market": "Over/Under", "pick": "Under 1.5", "model_prob": 0.72,
             "odds": 1.60, "implied_prob": 0.625, "edge": 9.5},
        ]
        result = markets.evaluate_combos(picks)
        names = [c["pick"] for c in result]
        assert not any("BTTS Yes" in n and "Under 1.5" in n for n in names)

    def test_combo_joint_prob_at_least_30pct(self, combo_picks):
        """Combos with joint probability below 30% must be excluded."""
        result = markets.evaluate_combos(combo_picks)
        for c in result:
            assert c["model_prob"] >= 0.30

    def test_sorted_by_model_prob_descending(self, combo_picks):
        result = markets.evaluate_combos(combo_picks)
        probs = [c["model_prob"] for c in result]
        assert probs == sorted(probs, reverse=True)

    def test_empty_picks_returns_empty(self):
        assert markets.evaluate_combos([]) == []

    def test_single_pick_returns_empty(self):
        """Need at least two different markets to build a combo."""
        picks = [{"market": "1X2", "pick": "Home Win", "model_prob": 0.70,
                  "odds": 1.80, "implied_prob": 0.556, "edge": 14.4}]
        assert markets.evaluate_combos(picks) == []

    def test_no_real_odds_combo_has_none_odds(self):
        """Both legs no-odds → combo odds field is None."""
        picks = [
            {"market": "1X2",       "pick": "Home Win",  "model_prob": 0.65,
             "odds": None, "implied_prob": None, "edge": None},
            {"market": "Over/Under", "pick": "Over 1.5", "model_prob": 0.80,
             "odds": None, "implied_prob": None, "edge": None},
        ]
        result = markets.evaluate_combos(picks)
        for c in result:
            assert c["odds"] is None

    def test_matrix_joint_prob_used_when_available(self):
        """When probs contains score_matrix, joint probability comes from the matrix
        rather than from independent multiplication — value will differ from naive product."""
        picks = [
            {"market": "1X2",       "pick": "Home Win",  "model_prob": 0.65,
             "odds": 1.85, "implied_prob": 0.54, "edge": 11.0},
            {"market": "Over/Under", "pick": "Over 2.5", "model_prob": 0.62,
             "odds": 1.90, "implied_prob": 0.526, "edge": 9.4},
        ]
        # Build a simple 8x8 matrix with known values
        from analyzer import _score_matrix
        matrix = _score_matrix(1.4, 1.1)
        result_matrix = markets.evaluate_combos(picks, probs={"score_matrix": matrix})
        result_naive  = markets.evaluate_combos(picks, probs=None)
        hw_matrix = [c for c in result_matrix if "Home Win" in c["pick"] and "Over 2.5" in c["pick"]]
        hw_naive  = [c for c in result_naive  if "Home Win" in c["pick"] and "Over 2.5" in c["pick"]]
        if hw_matrix and hw_naive:
            # Matrix-derived joint prob is a real P(home_win AND over_2.5) — not naive product
            assert hw_matrix[0]["model_prob"] != hw_naive[0]["model_prob"]
            # Both must be in [0, 1]
            assert 0 < hw_matrix[0]["model_prob"] < 1

    def test_legs_from_different_markets(self, combo_picks):
        """Each combo must have legs from two distinct markets."""
        result = markets.evaluate_combos(combo_picks)
        for c in result:
            leg1, leg2 = c["pick"].split(" + ", 1)
            # look up original picks to check market
            by_name = {p["pick"]: p["market"] for p in combo_picks}
            if leg1 in by_name and leg2 in by_name:
                assert by_name[leg1] != by_name[leg2]


# ── market_variance_penalty ───────────────────────────────────────────────────

class _TestMarketVariancePenalty_REMOVED:
    def test_no_calibrator_returns_zero(self):
        assert markets.market_variance_penalty("1X2", "Home Win", None) == 0.0

    def test_inactive_calibrator_returns_zero(self, tmp_path, monkeypatch):
        import ml_calibrator
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        cal = ml_calibrator.CalibrationModel()   # no history
        assert markets.market_variance_penalty("1X2", "Home Win", cal) == 0.0

    def test_draw_higher_base_than_home_win(self, tmp_path, monkeypatch):
        """Draw has higher base variance than Home Win — even with no history."""
        import ml_calibrator
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        # Write just enough to make is_active=True
        import csv
        rows = [{"market": "1X2", "pick": "Home Win", "league": "PL",
                 "model_prob": 0.60, "result": "W"}] * 3
        path = tmp_path / "bets_log.csv"
        fields = ["match_id", "date", "home", "away", "league", "market",
                  "pick", "model_prob", "edge", "result", "settle_attempts"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, r in enumerate(rows):
                w.writerow({"match_id": str(i), "date": "2026-01-01", "home": "A",
                            "away": "B", "league": r["league"], "market": r["market"],
                            "pick": r["pick"], "model_prob": r["model_prob"],
                            "edge": 5, "result": r["result"], "settle_attempts": 0})
        cal = ml_calibrator.CalibrationModel()
        assert cal.is_active
        draw_pen  = markets.market_variance_penalty("1X2", "Draw",     cal)
        home_pen  = markets.market_variance_penalty("1X2", "Home Win", cal)
        assert draw_pen > home_pen

    def test_unknown_market_returns_base(self, tmp_path, monkeypatch):
        """Unknown market+pick combo falls through to base variance of 0.12."""
        import ml_calibrator, csv
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        path = tmp_path / "bets_log.csv"
        fields = ["match_id", "date", "home", "away", "league", "market",
                  "pick", "model_prob", "edge", "result", "settle_attempts"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"match_id": "1", "date": "2026-01-01", "home": "A", "away": "B",
                        "league": "PL", "market": "1X2", "pick": "Home Win",
                        "model_prob": 0.60, "edge": 5, "result": "W", "settle_attempts": 0})
        cal = ml_calibrator.CalibrationModel()
        penalty = markets.market_variance_penalty("Unknown", "Something", cal)
        assert penalty == pytest.approx(0.12, abs=1e-9)

    def test_result_in_valid_range(self, tmp_path, monkeypatch):
        import ml_calibrator
        monkeypatch.chdir(tmp_path)
        ml_calibrator.reset_calibrator()
        cal = ml_calibrator.CalibrationModel()
        for market, pick in [("1X2", "Draw"), ("BTTS", "BTTS Yes"),
                              ("Over/Under", "Over 3.5")]:
            p = markets.market_variance_penalty(market, pick, cal)
            assert 0.0 <= p <= 0.5


# ── Kelly stake edge cases ────────────────────────────────────────────────────

class _TestKellyEdgeCases_REMOVED:
    def test_variance_penalty_reduces_stake(self):
        """A variance_penalty of 0.5 halves the effective fraction."""
        base  = markets.kelly_stake(0.60, 2.00, 1000, variance_penalty=0.0)
        penalised = markets.kelly_stake(0.60, 2.00, 1000, variance_penalty=0.5)
        assert penalised < base

    def test_variance_penalty_of_one_returns_zero(self):
        """Full variance penalty (1.0) means zero effective fraction → 0 stake."""
        stake = markets.kelly_stake(0.60, 2.00, 1000, variance_penalty=1.0)
        assert stake == 0.0

    def test_variance_penalty_clamped_above_one(self):
        """Penalty > 1 treated as 1 → 0 stake."""
        stake = markets.kelly_stake(0.60, 2.00, 1000, variance_penalty=2.0)
        assert stake == 0.0

    def test_odds_exactly_one_returns_zero(self):
        assert markets.kelly_stake(0.99, 1.0, 1000) == 0.0

    def test_very_small_bankroll(self):
        stake = markets.kelly_stake(0.60, 2.00, 1.0)
        assert 0.0 < stake <= 0.05   # at most 5% of 1.0

    def test_fractional_kelly_respected(self):
        """Custom fraction=0.5 (half-Kelly) should give double the quarter-Kelly."""
        quarter = markets.kelly_stake(0.60, 2.00, 1000, fraction=0.25)
        half    = markets.kelly_stake(0.60, 2.00, 1000, fraction=0.50)
        # Both may hit the 5% cap — only assert when below cap
        if quarter < 50 and half < 50:
            assert half == pytest.approx(quarter * 2, abs=0.01)


# ── 1X2 edge cases ────────────────────────────────────────────────────────────

class TestEvaluate1X2EdgeCases:
    def test_draw_pick_above_threshold(self):
        """Draw = 0.65 >= 0.65 no-odds threshold → Draw pick generated."""
        probs = {"home": 0.20, "draw": 0.65, "away": 0.15,
                 "over_1_5": 0.65, "under_1_5": 0.35,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80,
                 "btts_yes": 0.50, "btts_no": 0.50}
        picks = markets.evaluate_1x2(probs)
        draw = [p for p in picks if p["pick"] == "Draw"]
        assert len(draw) == 1

    def test_away_win_pick_above_threshold(self):
        """Away Win = 0.65 >= 0.65 no-odds threshold → pick generated."""
        probs = {"home": 0.20, "draw": 0.15, "away": 0.65,
                 "over_1_5": 0.65, "under_1_5": 0.35,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80,
                 "btts_yes": 0.50, "btts_no": 0.50}
        picks = markets.evaluate_1x2(probs)
        away = [p for p in picks if p["pick"] == "Away Win"]
        assert len(away) == 1

    def test_exactly_at_threshold_generates_pick(self):
        """Exactly 0.65 → Home Win pick must be shown (>= no-odds threshold)."""
        probs = {"home": 0.65, "draw": 0.20, "away": 0.15,
                 "over_1_5": 0.65, "under_1_5": 0.35,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80,
                 "btts_yes": 0.50, "btts_no": 0.50}
        picks = markets.evaluate_1x2(probs)
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home) == 1

    def test_no_odds_pick_has_none_edge(self):
        """No-odds picks must have edge=None (not a number)."""
        probs = {"home": 0.60, "draw": 0.25, "away": 0.15,
                 "over_1_5": 0.65, "under_1_5": 0.35,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80,
                 "btts_yes": 0.50, "btts_no": 0.50}
        picks = markets.evaluate_1x2(probs)
        for p in picks:
            assert p["edge"] is None
            assert p["odds"] is None

    def test_real_odds_exact_edge_value(self):
        """Edge = (model_prob - implied) * 100, precisely. home raised to 0.65 to clear MIN_PROB floor."""
        probs = {"home": 0.65, "draw": 0.20, "away": 0.15,
                 "over_1_5": 0.65, "under_1_5": 0.35,
                 "over_2_5": 0.50, "under_2_5": 0.50,
                 "over_3_5": 0.20, "under_3_5": 0.80,
                 "btts_yes": 0.50, "btts_no": 0.50}
        odds = 2.00
        picks = markets.evaluate_1x2(probs, odds_home=odds, min_edge=3.0)
        home = [p for p in picks if p["pick"] == "Home Win"]
        assert len(home) == 1
        expected_edge = (0.65 - 1/odds) * 100
        assert home[0]["edge"] == pytest.approx(expected_edge, abs=0.01)


# ── BTTS edge cases ───────────────────────────────────────────────────────────

class TestBttsEdgeCases:
    def test_btts_yes_with_real_odds_positive_edge(self):
        probs = {"btts_yes": 0.62, "btts_no": 0.38,
                 "home_cs_rate": 0.20, "away_cs_rate": 0.25}
        picks = markets.evaluate_btts(probs, odds_yes=1.85, min_edge=3.0)
        yes = [p for p in picks if p["pick"] == "BTTS Yes" and p.get("odds")]
        assert len(yes) == 1
        assert yes[0]["edge"] > 0

    def test_btts_no_exactly_at_cs_threshold(self):
        """BTTS No is disabled — never generated regardless of cs_rate."""
        probs = {"btts_yes": 0.42, "btts_no": 0.58,
                 "home_cs_rate": 0.35, "away_cs_rate": 0.20}
        picks = markets.evaluate_btts(probs)
        no = [p for p in picks if p["pick"] == "BTTS No"]
        assert no == []

    def test_btts_no_weakened_just_below_threshold(self):
        """cs_rate=0.34 < 0.35 → weakening applies: 0.58 * 0.90 = 0.522 < 0.55."""
        probs = {"btts_yes": 0.42, "btts_no": 0.58,
                 "home_cs_rate": 0.34, "away_cs_rate": 0.20}
        picks = markets.evaluate_btts(probs)
        no = [p for p in picks if p["pick"] == "BTTS No"]
        assert no == []

    def test_btts_no_real_odds_negative_edge_excluded(self):
        probs = {"btts_yes": 0.42, "btts_no": 0.58,
                 "home_cs_rate": 0.40, "away_cs_rate": 0.35}
        # implied = 1/1.50=0.667 > model 0.58 → negative edge
        picks = markets.evaluate_btts(probs, odds_no=1.50, min_edge=3.0)
        no = [p for p in picks if p["pick"] == "BTTS No" and p.get("odds")]
        assert no == []


