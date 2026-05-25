"""
test_staking.py — Tests for quarter-Kelly stake sizing.
"""
import pytest
import staking


class TestComputeStakeUnits:
    def test_no_odds_returns_none(self):
        assert staking.compute_stake_units(prob=0.70, odds=None) is None

    def test_zero_or_invalid_odds_returns_none(self):
        assert staking.compute_stake_units(prob=0.70, odds=1.0) is None
        assert staking.compute_stake_units(prob=0.70, odds=0.5) is None

    def test_invalid_prob_returns_none(self):
        assert staking.compute_stake_units(prob=0.0, odds=2.0) is None
        assert staking.compute_stake_units(prob=1.0, odds=2.0) is None
        assert staking.compute_stake_units(prob=None, odds=2.0) is None

    def test_negative_edge_returns_none(self):
        # 50% prob @ 1.50 odds → implied 66.7% → negative edge
        assert staking.compute_stake_units(prob=0.50, odds=1.50) is None

    def test_zero_edge_returns_none(self):
        # 50% prob @ 2.00 odds → exact zero edge
        assert staking.compute_stake_units(prob=0.50, odds=2.00) is None

    def test_explicit_negative_edge_param_returns_none(self):
        # Even if probabilities would suggest positive Kelly, caller-passed
        # negative edge short-circuits
        assert staking.compute_stake_units(prob=0.70, odds=2.00, edge=-1.0) is None

    def test_classic_kelly_example(self):
        # 70% prob @ 1.50 odds, full Kelly = (0.5 * 0.7 - 0.3) / 0.5 = 0.10 (10%).
        # Quarter Kelly = 2.5% → 3 units (round to nearest, 2.5 -> 2 banker's,
        # so accept 2 or 3).
        units = staking.compute_stake_units(prob=0.70, odds=1.50)
        assert units in (2, 3)

    def test_high_edge_capped(self):
        # 80% prob @ 3.00 odds, full Kelly = (2 * 0.8 - 0.2) / 2 = 0.70 (70%!).
        # Quarter Kelly = 17.5% → 18 units, capped to MAX_STAKE_UNITS.
        units = staking.compute_stake_units(prob=0.80, odds=3.00)
        assert units == staking.MAX_STAKE_UNITS

    def test_tiny_positive_edge_returns_none(self):
        # 50.5% prob @ 2.00 odds → Kelly fraction ~0.5%, quarter ~0.125%,
        # which rounds to 0 units. We return None rather than force-clamping
        # to 1u, since a 1u stake here would over-bet vs Kelly's recommendation.
        units = staking.compute_stake_units(prob=0.505, odds=2.00)
        assert units is None

    def test_max_boundary_respected(self):
        # Any positive recommendation must respect the upper cap; None is also valid.
        for prob, odds in [(0.55, 1.80), (0.60, 2.10), (0.65, 1.60), (0.75, 1.30)]:
            u = staking.compute_stake_units(prob=prob, odds=odds)
            if u is not None:
                assert 1 <= u <= staking.MAX_STAKE_UNITS

    def test_returns_integer(self):
        # Caller can pass floats safely; we always return int (or None)
        u = staking.compute_stake_units(prob=0.68, odds=1.70)
        assert u is None or isinstance(u, int)


class TestUnitsConversion:
    def test_units_to_eur_uses_default_bankroll(self):
        # 1 unit on default €10 bankroll = €0.10
        assert staking.units_to_eur(1) == pytest.approx(0.10)
        assert staking.units_to_eur(5) == pytest.approx(0.50)
        assert staking.units_to_eur(0) == 0

    def test_units_to_eur_none(self):
        assert staking.units_to_eur(None) is None

    def test_format_stake_with_units(self):
        s = staking.format_stake(3)
        assert "3u" in s and "0.30" in s

    def test_format_stake_none(self):
        assert staking.format_stake(None) == "—"


class TestKellyMathExact:
    """Verify the Kelly formula matches known textbook values."""

    def test_balanced_bet_no_stake(self):
        # 50% prob @ 2.0 odds → 0 edge → no bet
        assert staking.compute_stake_units(0.50, 2.0) is None

    def test_documented_growth_example(self):
        # 60% prob @ 2.0 odds (clean example):
        # b=1, p=0.6, q=0.4 → f* = (1*0.6 - 0.4)/1 = 0.20 → 20% bankroll full Kelly.
        # Quarter Kelly = 5% → 5 units exactly.
        assert staking.compute_stake_units(0.60, 2.0) == 5
