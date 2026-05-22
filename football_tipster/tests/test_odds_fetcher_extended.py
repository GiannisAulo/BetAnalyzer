"""
test_odds_fetcher_extended.py — additional edge case tests for odds_fetcher.py.
Covers: over/under 1.5 keys, partial bookmakers, single bookmaker, tie-breaking,
lookup fuzzy matching edge cases, get_quota_remaining, sport key completeness.
"""
import pytest
import odds_fetcher
from odds_fetcher import (
    _normalise, _norm_key, lookup_odds,
    _best_h2h, _best_totals,
    get_odds_for_league,
    OddsQuotaExhausted, OddsAPIError,
)


# ── Over/Under 1.5 keys present ──────────────────────────────────────────────

class TestOver15Keys:
    """get_odds_for_league must include over_1.5 and under_1.5 in its output."""

    _MOCK_EVENTS = [
        {
            "id": "xyz999",
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-04-20T15:00:00Z",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.90},
                                {"name": "Draw",    "price": 3.50},
                                {"name": "Chelsea", "price": 4.00},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over",  "price": 1.45, "point": 1.5},
                                {"name": "Under", "price": 2.70, "point": 1.5},
                                {"name": "Over",  "price": 1.80, "point": 2.5},
                                {"name": "Under", "price": 2.00, "point": 2.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    def test_over_15_key_present(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self._MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        result = get_odds_for_league("PL", use_cache=False)
        entry = result.get("arsenal|chelsea", {})
        assert "over_1.5" in entry

    def test_under_15_key_present(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self._MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        result = get_odds_for_league("PL", use_cache=False)
        entry = result.get("arsenal|chelsea", {})
        assert "under_1.5" in entry

    def test_over_15_value_correct(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self._MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        result = get_odds_for_league("PL", use_cache=False)
        assert result["arsenal|chelsea"]["over_1.5"] == pytest.approx(1.45)

    def test_under_15_value_correct(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self._MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        result = get_odds_for_league("PL", use_cache=False)
        assert result["arsenal|chelsea"]["under_1.5"] == pytest.approx(2.70)


# ── _best_totals edge cases ───────────────────────────────────────────────────

class TestBestTotalsEdgeCases:
    def test_single_bookmaker_returns_its_odds(self):
        bms = [{"key": "bm1", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over",  "price": 1.85, "point": 2.5},
            {"name": "Under", "price": 1.95, "point": 2.5},
        ]}]}]
        result = _best_totals(bms)
        assert result["over_2.5"] == pytest.approx(1.85)
        assert result["under_2.5"] == pytest.approx(1.95)

    def test_best_of_two_bookmakers_picked(self):
        bms = [
            {"key": "bm1", "markets": [{"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.80, "point": 2.5},
            ]}]},
            {"key": "bm2", "markets": [{"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.92, "point": 2.5},
            ]}]},
        ]
        result = _best_totals(bms)
        assert result["over_2.5"] == pytest.approx(1.92)

    def test_missing_totals_market_returns_empty(self):
        bms = [{"key": "bm1", "markets": [{"key": "h2h", "outcomes": []}]}]
        assert _best_totals(bms) == {}

    def test_over_15_and_under_15_extracted(self):
        bms = [{"key": "bm1", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over",  "price": 1.40, "point": 1.5},
            {"name": "Under", "price": 2.90, "point": 1.5},
        ]}]}]
        result = _best_totals(bms)
        assert "over_1.5" in result
        assert "under_1.5" in result
        assert result["over_1.5"] == pytest.approx(1.40)
        assert result["under_1.5"] == pytest.approx(2.90)

    def test_outcome_missing_point_defaults_to_zero(self):
        """Totals outcome without a 'point' field defaults to point=0.0."""
        bms = [{"key": "bm1", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": 1.80},   # no "point" key
        ]}]}]
        result = _best_totals(bms)
        # point defaults to 0 → key "over_0.0" is created (not filtered)
        assert "over_0.0" in result
        assert result["over_0.0"] == pytest.approx(1.80)


# ── _best_h2h edge cases ──────────────────────────────────────────────────────

class TestBestH2HEdgeCases:
    def test_single_bookmaker_three_outcomes(self):
        """_best_h2h requires exactly 3 outcomes per market (home/draw/away)."""
        bms = [{"key": "bm1", "markets": [{"key": "h2h", "outcomes": [
            {"name": "TeamA", "price": 2.10},
            {"name": "Draw",  "price": 3.20},
            {"name": "TeamB", "price": 3.50},
        ]}]}]
        result = _best_h2h(bms)
        assert result["TeamA"] == pytest.approx(2.10)

    def test_single_outcome_skipped(self):
        """A market with only 1 outcome (not a valid 3-way h2h) is skipped."""
        bms = [{"key": "bm1", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 1.90},
        ]}]}]
        result = _best_h2h(bms)
        assert result == {}

    def test_tie_between_bookmakers_returns_either(self):
        """When two bookmakers offer identical odds for 3-outcome market."""
        bms = [
            {"key": "bm1", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 1.90},
                {"name": "Draw",    "price": 3.50},
                {"name": "Chelsea", "price": 4.00},
            ]}]},
            {"key": "bm2", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 1.90},
                {"name": "Draw",    "price": 3.50},
                {"name": "Chelsea", "price": 4.00},
            ]}]},
        ]
        result = _best_h2h(bms)
        assert result["Arsenal"] == pytest.approx(1.90)

    def test_multiple_outcomes_all_picked(self):
        bms = [{"key": "bm1", "markets": [{"key": "h2h", "outcomes": [
            {"name": "TeamA", "price": 1.80},
            {"name": "Draw",  "price": 3.20},
            {"name": "TeamB", "price": 4.50},
        ]}]}]
        result = _best_h2h(bms)
        assert set(result.keys()) == {"TeamA", "Draw", "TeamB"}


# ── lookup_odds edge cases ────────────────────────────────────────────────────

class TestLookupOddsEdgeCases:
    @pytest.fixture
    def sample_map(self):
        return {
            "arsenal|chelsea": {
                "home_odds": 1.95, "draw_odds": 3.50, "away_odds": 4.00,
                "over_1.5": 1.42, "under_1.5": 2.80,
                "over_2.5": 1.80, "under_2.5": 2.00,
                "over_3.5": 3.20, "under_3.5": 1.35,
            }
        }

    def test_over_15_returned_in_lookup(self, sample_map):
        result = lookup_odds(sample_map, "Arsenal", "Chelsea")
        assert result.get("over_1.5") == pytest.approx(1.42)

    def test_under_15_returned_in_lookup(self, sample_map):
        result = lookup_odds(sample_map, "Arsenal", "Chelsea")
        assert result.get("under_1.5") == pytest.approx(2.80)

    def test_no_match_returns_all_none(self, sample_map):
        result = lookup_odds(sample_map, "Real Madrid CF", "Atletico Madrid")
        for key in ("home_odds", "draw_odds", "away_odds",
                    "over_1.5", "under_1.5", "over_2.5", "under_2.5"):
            assert result[key] is None

    def test_empty_map_returns_none_dict(self):
        result = lookup_odds({}, "Arsenal", "Chelsea")
        assert result["home_odds"] is None

    def test_case_insensitive_lookup(self, sample_map):
        result = lookup_odds(sample_map, "ARSENAL", "CHELSEA")
        assert result["home_odds"] == pytest.approx(1.95)

    def test_partial_name_fuzzy_match(self, sample_map):
        """'Arsenal FC' should SequenceMatcher-fuzzy-match 'arsenal' (ratio ~0.82)."""
        result = lookup_odds(sample_map, "Arsenal FC", "Chelsea FC")
        assert result["home_odds"] == pytest.approx(1.95)


# ── get_quota_remaining ───────────────────────────────────────────────────────

class TestGetQuotaRemaining:
    def test_returns_none_before_any_call(self, monkeypatch):
        monkeypatch.setattr(odds_fetcher, "_quota_remaining", None)
        assert odds_fetcher.get_quota_remaining() is None

    def test_updated_after_successful_fetch(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.setattr(odds_fetcher, "_quota_remaining", None)
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=[],
            headers={"x-requests-remaining": "123"},
        )
        get_odds_for_league("PL", use_cache=False)
        assert odds_fetcher.get_quota_remaining() == 123


# ── OddsAPIError / OddsQuotaExhausted exceptions ─────────────────────────────

class TestOddsExceptions:
    def test_quota_exhausted_is_exception(self):
        with pytest.raises(OddsQuotaExhausted):
            raise OddsQuotaExhausted("quota gone")

    def test_api_error_is_exception(self):
        with pytest.raises(OddsAPIError):
            raise OddsAPIError("server error")

    def test_quota_and_api_error_are_independent(self):
        """OddsQuotaExhausted and OddsAPIError are separate exception classes."""
        assert not issubclass(OddsQuotaExhausted, OddsAPIError)
        assert not issubclass(OddsAPIError, OddsQuotaExhausted)


# ── _normalise edge cases ─────────────────────────────────────────────────────

class TestNormaliseEdgeCases:
    def test_empty_string_returned_as_is(self):
        assert _normalise("") == ""

    def test_whitespace_returned_as_is(self):
        assert _normalise("   ") == "   "

    def test_known_mapping_case_insensitive(self):
        """Mappings are exact — different casing returns as-is (no case folding)."""
        result = _normalise("fc barcelona")
        # Lower-case variant not in map → returned unchanged
        assert result == "fc barcelona"

    def test_norm_key_swapped_order_differs(self):
        """Key order matters: home|away != away|home."""
        k1 = _norm_key("Arsenal", "Chelsea")
        k2 = _norm_key("Chelsea", "Arsenal")
        assert k1 != k2

    def test_norm_key_with_mapped_names(self):
        key = _norm_key("FC Bayern München", "Borussia Dortmund")
        assert key == "bayern munich|borussia dortmund"
