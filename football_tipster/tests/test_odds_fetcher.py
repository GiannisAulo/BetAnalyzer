"""
test_odds_fetcher.py — unit tests for odds_fetcher.py.
All HTTP calls are mocked — no real API key required.
"""
import pytest
import odds_fetcher
from odds_fetcher import (
    _normalise, _norm_key, lookup_odds,
    _best_h2h, _best_totals,
    get_odds_for_league,
    OddsQuotaExhausted, OddsAPIError,
)


# ── Name normalisation ────────────────────────────────────────────────────────

class TestNormalise:
    def test_known_name_mapped(self):
        assert _normalise("FC Bayern München") == "Bayern Munich"

    def test_unknown_name_returned_as_is(self):
        assert _normalise("Unknown United FC") == "Unknown United FC"

    def test_norm_key_lowercase(self):
        key = _norm_key("FC Bayern München", "Borussia Dortmund")
        assert key == "bayern munich|borussia dortmund"

    def test_norm_key_unknown_names(self):
        key = _norm_key("Team A", "Team B")
        assert key == "team a|team b"


# ── Price extraction helpers ──────────────────────────────────────────────────

_SAMPLE_BOOKMAKERS = [
    {
        "key": "bet365",
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal",  "price": 1.90},
                    {"name": "Draw",     "price": 3.50},
                    {"name": "Chelsea",  "price": 4.00},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over",  "price": 1.80, "point": 2.5},
                    {"name": "Under", "price": 2.00, "point": 2.5},
                    {"name": "Over",  "price": 3.20, "point": 3.5},
                    {"name": "Under", "price": 1.35, "point": 3.5},
                ],
            },
        ],
    },
    {
        "key": "skybet",
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal",  "price": 1.95},   # higher — should win
                    {"name": "Draw",     "price": 3.40},
                    {"name": "Chelsea",  "price": 3.90},
                ],
            },
        ],
    },
]


class TestBestH2H:
    def test_picks_highest_home_odds(self):
        result = _best_h2h(_SAMPLE_BOOKMAKERS)
        assert result["Arsenal"] == pytest.approx(1.95)

    def test_picks_highest_draw_odds(self):
        result = _best_h2h(_SAMPLE_BOOKMAKERS)
        assert result["Draw"] == pytest.approx(3.50)

    def test_picks_highest_away_odds(self):
        result = _best_h2h(_SAMPLE_BOOKMAKERS)
        assert result["Chelsea"] == pytest.approx(4.00)

    def test_empty_bookmakers(self):
        assert _best_h2h([]) == {}

    def test_missing_h2h_market(self):
        bms = [{"key": "bm", "markets": [{"key": "totals", "outcomes": []}]}]
        assert _best_h2h(bms) == {}


class TestBestTotals:
    def test_over_25(self):
        result = _best_totals(_SAMPLE_BOOKMAKERS)
        assert result["over_2.5"] == pytest.approx(1.80)

    def test_under_25(self):
        result = _best_totals(_SAMPLE_BOOKMAKERS)
        assert result["under_2.5"] == pytest.approx(2.00)

    def test_over_35(self):
        result = _best_totals(_SAMPLE_BOOKMAKERS)
        assert result["over_3.5"] == pytest.approx(3.20)

    def test_under_35(self):
        result = _best_totals(_SAMPLE_BOOKMAKERS)
        assert result["under_3.5"] == pytest.approx(1.35)

    def test_empty_bookmakers(self):
        assert _best_totals([]) == {}


# ── lookup_odds ───────────────────────────────────────────────────────────────

class TestLookupOdds:
    @pytest.fixture
    def sample_map(self):
        return {
            "arsenal|chelsea": {
                "home_odds": 1.95, "draw_odds": 3.50, "away_odds": 4.00,
                "over_2.5": 1.80, "under_2.5": 2.00,
                "over_3.5": 3.20, "under_3.5": 1.35,
            }
        }

    def test_exact_match(self, sample_map):
        result = lookup_odds(sample_map, "Arsenal", "Chelsea")
        assert result["home_odds"] == pytest.approx(1.95)

    def test_mapped_name_match(self, sample_map):
        # "FC Bayern München" should map to "Bayern Munich" then fuzzy-match
        bm = {"bayern munich|borussia dortmund": {
            "home_odds": 1.50, "draw_odds": 4.00, "away_odds": 6.00,
            "over_2.5": 1.70, "under_2.5": 2.10, "over_3.5": 2.80, "under_3.5": 1.40,
        }}
        result = lookup_odds(bm, "FC Bayern München", "Borussia Dortmund")
        assert result["home_odds"] == pytest.approx(1.50)

    def test_missing_match_returns_none_dict(self, sample_map):
        result = lookup_odds(sample_map, "Real Madrid CF", "FC Barcelona")
        assert result["home_odds"] is None
        assert result["draw_odds"] is None
        assert result["away_odds"] is None

    def test_fuzzy_substring_match(self, sample_map):
        # "Arsenal FC" should fuzzy-match "arsenal"
        result = lookup_odds(sample_map, "Arsenal FC", "Chelsea FC")
        assert result["home_odds"] == pytest.approx(1.95)


# ── get_odds_for_league — mocked HTTP ────────────────────────────────────────

_MOCK_EVENTS = [
    {
        "id": "abc123",
        "sport_key": "soccer_epl",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-04-13T15:00:00Z",
        "bookmakers": _SAMPLE_BOOKMAKERS,
    }
]


class TestGetOddsForLeague:
    def test_returns_empty_without_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "")
        monkeypatch.chdir(tmp_path)
        result = get_odds_for_league("PL", use_cache=False)
        assert result == {}

    def test_unknown_league_returns_empty(self, monkeypatch):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        result = get_odds_for_league("UNKOWN")
        assert result == {}

    def test_successful_fetch(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)   # isolate from real cache
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=_MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        result = get_odds_for_league("PL")
        assert "arsenal|chelsea" in result
        entry = result["arsenal|chelsea"]
        assert entry["home_odds"] == pytest.approx(1.95)
        assert entry["draw_odds"] == pytest.approx(3.50)
        assert entry["over_2.5"] == pytest.approx(1.80)

    def test_quota_exhausted_returns_empty(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)   # fresh cache dir — no stale hits
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            status_code=429,
        )
        result = get_odds_for_league("PL", use_cache=True)
        assert result == {}

    def test_api_error_returns_empty(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)   # fresh cache dir — no stale hits
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            status_code=500,
            text="Internal Server Error",
        )
        result = get_odds_for_league("PL", use_cache=True)
        assert result == {}

    def test_quota_updated_after_call(self, monkeypatch, requests_mock, tmp_path):
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.setattr(odds_fetcher, "_quota_remaining", None)
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=_MOCK_EVENTS,
            headers={"x-requests-remaining": "487"},
        )
        get_odds_for_league("PL")
        assert odds_fetcher.get_quota_remaining() == 487

    def test_result_cached_after_fetch(self, monkeypatch, requests_mock, tmp_path):
        """Second call should hit cache and NOT make another HTTP request."""
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=_MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        first  = get_odds_for_league("PL", use_cache=True)
        second = get_odds_for_league("PL", use_cache=True)
        # Only one real HTTP call should have been made
        assert requests_mock.call_count == 1
        assert first == second

    def test_no_cache_always_fetches(self, monkeypatch, requests_mock, tmp_path):
        """use_cache=False should hit the API every time."""
        monkeypatch.setattr(odds_fetcher, "ODDS_API_KEY", "testkey")
        monkeypatch.chdir(tmp_path)
        requests_mock.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=_MOCK_EVENTS,
            headers={"x-requests-remaining": "490"},
        )
        get_odds_for_league("PL", use_cache=False)
        get_odds_for_league("PL", use_cache=False)
        assert requests_mock.call_count == 2

    def test_all_sport_keys_mapped(self):
        for code in ["PL", "PD", "BL1", "SA", "FL1", "CL", "PPL", "DED", "ELC", "BSA"]:
            assert code in odds_fetcher.SPORT_KEY, f"{code} missing from SPORT_KEY"
