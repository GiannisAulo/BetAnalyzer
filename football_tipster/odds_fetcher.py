"""
odds_fetcher.py — The Odds API v4 integration.

Fetches live bookmaker prices for today's fixtures and returns a normalised
odds dict that the market evaluators in markets.py can consume directly.

Free tier: 500 requests/month.  Each call to get_odds() costs 1 credit
(1 region × 1 market key).  We fetch h2h + totals in one call (2 credits).

Usage
-----
    from odds_fetcher import get_odds_for_league, OddsQuotaExhausted

    odds_map = get_odds_for_league("PL", date_str)
    # odds_map: { normalised_key: { "h2h": {...}, "totals": {...} }, ... }
"""

import time
import logging
from datetime import datetime
from difflib import SequenceMatcher
import requests
from cache import get_cached, set_cache, TTL_ODDS
from config import ODDS_API_KEY
import warn_log

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# How many credits remain this month (updated after each call)
_quota_remaining: int | None = None

# ---------------------------------------------------------------------------
# League → sport_key mapping
# ---------------------------------------------------------------------------

SPORT_KEY = {
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA":  "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "CL":  "soccer_uefa_champs_league",
    "PPL": "soccer_portugal_primeira_liga",
    "DED": "soccer_netherlands_eredivisie",
    "ELC": "soccer_efl_champ",
    "BSA": "soccer_brazil_campeonato",
}

# ---------------------------------------------------------------------------
# Team name normalisation
# football-data.org uses full official names; The Odds API uses common names.
# Add entries here whenever you see a mismatch in practice.
# ---------------------------------------------------------------------------

_NAME_MAP: dict[str, str] = {
    # Premier League
    "Nottingham Forest":          "Nottingham Forest",
    "Brighton & Hove Albion":     "Brighton and Hove Albion",
    "Wolverhampton Wanderers":    "Wolverhampton Wanderers",
    "West Ham United":            "West Ham United",
    "Tottenham Hotspur":          "Tottenham Hotspur",
    "Manchester United":          "Manchester United",
    "Manchester City":            "Manchester City",
    "Newcastle United":           "Newcastle United",
    "Luton Town":                 "Luton Town",
    # Bundesliga
    "Bayer 04 Leverkusen":        "Bayer Leverkusen",
    "Borussia Dortmund":          "Borussia Dortmund",
    "FC Bayern München":          "Bayern Munich",
    "RB Leipzig":                 "RB Leipzig",
    "Eintracht Frankfurt":        "Eintracht Frankfurt",
    "1. FC Köln":                 "FC Koln",
    "1. FSV Mainz 05":            "Mainz 05",
    "SC Freiburg":                "SC Freiburg",
    "VfB Stuttgart":              "VfB Stuttgart",
    "VfL Wolfsburg":              "Wolfsburg",
    "VfL Bochum 1848":            "VfL Bochum",
    "Borussia Mönchengladbach":   "Borussia Monchengladbach",
    "FC Augsburg":                "FC Augsburg",
    "TSG 1899 Hoffenheim":        "Hoffenheim",
    "SV Werder Bremen":           "Werder Bremen",
    "FC Heidenheim 1846":         "FC Heidenheim",
    # Serie A
    "FC Internazionale Milano":   "Inter Milan",
    "AC Milan":                   "AC Milan",
    "SS Lazio":                   "Lazio",
    "ACF Fiorentina":             "Fiorentina",
    "AS Roma":                    "AS Roma",
    "SSC Napoli":                 "Napoli",
    "Juventus":                   "Juventus",
    "Atalanta":                   "Atalanta",
    "Torino":                     "Torino",
    "Udinese Calcio":             "Udinese",
    "Hellas Verona":              "Hellas Verona",
    "Bologna":                    "Bologna",
    "Monza":                      "AC Monza",
    "Genoa CFC":                  "Genoa",
    # La Liga
    "Real Madrid CF":             "Real Madrid",
    "FC Barcelona":               "Barcelona",
    "Club Atlético de Madrid":    "Atletico Madrid",
    "Real Betis Balompié":        "Real Betis",
    "Real Sociedad de Fútbol":    "Real Sociedad",
    "Athletic Club":              "Athletic Bilbao",
    "Villarreal CF":              "Villarreal",
    "Sevilla FC":                 "Sevilla",
    "Valencia CF":                "Valencia",
    "Rayo Vallecano de Madrid":   "Rayo Vallecano",
    # Ligue 1
    "Paris Saint-Germain":        "Paris Saint Germain",
    "Olympique de Marseille":     "Marseille",
    "Olympique Lyonnais":         "Lyon",
    "Stade Rennais FC 1901":      "Rennes",
    "RC Lens":                    "Lens",
    "Lille OSC":                  "Lille",
    "OGC Nice":                   "Nice",
    "AS Monaco":                  "Monaco",
    # Eredivisie
    "AFC Ajax":                   "Ajax",
    "PSV Eindhoven":              "PSV Eindhoven",
    "Feyenoord":                  "Feyenoord",
    "AZ":                         "AZ Alkmaar",
    "FC Utrecht":                 "Utrecht",
    "FC Twente":                  "FC Twente",
    # Primeira Liga
    "SL Benfica":                 "Benfica",
    "FC Porto":                   "Porto",
    "Sporting CP":                "Sporting CP",
    "SC Braga":                   "Braga",
}


def _normalise(name: str) -> str:
    """Return the Odds API equivalent of a football-data.org team name."""
    return _NAME_MAP.get(name, name)


def _norm_key(home: str, away: str) -> str:
    """Canonical match key for dict lookup."""
    return f"{_normalise(home)}|{_normalise(away)}".lower()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OddsQuotaExhausted(Exception):
    """Raised when the monthly free-tier quota is used up."""


class OddsAPIError(Exception):
    """Raised for non-quota API errors."""


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _update_quota(headers: dict):
    global _quota_remaining
    try:
        _quota_remaining = int(headers.get("x-requests-remaining", _quota_remaining or 0))
    except (ValueError, TypeError):
        pass


def get_quota_remaining() -> int | None:
    """Return the last-known remaining monthly quota, or None if unknown."""
    return _quota_remaining


def _fetch_odds(sport_key: str, regions: str = "eu", markets: str = "h2h,totals") -> list:
    """
    Raw fetch from /v4/sports/{sport_key}/odds.
    Returns the decoded JSON list (one entry per event).
    Raises OddsQuotaExhausted when credits are gone.
    Raises OddsAPIError on other HTTP failures.
    Returns [] when ODDS_API_KEY is not configured.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     regions,
        "markets":     markets,
        "oddsFormat":  "decimal",
        "dateFormat":  "iso",
    }

    resp = requests.get(url, params=params, timeout=15)
    _update_quota(resp.headers)

    if resp.status_code == 401:
        raise OddsAPIError("Invalid ODDS_API_KEY — check your .env file.")
    if resp.status_code == 429:
        raise OddsQuotaExhausted("Monthly quota exhausted (429).")
    if resp.status_code != 200:
        raise OddsAPIError(f"Odds API returned {resp.status_code}: {resp.text[:200]}")

    time.sleep(0.5)   # polite pause — not rate-limited but good practice
    return resp.json()


# ---------------------------------------------------------------------------
# Price extraction helpers
# ---------------------------------------------------------------------------

def _best_h2h(bookmakers: list) -> dict:
    """
    Return the best (highest) decimal price per outcome across all bookmakers.
    Outcomes: home_team name, away_team name, "Draw".
    Returns {"home": float, "draw": float, "away": float} or {} if unavailable.
    """
    best: dict[str, float] = {}
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) != 3:
                continue
            for o in outcomes:
                # Skip outcomes with missing/malformed fields rather than
                # crashing the whole odds map on one bad bookmaker entry.
                try:
                    label = o["name"]
                    price = float(o["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if label not in best or price > best[label]:
                    best[label] = price
    return best


def _best_totals(bookmakers: list) -> dict:
    """
    Return best over/under prices for each line (2.5, 3.5).
    Returns {"over_2.5": float, "under_2.5": float, "over_3.5": float, "under_3.5": float}
    """
    best: dict[str, float] = {}
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "totals":
                continue
            for o in mkt.get("outcomes", []):
                # Defensive parsing: malformed outcomes from any single
                # bookmaker should be skipped, not crash the whole odds fetch.
                try:
                    direction = o["name"].lower()    # "over" or "under"
                    point     = float(o.get("point", 0))
                    price     = float(o["price"])
                except (KeyError, AttributeError, TypeError, ValueError):
                    continue
                key = f"{direction}_{point}"     # e.g. "over_2.5"
                if key not in best or price > best[key]:
                    best[key] = price
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_odds_for_league(league_code: str, use_cache: bool = True) -> dict:
    """
    Fetch today's odds for every fixture in `league_code`.

    Results are cached per league per calendar date for TTL_ODDS (24 h).
    Subsequent runs on the same day cost zero credits.

    Returns a dict keyed by normalised match key:
        { "home_name|away_name" (lowercase): {
              "home_odds":    float | None,
              "draw_odds":    float | None,
              "away_odds":    float | None,
              "over_1.5" ..., "under_3.5": float | None,
          }, ... }
    """
    sport_key = SPORT_KEY.get(league_code)
    if not sport_key:
        return {}

    # ── Cache check ──────────────────────────────────────────────────────
    today     = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"odds_{league_code}_{today}"

    if use_cache:
        cached = get_cached(cache_key, ttl=TTL_ODDS)
        if cached is not None:
            return cached

    # ── Live fetch ───────────────────────────────────────────────────────
    try:
        events = _fetch_odds(sport_key, regions="eu", markets="h2h,totals")
    except OddsQuotaExhausted:
        logger.warning("Odds API quota exhausted — running without real odds.")
        return {}
    except OddsAPIError as exc:
        logger.warning("Odds API error: %s", exc)
        return {}

    result = {}
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        if not home or not away:
            continue

        bookmakers = event.get("bookmakers", [])
        h2h_prices = _best_h2h(bookmakers)
        totals     = _best_totals(bookmakers)

        home_odds = draw_odds = away_odds = None
        for label, price in h2h_prices.items():
            if label == "Draw":
                draw_odds = price
            elif label.lower() == home.lower():
                home_odds = price
            elif label.lower() == away.lower():
                away_odds = price

        result[_norm_key(home, away)] = {
            "home_odds":  home_odds,
            "draw_odds":  draw_odds,
            "away_odds":  away_odds,
            "over_1.5":   totals.get("over_1.5"),
            "under_1.5":  totals.get("under_1.5"),
            "over_2.5":   totals.get("over_2.5"),
            "under_2.5":  totals.get("under_2.5"),
            "over_3.5":   totals.get("over_3.5"),
            "under_3.5":  totals.get("under_3.5"),
        }

    # ── Store in cache ───────────────────────────────────────────────────
    if use_cache and result:
        set_cache(cache_key, result)

    return result


def lookup_odds(odds_map: dict, home_name: str, away_name: str) -> dict:
    """
    Look up odds for a specific fixture from the odds_map returned by
    get_odds_for_league().  Tries the normalised key first, then a fuzzy
    fallback that checks if either name is a substring.

    Returns the odds dict, or a dict of all-None values if not found.
    """
    _empty = {
        "home_odds": None, "draw_odds": None, "away_odds": None,
        "over_1.5": None, "under_1.5": None,
        "over_2.5": None, "under_2.5": None, "over_3.5": None, "under_3.5": None,
    }

    # Exact normalised key
    key = _norm_key(home_name, away_name)
    if key in odds_map:
        return odds_map[key]

    # Fuzzy fallback: SequenceMatcher on normalised names, threshold 0.75.
    # Score = min(home_ratio, away_ratio) so both teams must match well.
    norm_home = _normalise(home_name).lower()
    norm_away = _normalise(away_name).lower()
    best_score, best_val = 0.0, None
    for k, v in odds_map.items():
        parts = k.split("|")
        if len(parts) != 2:
            continue
        api_home, api_away = parts
        r_home = SequenceMatcher(None, norm_home, api_home).ratio()
        r_away = SequenceMatcher(None, norm_away, api_away).ratio()
        score = min(r_home, r_away)
        if score > best_score:
            best_score, best_val = score, v

    if best_score >= 0.75 and best_val is not None:
        return best_val

    if odds_map:
        warn_log.fallback(
            f"no odds match for '{home_name}' vs '{away_name}' (best score {best_score:.2f})",
            "returning empty odds",
        )
    return _empty
