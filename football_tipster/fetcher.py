import time
import requests
from config import API_KEY, BASE_URL, RATE_LIMIT_SLEEP, CACHE_VERSION
from cache import get_cached, set_cache, TTL_FIXTURES, TTL_STANDINGS, TTL_TEAM, TTL_H2H
import match_store
import warn_log

HEADERS = {"X-Auth-Token": API_KEY}

# Counts live API calls made in the current run (cache hits don't increment).
# Logged at end of _run_analysis() to detect quota pressure.
_api_call_count: int = 0
_API_CALL_WARN_THRESHOLD = 80   # football-data.org free tier: 10 req/min, ~100/day


def get_api_call_count() -> int:
    return _api_call_count


def reset_api_call_count() -> None:
    global _api_call_count
    _api_call_count = 0


def _get(endpoint, params=None, cache_key=None, use_cache=True, ttl=TTL_STANDINGS):
    global _api_call_count

    if use_cache and cache_key:
        versioned_key = f"v{CACHE_VERSION}_{cache_key}"
        cached = get_cached(versioned_key, ttl=ttl)
        if cached is not None:
            return cached

    _api_call_count += 1
    url = BASE_URL + endpoint
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if use_cache and cache_key:
        set_cache(versioned_key, data)

    time.sleep(RATE_LIMIT_SLEEP)
    return data


def get_fixtures(league_code, date_str, use_cache=True):
    """Get all matches for a league on a specific date (YYYY-MM-DD)."""
    return _get(
        f"/competitions/{league_code}/matches",
        params={"dateFrom": date_str, "dateTo": date_str},
        cache_key=f"fixtures_{league_code}_{date_str}",
        use_cache=use_cache,
        ttl=TTL_FIXTURES,
    )


def get_standings(league_code, use_cache=True):
    """Get current standings for a league."""
    return _get(
        f"/competitions/{league_code}/standings",
        cache_key=f"standings_{league_code}",
        use_cache=use_cache,
        ttl=TTL_STANDINGS,
    )


def get_team_matches(team_id, use_cache=True, league=""):
    """
    Get last 30 finished matches for a team (full half-season rolling window).
    E.3: Results are persisted to SQLite so that across multiple runs the model
    accumulates up to two full seasons of history.
    """
    data = _get(
        f"/teams/{team_id}/matches",
        params={"status": "FINISHED", "limit": 30},
        cache_key=f"team_matches_{team_id}",
        use_cache=use_cache,
        ttl=TTL_TEAM,
    )
    # Persist to SQLite (no-op on cache hits since we only write fresh data when
    # the API was actually called — store_matches deduplicates via INSERT OR IGNORE).
    # Persistence is best-effort: we never break the main pipeline if SQLite is
    # locked or the schema is in flux, but we DO log so silent corruption can be
    # diagnosed from warnings.log later.
    try:
        match_store.store_matches(data, league=league)
    except Exception as exc:
        warn_log.fallback(
            f"store_matches failed: {exc.__class__.__name__}: {exc}",
            "match history not persisted for this team",
            league=league, match_id=str(team_id),
        )
    return data


def get_head2head(match_id, use_cache=True):
    """Get head-to-head history for a match (last 10 meetings)."""
    return _get(
        f"/matches/{match_id}/head2head",
        params={"limit": 10},
        cache_key=f"h2h_{match_id}",
        use_cache=use_cache,
        ttl=TTL_H2H,
    )


def get_last_match_date(team_id, use_cache=True):
    """
    Return the utcDate string of the most recently finished match for a team,
    or None if unavailable.  Uses a 1-match fetch (minimal API cost).
    """
    try:
        data = _get(
            f"/teams/{team_id}/matches",
            params={"status": "FINISHED", "limit": 1},
            cache_key=f"last_match_{team_id}",
            use_cache=use_cache,
            ttl=TTL_TEAM,
        )
        matches = data.get("matches", [])
        if matches:
            return matches[-1].get("utcDate")
    except Exception as exc:
        # Surface the failure rather than silently returning None — fatigue
        # detection silently disables when this happens; the warning lets us
        # see how often it's firing.
        warn_log.fallback(
            f"get_last_match_date failed: {exc.__class__.__name__}: {exc}",
            "fatigue check disabled for team",
            match_id=str(team_id),
        )
    return None


def get_season_matches(league_code, season, use_cache=True):
    """
    Get all matches for a league in a given season year (e.g. 2023 for 2023/24).
    Used by backtest.py to replay full seasons.
    """
    return _get(
        f"/competitions/{league_code}/matches",
        params={"season": season},
        cache_key=f"season_{league_code}_{season}",
        use_cache=use_cache,
        ttl=TTL_STANDINGS,
    )


def get_match(match_id):
    """Fetch a single match by ID (never cached — used for result settlement)."""
    return _get(
        f"/matches/{match_id}",
        cache_key=None,
        use_cache=False,
    )


