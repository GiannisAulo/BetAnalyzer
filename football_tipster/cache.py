import json
import os
import time

CACHE_DIR = ".cache"

# Default TTLs per data type (seconds)
TTL_FIXTURES    = 2  * 3600   # 2 h  — fixture list for the day
TTL_STANDINGS   = 12 * 3600   # 12 h — standings change slowly
TTL_TEAM        = 24 * 3600   # 24 h — recent match history
TTL_H2H         = 24 * 3600   # 24 h — H2H history is static within a day
TTL_ODDS        = 24 * 3600   # 24 h — one fetch per day is enough


def get_cached(key, ttl=TTL_STANDINGS):
    """Return cached data if it exists and is younger than ttl seconds."""
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl:
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt or unreadable cache file — discard and fall through to live fetch
                try:
                    os.remove(path)
                except OSError:
                    pass
    return None


def set_cache(key, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass  # disk full or permission error — cache miss on next run is fine


def get_cache_age_hours(key: str) -> float | None:
    """Return age in hours of a cache entry, or None if it doesn't exist."""
    path = os.path.join(CACHE_DIR, key + ".json")
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600


def evict_stale_cache(max_age_seconds: int = 48 * 3600) -> int:
    """
    Delete cache files older than max_age_seconds (default 48 h = 2× longest TTL).
    Returns the number of files deleted.
    Call from startup to prevent indefinite accumulation of dated cache files.
    """
    if not os.path.exists(CACHE_DIR):
        return 0
    deleted = 0
    now = time.time()
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            if now - os.path.getmtime(path) > max_age_seconds:
                os.remove(path)
                deleted += 1
        except OSError:
            pass
    return deleted
