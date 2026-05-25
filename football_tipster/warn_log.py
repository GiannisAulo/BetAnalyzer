"""
warn_log.py — lightweight fallback warning logger.

Writes one line per fallback to warnings.log:
    date | league | match_id | reason | fallback_used

Import and call `fallback()` at each site where the model degrades silently.
Thread-safe (file-level append with no shared state).
"""
import datetime

_LOG_FILE = "warnings.log"


def fallback(reason: str, fallback_used: str, league: str = "", match_id: str = "") -> None:
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{date} | {league or '-'} | {match_id or '-'} | {reason} | {fallback_used}\n"
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
