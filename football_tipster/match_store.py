"""
match_store.py — SQLite persistence for finished match data.

Every finished match seen by the fetcher is written here.  On the next run,
parse_team_history() merges the fresh API window (last 30) with the full
SQLite history, giving the model up to two full seasons of data instead of
half a season.

Schema
------
matches (
    match_id    INTEGER PRIMARY KEY,
    home_id     INTEGER NOT NULL,
    away_id     INTEGER NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    winner      TEXT,           -- HOME_TEAM | AWAY_TEAM | DRAW
    utc_date    TEXT NOT NULL,  -- ISO-8601 string from API
    league      TEXT,           -- competition code, e.g. "PL"
    stage       TEXT            -- e.g. "REGULAR_SEASON", "QUARTER_FINALS"
)
"""

import sqlite3
import os
import time
from pathlib import Path

_DB_DIR  = Path(__file__).parent / "data"
_DB_PATH = _DB_DIR / "matches.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY,
    home_id     INTEGER NOT NULL,
    away_id     INTEGER NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    winner      TEXT,
    utc_date    TEXT NOT NULL,
    league      TEXT,
    stage       TEXT,
    home_shots_on_target  INTEGER,
    away_shots_on_target  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_home ON matches(home_id);
CREATE INDEX IF NOT EXISTS idx_away ON matches(away_id);
CREATE INDEX IF NOT EXISTS idx_date ON matches(utc_date);
"""

# Migration: add shot stats columns to existing databases.
_MIGRATE_SHOTS_SQL = [
    "ALTER TABLE matches ADD COLUMN home_shots_on_target INTEGER",
    "ALTER TABLE matches ADD COLUMN away_shots_on_target INTEGER",
]

_UPSERT_SQL = """
INSERT OR IGNORE INTO matches
    (match_id, home_id, away_id, home_goals, away_goals, winner, utc_date, league, stage,
     home_shots_on_target, away_shots_on_target)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _connect(db_path=None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    os.makedirs(path.parent, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create the database and table if they don't exist yet.
    Also runs migrations and prunes matches older than MAX_HISTORY_DAYS."""
    conn = _connect(db_path)
    conn.executescript(_CREATE_SQL)
    # Migrate existing databases: add shot stats columns if missing.
    # OperationalError "duplicate column name" is expected after the first run;
    # any other error is a real schema problem and should propagate.
    for sql in _MIGRATE_SHOTS_SQL:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()
    conn.close()
    prune_old_matches(db_path)


def store_matches(matches_data: dict, league: str = "", db_path=None):
    """
    Persist finished matches from an API response dict into SQLite.
    Silently skips rows that already exist (INSERT OR IGNORE).
    Only stores matches with complete fullTime scores.

    matches_data: the dict returned by fetcher.get_team_matches() or similar.
    league:       competition code string, e.g. "PL".
    """
    if not matches_data or "matches" not in matches_data:
        return

    rows = []
    for m in matches_data["matches"]:
        if m.get("status") != "FINISHED":
            continue
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue

        match_id = m.get("id")
        if not match_id:
            continue

        home_id  = m.get("homeTeam", {}).get("id")
        away_id  = m.get("awayTeam", {}).get("id")
        winner   = m.get("score", {}).get("winner", "")
        utc_date = m.get("utcDate", "")
        stage    = m.get("stage", "")

        # Derive league from competition code if not provided
        comp_code = m.get("competition", {}).get("code", league)

        if not home_id or not away_id:
            continue

        # Extract shot stats from match statistics when available.
        # football-data.org v4 can return stats in several locations:
        #   1. Pre-extracted fields (from DB round-trip): m["home_shots_on_target"]
        #   2. Top-level statistics dict: m["statistics"]["home"]["shotsOnTarget"]
        #   3. Nested in team dicts: m["homeTeam"]["statistics"]["shotsOnTarget"]
        home_sot = m.get("home_shots_on_target")   # pre-extracted field
        away_sot = m.get("away_shots_on_target")   # pre-extracted field
        if home_sot is None or away_sot is None:
            stats = m.get("statistics", {})
            if isinstance(stats, dict):
                home_sot = home_sot if home_sot is not None else stats.get("home", {}).get("shotsOnTarget")
                away_sot = away_sot if away_sot is not None else stats.get("away", {}).get("shotsOnTarget")
        if home_sot is None or away_sot is None:
            ht_stats = m.get("homeTeam", {}).get("statistics", {})
            at_stats = m.get("awayTeam", {}).get("statistics", {})
            if isinstance(ht_stats, dict):
                home_sot = home_sot if home_sot is not None else ht_stats.get("shotsOnTarget")
            if isinstance(at_stats, dict):
                away_sot = away_sot if away_sot is not None else at_stats.get("shotsOnTarget")

        rows.append((match_id, home_id, away_id, hg, ag, winner, utc_date, comp_code, stage,
                      home_sot, away_sot))

    if not rows:
        return

    # Retry on database-locked errors (parallel league fetches share one DB file).
    for attempt in range(4):
        conn = None
        try:
            conn = _connect(db_path)
            conn.executemany(_UPSERT_SQL, rows)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc) and attempt < 3:
                time.sleep(0.15 * (2 ** attempt))
            else:
                raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


def get_team_match_history(team_id: int, limit: int = 76, db_path=None) -> list[dict]:
    """
    Return up to `limit` finished matches for team_id from SQLite, newest first.
    Each dict matches the structure expected by parse_team_history():
        {
            "id": match_id,
            "utcDate": str,
            "status": "FINISHED",
            "stage": str,
            "homeTeam": {"id": int},
            "awayTeam": {"id": int},
            "score": {
                "fullTime": {"home": int, "away": int},
                "winner": str,
            },
        }
    Returns [] if team_id has no stored matches.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT match_id, home_id, away_id, home_goals, away_goals,
                   winner, utc_date, league, stage,
                   home_shots_on_target, away_shots_on_target
            FROM matches
            WHERE home_id = ? OR away_id = ?
            ORDER BY utc_date DESC
            LIMIT ?
            """,
            (team_id, team_id, limit),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for r in rows:
        entry = {
            "id":       r["match_id"],
            "utcDate":  r["utc_date"],
            "status":   "FINISHED",
            "stage":    r["stage"] or "",
            "homeTeam": {"id": r["home_id"]},
            "awayTeam": {"id": r["away_id"]},
            "score": {
                "fullTime": {"home": r["home_goals"], "away": r["away_goals"]},
                "winner":   r["winner"] or "",
            },
        }
        # Attach shot stats when available (NULL in old rows → None)
        home_sot = r["home_shots_on_target"]
        away_sot = r["away_shots_on_target"]
        if home_sot is not None:
            entry["home_shots_on_target"] = home_sot
        if away_sot is not None:
            entry["away_shots_on_target"] = away_sot
        result.append(entry)
    return result


def merge_match_history(api_matches: list[dict], db_matches: list[dict]) -> list[dict]:
    """
    Merge fresh API matches with older SQLite matches.
    API matches take precedence (most recent); duplicates are deduplicated by match_id.
    Returns a combined list sorted newest-first.
    """
    seen = set()
    merged = []

    for m in api_matches:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            merged.append(m)

    for m in db_matches:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            merged.append(m)

    # Sort newest first by utcDate string (ISO-8601 sorts lexicographically)
    merged.sort(key=lambda m: m.get("utcDate", ""), reverse=True)
    return merged


def update_match_stats(match_id: int, home_sot: int, away_sot: int, db_path=None):
    """
    Backfill shot stats for an existing match row.
    Used when we fetch detailed stats from /matches/{id} after the match was
    initially stored without shot data.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE matches
            SET home_shots_on_target = ?, away_shots_on_target = ?
            WHERE match_id = ? AND home_shots_on_target IS NULL
            """,
            (home_sot, away_sot, match_id),
        )
        conn.commit()
    finally:
        conn.close()


def match_count(db_path=None) -> int:
    """Return the total number of stored matches (for diagnostics)."""
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finally:
        conn.close()


# Matches older than this are removed by prune_old_matches().
# Two seasons (730 days) covers all form/H2H data the model uses; older rows
# contribute essentially zero via recency decay and only slow queries.
MAX_HISTORY_DAYS = 730


def prune_old_matches(db_path=None) -> int:
    """
    Delete matches older than MAX_HISTORY_DAYS from matches.db.
    Returns the number of rows deleted.
    Called from init_db() so pruning happens once at startup.
    """
    conn = _connect(db_path)
    try:
        # Use SQLite date arithmetic — utc_date is an ISO-8601 string (sorts correctly).
        cutoff = f"datetime('now', '-{MAX_HISTORY_DAYS} days')"
        deleted = conn.execute(
            f"DELETE FROM matches WHERE utc_date < {cutoff}"
        ).rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()
