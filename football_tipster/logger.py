import csv
import os
import time
from datetime import datetime
import warn_log

LOG_FILE = "bets_log.csv"
# Increment this string whenever thresholds or model logic change significantly.
# Logged with every new pick so analysis can filter by model version.
MODEL_VERSION = "v2026-04-27"

FIELDS = [
    "match_id", "date", "home", "away", "league",
    "market", "pick", "model_prob", "odds_taken", "edge", "result", "roi",
    "settle_attempts",
    "home_position", "away_position", "form_adv", "expected_total",
    "model_version",
]
# odds_taken:      bookmaker decimal odds at pick time (blank when no odds available)
# roi:             realised return per unit staked = (odds_taken - 1) on W, -1 on L
# home_position:   home team league position normalised 0–1 (1 = top, 0 = bottom)
# away_position:   away team league position normalised 0–1
# form_adv:        home form score minus away form score (range ~-1 to +1)
# expected_total:  model expected total goals (xG home + xG away)
_MAX_SETTLE_ATTEMPTS = 10

# API statuses that mean the match is fully over
_FINISHED_STATUSES = {"FINISHED"}


def _load_rows():
    """Return all rows as a list of dicts, or [] if the file doesn't exist."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_logged_ids():
    """Return the set of (match_id, market, pick) tuples already in the log."""
    result = set()
    for r in _load_rows():
        mid = r.get("match_id", "").strip()
        if mid:
            result.add((mid, r.get("market", "").strip(), r.get("pick", "").strip()))
    return result


def _evaluate_result(pick_name, market, match_data):
    """
    Given the pick label and a finished match API response, return "W" or "L".
    Returns None if the match isn't finished yet or data is missing.
    """
    status = match_data.get("status", "")
    if status not in _FINISHED_STATUSES:
        return None

    score = match_data.get("score", {})
    full_time = score.get("fullTime", {})
    home_goals = full_time.get("home")
    away_goals = full_time.get("away")

    if home_goals is None or away_goals is None:
        return None

    total_goals = home_goals + away_goals
    winner = score.get("winner", "")   # HOME_TEAM | AWAY_TEAM | DRAW

    if market == "1X2":
        if pick_name == "Home Win":
            return "W" if winner == "HOME_TEAM" else "L"
        if pick_name == "Away Win":
            return "W" if winner == "AWAY_TEAM" else "L"
        if pick_name == "Draw":
            return "W" if winner == "DRAW" else "L"

    if market == "Double Chance":
        if "Home or Draw" in pick_name:   # 1X
            return "W" if winner in ("HOME_TEAM", "DRAW") else "L"
        if "Draw or Away" in pick_name:   # X2
            return "W" if winner in ("AWAY_TEAM", "DRAW") else "L"
        if "Home or Away" in pick_name:   # 12
            return "W" if winner in ("HOME_TEAM", "AWAY_TEAM") else "L"

    if market == "Over/Under":
        if pick_name == "Over 1.5":
            return "W" if total_goals > 1 else "L"
        if pick_name == "Under 1.5":
            return "W" if total_goals < 2 else "L"
        if pick_name == "Over 2.5":
            return "W" if total_goals > 2 else "L"
        if pick_name == "Under 2.5":
            return "W" if total_goals < 3 else "L"
        if pick_name == "Over 3.5":
            return "W" if total_goals > 3 else "L"
        if pick_name == "Under 3.5":
            return "W" if total_goals < 4 else "L"

    if market == "BTTS":
        both_scored = home_goals > 0 and away_goals > 0
        if pick_name == "BTTS Yes":
            return "W" if both_scored else "L"
        if pick_name == "BTTS No":
            return "W" if not both_scored else "L"

    if market == "Combo":
        if " + " not in pick_name:
            return None
        leg1, leg2 = pick_name.split(" + ", 1)
        r1 = _evaluate_leg(leg1, home_goals, away_goals, winner, total_goals)
        r2 = _evaluate_leg(leg2, home_goals, away_goals, winner, total_goals)
        if r1 is None or r2 is None:
            # Unrecognised leg — can't evaluate. Caller handles "ERR".
            return "ERR"
        return "W" if r1 == "W" and r2 == "W" else "L"

    return None   # unrecognised market/pick — leave unsettled


def _evaluate_leg(leg_name, home_goals, away_goals, winner, total_goals):
    """Evaluate a single combo leg. Returns 'W', 'L', or None."""
    if leg_name in ("Home Win", "Away Win", "Draw"):
        if leg_name == "Home Win":
            return "W" if winner == "HOME_TEAM" else "L"
        if leg_name == "Away Win":
            return "W" if winner == "AWAY_TEAM" else "L"
        if leg_name == "Draw":
            return "W" if winner == "DRAW" else "L"

    if leg_name in ("Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5"):
        if leg_name == "Over 1.5":
            return "W" if total_goals > 1 else "L"
        if leg_name == "Under 1.5":
            return "W" if total_goals < 2 else "L"
        if leg_name == "Over 2.5":
            return "W" if total_goals > 2 else "L"
        if leg_name == "Under 2.5":
            return "W" if total_goals < 3 else "L"
        if leg_name == "Over 3.5":
            return "W" if total_goals > 3 else "L"
        if leg_name == "Under 3.5":
            return "W" if total_goals < 4 else "L"

    if leg_name in ("BTTS Yes", "BTTS No"):
        both_scored = home_goals > 0 and away_goals > 0
        if leg_name == "BTTS Yes":
            return "W" if both_scored else "L"
        if leg_name == "BTTS No":
            return "W" if not both_scored else "L"

    if "Home or Draw" in leg_name:   # 1X
        return "W" if winner in ("HOME_TEAM", "DRAW") else "L"
    if "Draw or Away" in leg_name:   # X2
        return "W" if winner in ("AWAY_TEAM", "DRAW") else "L"
    if "Home or Away" in leg_name:   # 12
        return "W" if winner in ("HOME_TEAM", "AWAY_TEAM") else "L"

    return None


def settle_bets(console=None):
    """
    Re-fetch every unsettled row (result == ""), evaluate the outcome, and
    rewrite the file with results filled in.  Called once at startup.
    Returns (settled, failed) counts.
    """
    import fetcher       # local import to avoid circular dependency at module load

    rows = _load_rows()
    unsettled = [r for r in rows if not r.get("result", "").strip()]
    if not unsettled:
        return 0, 0

    settled = failed = 0

    for row in unsettled:
        mid = row.get("match_id", "").strip()
        if not mid:
            continue

        attempts = int(row.get("settle_attempts") or 0)
        if attempts >= _MAX_SETTLE_ATTEMPTS:
            if console:
                console.print(f"  [dim]Skipping match {mid} — {attempts} failed attempts[/dim]")
            failed += 1
            continue

        try:
            data = fetcher.get_match(mid)
            match = data.get("match") or data   # API returns {"match": {...}} or the match directly
            outcome = _evaluate_result(row.get("pick", ""), row.get("market", ""), match)
            if outcome == "ERR":
                warn_log.fallback(
                    "unrecognised combo leg — cannot evaluate outcome",
                    "settled as ERR",
                    league=row.get("league", ""),
                    match_id=mid,
                )
                row["result"] = "ERR"
                row["roi"] = ""
                settled += 1
            elif outcome:
                row["result"] = outcome
                # Compute ROI: (odds - 1) on win, -1 on loss
                try:
                    odds_taken = float(row.get("odds_taken") or 0)
                    if odds_taken > 1:
                        row["roi"] = f"{(odds_taken - 1):.3f}" if outcome == "W" else "-1.000"
                    else:
                        row["roi"] = ""   # no odds logged — can't compute ROI
                except (ValueError, TypeError):
                    row["roi"] = ""

                settled += 1
            else:
                row["settle_attempts"] = attempts + 1
                failed += 1
            time.sleep(0.15)   # stay well within 10 req/min free-tier limit
        except Exception:
            row["settle_attempts"] = attempts + 1
            failed += 1

    if settled or failed:
        _write_rows(rows)

    return settled, failed



def _write_rows(rows):
    """Overwrite the log file with the given rows (preserving column order)."""
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _has_header():
    """Return True if the log file exists and its first line is the expected header."""
    if not os.path.exists(LOG_FILE):
        return False
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    return first == ",".join(FIELDS)


def compute_roi_summary():
    """
    Return a dict with cumulative ROI stats from all settled bets that have
    an odds_taken value recorded.

    Returns:
        {
          "total":   int,    # settled bets with odds
          "wins":    int,
          "losses":  int,
          "roi_pct": float,  # cumulative ROI as a percentage (sum(roi) / n * 100)
          "yield_pct": float # yield = total_profit / total_staked * 100
        }
    Returns None when fewer than 5 qualifying bets exist.
    """
    rows = _load_rows()
    roi_values = []
    for r in rows:
        result = (r.get("result") or "").strip().upper()
        if result not in {"W", "L"}:
            continue
        roi_raw = (r.get("roi") or "").strip()
        if not roi_raw:
            continue
        try:
            roi_values.append((result, float(roi_raw)))
        except ValueError:
            continue

    if len(roi_values) < 5:
        return None

    wins   = sum(1 for r, _ in roi_values if r == "W")
    losses = sum(1 for r, _ in roi_values if r == "L")
    total  = len(roi_values)
    total_roi = sum(v for _, v in roi_values)
    roi_pct   = (total_roi / total) * 100

    return {
        "total":     total,
        "wins":      wins,
        "losses":    losses,
        "roi_pct":   roi_pct,
    }



def log_bets(picks):
    """
    Append picks to bets_log.csv, skipping any whose match_id is already logged.
    Each pick dict must contain: match_id, home, away, league, market, pick, model_prob.
    'result' is left blank — filled in automatically next run via settle_bets().
    """
    already_logged = _load_logged_ids()
    needs_header = not _has_header()
    date_str = datetime.now().strftime("%Y-%m-%d")

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()

        for pick in picks:
            mid     = str(pick.get("match_id", ""))
            market  = pick.get("market", "")
            pick_nm = pick.get("pick", "")
            key     = (mid, market, pick_nm)
            if mid and key in already_logged:
                continue   # already logged — skip

            odds_taken = pick.get("odds")
            home_pos = pick.get("home_position")
            away_pos = pick.get("away_position")
            form_adv = pick.get("form_adv")
            exp_tot  = pick.get("expected_total")
            writer.writerow({
                "match_id":        mid,
                "date":            date_str,
                "home":            pick.get("home", ""),
                "away":            pick.get("away", ""),
                "league":          pick.get("league", ""),
                "market":          pick.get("market", ""),
                "pick":            pick.get("pick", ""),
                "model_prob":      f"{pick.get('model_prob', 0):.3f}",
                "odds_taken":      f"{odds_taken:.2f}" if odds_taken is not None else "",
                "edge":            f"{pick['edge']:.1f}" if pick.get("edge") is not None else "",
                "result":          "",
                "roi":             "",
                "settle_attempts": 0,
                "home_position":   f"{home_pos:.3f}" if home_pos is not None else "",
                "away_position":   f"{away_pos:.3f}" if away_pos is not None else "",
                "form_adv":        f"{form_adv:.3f}" if form_adv is not None else "",
                "expected_total":  f"{exp_tot:.3f}" if exp_tot is not None else "",
                "model_version":   MODEL_VERSION,
            })
            already_logged.add(key)
