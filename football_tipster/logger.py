import csv
import os
import time
from datetime import datetime
import warn_log

LOG_FILE = "bets_log.csv"
# Sequential model version. Bump (v1 -> v2 -> v3) whenever thresholds or model
# logic change significantly enough that the previous cohort's stats no longer
# represent the live model. Stamped on every new pick so analysis can filter by
# version (see logger.compute_roi_summary(model_version=...)).
#
# v1: post-Phase-1 thresholds, per-league DC_RHO, xG proxy, momentum, streaks,
#     winter under-scoring proxy, cross-fixture accumulators. The acca release
#     (2026-05-24) is additive — same single-pick logic, so stays in v1.
MODEL_VERSION = "v1"

FIELDS = [
    "match_id", "date", "home", "away", "league",
    "market", "pick", "model_prob", "odds_taken", "edge", "result", "roi",
    "settle_attempts",
    "home_position", "away_position", "form_adv", "expected_total",
    "model_version", "stake_units",
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

# Cross-fixture accumulator support.
# Rows with a match_id starting with this prefix are multi-leg accumulators
# whose match_id encodes the constituent fixture ids as "ACC:id1+id2+id3"
# and whose pick string encodes each leg as "<team>: <pick_name> @<odds>"
# joined with " + " (same separator already used for same-fixture combos).
_ACC_PREFIX = "ACC:"
# Statuses that cause a leg to be dropped from the acca (skip and settle on the rest)
_ACC_VOID_STATUSES = {"POSTPONED", "CANCELLED", "SUSPENDED"}
# Statuses that mean the leg has a final result we can grade
_ACC_FINAL_STATUSES = {"FINISHED", "AWARDED"}


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


def _settle_acca(row, fetcher_mod):
    """Settle a cross-fixture accumulator row.

    Returns a tuple (outcome, roi_str, retry):
        outcome  : "W" | "L" | "VOID" | None
        roi_str  : string ROI value (positive on W, "-1.000" on L, "" on VOID)
        retry    : True when the acca should stay pending (some leg still in play
                   or transient API error) — caller should bump settle_attempts.

    Postponed/cancelled/suspended legs are skipped and the acca settles on the
    surviving legs. The payout odds in that case are the product of only the
    surviving legs' odds — matching the standard bookmaker rule for void legs.
    """
    mid_field  = (row.get("match_id") or "").strip()
    pick_field = (row.get("pick") or "").strip()
    if not mid_field.startswith(_ACC_PREFIX) or not pick_field:
        return None, "", True

    leg_ids  = mid_field[len(_ACC_PREFIX):].split("+")
    leg_strs = pick_field.split(" + ")
    if len(leg_ids) != len(leg_strs) or not leg_ids:
        # malformed — don't corrupt; keep retrying so a fix can be applied later
        return None, "", True

    surviving_odds = 1.0
    surviving_any  = False
    all_wins       = True

    for leg_id, leg_str in zip(leg_ids, leg_strs):
        # Parse "<team>: <pick_name> @<odds>"
        try:
            pick_part, odds_part = leg_str.rsplit(" @", 1)
            leg_odds = float(odds_part)
            _team, leg_pick_name = pick_part.split(": ", 1)
            leg_pick_name = leg_pick_name.strip()
        except (ValueError, AttributeError):
            return None, "", True   # malformed leg encoding — retry

        try:
            data = fetcher_mod.get_match(leg_id)
        except Exception:
            return None, "", True   # API error — stay pending

        match  = data.get("match") or data
        status = match.get("status", "")

        if status in _ACC_VOID_STATUSES:
            continue   # void leg — skip but keep rest

        if status not in _ACC_FINAL_STATUSES:
            return None, "", True   # still pending — wait

        score = match.get("score", {})
        ft    = score.get("fullTime", {})
        hg    = ft.get("home")
        ag    = ft.get("away")
        if hg is None or ag is None:
            return None, "", True

        total_g = hg + ag
        winner  = score.get("winner", "")
        leg_res = _evaluate_leg(leg_pick_name, hg, ag, winner, total_g)
        if leg_res is None:
            # unrecognised leg type — log and treat as ERR so it settles
            warn_log.fallback(
                f"unrecognised acca leg '{leg_pick_name}' — cannot evaluate",
                "acca settled as ERR",
                match_id=str(leg_id),
            )
            return "ERR", "", False

        surviving_any = True
        surviving_odds *= leg_odds
        if leg_res == "L":
            all_wins = False
        time.sleep(0.15)   # stay within free-tier rate limit

    if not surviving_any:
        # every leg voided — bet returns stake at the bookmaker
        return "VOID", "", False

    if all_wins:
        return "W", f"{(surviving_odds - 1):.3f}", False
    return "L", "-1.000", False


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

        # ── Cross-fixture accumulator ─────────────────────────────────────
        if mid.startswith(_ACC_PREFIX):
            outcome, roi_str, retry = _settle_acca(row, fetcher)
            if retry:
                row["settle_attempts"] = attempts + 1
                failed += 1
            else:
                row["result"] = outcome
                row["roi"]    = roi_str
                settled += 1
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


def compute_roi_summary(model_version=None):
    """
    Return a dict with cumulative ROI stats from all settled bets that have
    an odds_taken value recorded.

    model_version: when set, only rows whose `model_version` column matches the
                   given value are counted. Use this to isolate the current model
                   cohort from older pre-threshold history that drags averages
                   down. When None (default), all settled rows are included.

    Returns:
        {
          "total":   int,    # settled bets with odds (after filter)
          "wins":    int,
          "losses":  int,
          "roi_pct": float,  # cumulative ROI as a percentage (sum(roi) / n * 100)
        }
    Returns None when fewer than 5 qualifying bets exist.
    """
    rows = _load_rows()
    roi_values = []
    for r in rows:
        if model_version is not None:
            if (r.get("model_version") or "").strip() != model_version:
                continue
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
            stake_u  = pick.get("stake_units")
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
                "stake_units":     str(stake_u) if stake_u is not None else "",
            })
            already_logged.add(key)
