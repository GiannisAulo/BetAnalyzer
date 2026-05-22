"""
calibrate_baselines.py — Empirical baseline validator for BetAnalyzer.

Reads all settled rows in bets_log.csv (result = W or L, market = 1X2),
groups by league, computes empirical home/draw/away win rates, and compares
them against the BASELINES dict in config.py.

Usage (run from football_tipster/ directory):
    python scripts/calibrate_baselines.py

Outputs a table showing empirical vs configured rates and the delta.
Positive delta means the config is overconfident for that outcome;
negative means it's underconfident.

Recommendation: when a league has >= 200 settled 1X2 bets and a delta
of >= 2 pp, update BASELINES in config.py to the empirical rates.

Also importable: check_baselines() returns a list of warning strings
for leagues that need updating -- used by main.py at startup.
"""

import csv
import os
import sys
from collections import defaultdict

# Allow imports from parent directory when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASELINES

LOG_FILE      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bets_log.csv")
MIN_BETS      = 30    # minimum bets per league to show in the report
RECOMMEND_MIN = 200   # minimum to recommend a config update


def _load():
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records = []
    for row in rows:
        result = (row.get("result") or "").strip().upper()
        if result not in {"W", "L"}:
            continue
        market = (row.get("market") or "").strip()
        if market != "1X2":
            continue
        pick   = (row.get("pick") or "").strip()
        league = (row.get("league") or "").strip()
        if not pick or not league:
            continue
        records.append({
            "league": league,
            "pick":   pick,
            "won":    1 if result == "W" else 0,
        })

    return records


def _compute(records):
    """Return {league: {pick_label: [wins, total]}}."""
    data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        entry = data[r["league"]][r["pick"]]
        entry[0] += r["won"]
        entry[1] += 1
    return data


def check_baselines() -> list:
    """
    Return a list of warning strings for leagues whose empirical 1X2 rates
    diverge from BASELINES by >= 2 pp with >= RECOMMEND_MIN settled bets.
    Returns [] when there is nothing to flag.
    Called automatically by main.py at startup.
    """
    records = _load()
    if not records:
        return []

    data = _compute(records)
    outcome_map = {"Home Win": "home", "Draw": "draw", "Away Win": "away"}
    warnings = []

    for league, outcomes in sorted(data.items()):
        baseline = BASELINES.get(league, {})
        for pick_label, config_key in outcome_map.items():
            wins, n = outcomes.get(pick_label, [0, 0])
            if n < RECOMMEND_MIN:
                continue
            configured = baseline.get(config_key)
            if configured is None:
                continue
            empirical = wins / n
            delta = empirical - configured
            if abs(delta) >= 0.02:
                sign = "+" if delta > 0 else ""
                warnings.append(
                    f"{league} {pick_label}: empirical {empirical:.1%} vs config {configured:.1%} "
                    f"({sign}{delta:.1%}, n={n}) — update BASELINES in config.py"
                )

    return warnings


def auto_update_baselines() -> list:
    """
    Compute empirical win rates and rewrite the BASELINES block in config.py
    for any league that has >= RECOMMEND_MIN bets and >= 2 pp drift.

    Uses a blended update: new_value = 0.7 * empirical + 0.3 * configured
    so the config moves toward the data without overreacting to a single season.

    Returns a list of strings describing each change made (empty if nothing changed).
    """
    import re

    records = _load()
    if not records:
        return []

    data      = _compute(records)
    outcome_map = {"Home Win": "home", "Draw": "draw", "Away Win": "away"}

    # Build the set of updates: {league: {key: new_value}}
    updates: dict = {}
    for league, outcomes in data.items():
        baseline = BASELINES.get(league, {})
        for pick_label, config_key in outcome_map.items():
            wins, n = outcomes.get(pick_label, [0, 0])
            if n < RECOMMEND_MIN:
                continue
            configured = baseline.get(config_key)
            if configured is None:
                continue
            empirical = wins / n
            if abs(empirical - configured) < 0.02:
                continue
            # Blend toward empirical — don't overwrite completely in one step
            blended = round(0.7 * empirical + 0.3 * configured, 3)
            updates.setdefault(league, {})[config_key] = blended

    if not updates:
        return []

    # Read config.py
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"
    )
    with open(config_path, encoding="utf-8") as f:
        source = f.read()

    changes = []
    new_source = source

    for league, keys in updates.items():
        old_baseline = BASELINES.get(league, {})
        for config_key, new_val in keys.items():
            old_val = old_baseline.get(config_key)
            # Match e.g. "PL":  {"home": 0.46, "draw": 0.25, "away": 0.29}
            # and replace just the target key's value inside that line
            pattern = (
                r'("' + re.escape(league) + r'"\s*:\s*\{[^}]*"'
                + re.escape(config_key) + r'"\s*:\s*)(\d+\.\d+)'
            )
            def _replacer(m, nv=new_val):
                return m.group(1) + f"{nv:.2f}"
            new_source, count = re.subn(pattern, _replacer, new_source)
            if count:
                changes.append(
                    f"{league} {config_key}: {old_val:.2f} -> {new_val:.2f}"
                )

    if changes:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_source)
        # Reload BASELINES in this process so subsequent calls see new values
        import importlib, config as cfg_mod
        importlib.reload(cfg_mod)
        BASELINES.clear()
        BASELINES.update(cfg_mod.BASELINES)

    return changes


# ---------------------------------------------------------------------------
# Standalone report (full table)
# ---------------------------------------------------------------------------

def _delta_color(delta):
    if abs(delta) < 0.01:
        return ""
    return "\033[93m" if delta > 0 else "\033[96m"

RESET = "\033[0m"


def main():
    records = _load()
    if not records:
        print("No settled 1X2 bets found in bets_log.csv.")
        return

    data = _compute(records)
    outcome_map = {"Home Win": "home", "Draw": "draw", "Away Win": "away"}
    has_recommendation = False

    print()
    print("=" * 78)
    print(f"{'BASELINE CALIBRATION REPORT':^78}")
    print("=" * 78)
    print(f"  Total settled 1X2 bets: {len(records)}")
    print()
    print(
        f"  {'League':<6}  {'Outcome':<12}  {'Empirical':>10}  {'Config':>8}  "
        f"{'Delta':>8}  {'N':>5}  Note"
    )
    print("  " + "-" * 70)

    for league in sorted(data.keys()):
        outcomes = data[league]
        total_bets = sum(v[1] for v in outcomes.values())
        if total_bets < MIN_BETS:
            continue

        baseline = BASELINES.get(league, {})
        first = True
        for pick_label, config_key in outcome_map.items():
            wins, n = outcomes.get(pick_label, [0, 0])
            if n == 0:
                continue

            empirical  = wins / n
            configured = baseline.get(config_key)

            if configured is None:
                delta_str = "  n/a"
                color = ""
            else:
                delta = empirical - configured
                delta_str = f"{delta:+.1%}"
                color = _delta_color(delta)

            note = ""
            if configured is not None and n >= RECOMMEND_MIN and abs(empirical - configured) >= 0.02:
                note = "<- UPDATE RECOMMENDED"
                has_recommendation = True

            league_col = league if first else ""
            first = False

            print(
                f"  {league_col:<6}  {pick_label:<12}  "
                f"{empirical:>10.1%}  {configured if configured else '':>8}  "
                f"{color}{delta_str:>8}{RESET}  {n:>5}  {note}"
            )

        print()

    print("=" * 78)
    if has_recommendation:
        print()
        print("  Some baselines have >= 2 pp drift with >= 200 bets.")
        print("  Consider updating BASELINES in config.py to the empirical rates.")
    else:
        print()
        print("  No updates recommended (all deltas < 2 pp or insufficient data).")
    print()


if __name__ == "__main__":
    main()
