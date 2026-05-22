"""
fit_xg_conv.py — Compute per-league XG_CONV and expected-total gate values from matches.db.

Run this script after 200+ matches have accumulated to get data-driven values to
update XG_CONV_BY_LEAGUE and EXPECTED_TOTAL_GATES_BY_LEAGUE in config.py.

Usage:
    python -m scripts.fit_xg_conv
    python -m scripts.fit_xg_conv --min-matches 50
"""

import sqlite3
import argparse
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "matches.db"


def _connect():
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"matches.db not found at {_DB_PATH}")
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def fit_xg_conv(min_matches: int = 100):
    """
    For each league in matches.db, compute:
      1. XG_CONV = sum(home_goals + away_goals) / sum(home_sot + away_sot)
         Only matches where both SoT columns are non-null are included.
      2. mean_total = mean(home_goals + away_goals) across all matches (SoT or not).
         Used to set per-league expected_total gates.
    """
    conn = _connect()
    try:
        leagues = [r[0] for r in conn.execute(
            "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL ORDER BY league"
        ).fetchall()]

        if not leagues:
            print("No league data in matches.db yet.")
            return

        print(f"\n{'League':<6}  {'SoT matches':>11}  {'XG_CONV':>8}  "
              f"{'All matches':>11}  {'Mean goals':>11}  {'Gate O2.5':>10}  {'Cap U2.5':>9}")
        print("-" * 80)

        xg_conv_out  = {}
        gate_o25_out = {}
        cap_u25_out  = {}

        for league in leagues:
            # XG_CONV: only matches with shot data
            sot_rows = conn.execute("""
                SELECT home_goals, away_goals, home_shots_on_target, away_shots_on_target
                FROM matches
                WHERE league = ?
                  AND home_shots_on_target IS NOT NULL
                  AND away_shots_on_target IS NOT NULL
                  AND stage = 'REGULAR_SEASON'
            """, (league,)).fetchall()

            # Mean goals: all regular-season matches
            all_rows = conn.execute("""
                SELECT home_goals, away_goals
                FROM matches
                WHERE league = ?
                  AND stage = 'REGULAR_SEASON'
            """, (league,)).fetchall()

            n_all = len(all_rows)
            n_sot = len(sot_rows)

            if n_all < min_matches:
                print(f"{league:<6}  {'(skip — only ' + str(n_all) + ' matches)':>60}")
                continue

            mean_total = sum(r[0] + r[1] for r in all_rows) / n_all
            gate_o25   = round(mean_total - 0.10, 2)
            cap_u25    = round(mean_total + 0.10, 2)

            gate_o25_out[league] = gate_o25
            cap_u25_out[league]  = cap_u25

            if n_sot >= min_matches // 2:
                total_goals = sum(r[0] + r[1] for r in sot_rows)
                total_sot   = sum(r[2] + r[3] for r in sot_rows)
                xg_conv     = round(total_goals / total_sot, 3) if total_sot else None
                xg_conv_out[league] = xg_conv
                xg_str = f"{xg_conv:.3f}" if xg_conv else "N/A"
            else:
                xg_str = f"(only {n_sot} SoT rows)"

            print(f"{league:<6}  {n_sot:>11}  {xg_str:>8}  "
                  f"{n_all:>11}  {mean_total:>10.2f}g  {gate_o25:>10.2f}  {cap_u25:>9.2f}")

        print()
        if xg_conv_out:
            print("# Paste into config.py — XG_CONV_BY_LEAGUE:")
            print("XG_CONV_BY_LEAGUE = {")
            for lg, v in xg_conv_out.items():
                print(f'    "{lg}": {v},')
            print("}")
            print()

        if gate_o25_out:
            print('# Paste into config.py — EXPECTED_TOTAL_GATES_BY_LEAGUE["Over 2.5"]:')
            for lg, v in gate_o25_out.items():
                print(f'    "{lg}": {v},   # mean {v + 0.10:.2f} goals/game')
            print()
            print('# Paste into config.py — EXPECTED_TOTAL_CAPS_BY_LEAGUE["Under 2.5"]:')
            for lg, v in cap_u25_out.items():
                print(f'    "{lg}": {v},')

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fit per-league xG conversion and goal rates from matches.db")
    parser.add_argument("--min-matches", type=int, default=100,
                        help="Minimum regular-season matches needed to report a league (default: 100)")
    args = parser.parse_args()
    fit_xg_conv(min_matches=args.min_matches)


if __name__ == "__main__":
    main()
