"""
scripts/validate_knockout_factor.py

Compute the empirical knockout goals factor from CL historical data.

Pulls multiple CL seasons, splits matches into group-stage vs knockout-stage,
computes average goals/game for each, and prints the ratio that should replace
KNOCKOUT_GOALS_FACTOR in config.py.

Usage:
    python -m scripts.validate_knockout_factor
    python -m scripts.validate_knockout_factor --seasons 2019 2020 2021 2022 2023
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetcher
from config import CL_KNOCKOUT_STAGES

GROUP_STAGES = {"GROUP_STAGE", "LEAGUE_PHASE"}


def analyse_season(season: int, use_cache: bool = True) -> dict | None:
    print(f"  Fetching CL {season} ...", end="", flush=True)
    raw = fetcher.get_season_matches("CL", season, use_cache=use_cache)
    matches = [m for m in raw.get("matches", []) if m.get("status") == "FINISHED"]
    print(f"  {len(matches)} finished")

    group_goals, group_n = 0, 0
    knockout_goals, knockout_n = 0, 0
    unknown = []

    for m in matches:
        stage = m.get("stage", "")
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue
        total = hg + ag

        if stage in GROUP_STAGES:
            group_goals += total
            group_n += 1
        elif stage in CL_KNOCKOUT_STAGES:
            knockout_goals += total
            knockout_n += 1
        else:
            unknown.append(stage)

    if unknown:
        unique_unknown = sorted(set(unknown))
        print(f"    Warning: unclassified stages: {unique_unknown}")

    if group_n == 0 or knockout_n == 0:
        print(f"    Skipping CL {season}: group={group_n}, knockout={knockout_n}")
        return None

    group_avg    = group_goals    / group_n
    knockout_avg = knockout_goals / knockout_n
    factor       = knockout_avg   / group_avg

    return {
        "season":        season,
        "group_n":       group_n,
        "group_avg":     group_avg,
        "knockout_n":    knockout_n,
        "knockout_avg":  knockout_avg,
        "factor":        factor,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate KNOCKOUT_GOALS_FACTOR from CL data")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2019, 2020, 2021, 2022, 2023],
                        help="CL season start years to include")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache")
    args = parser.parse_args()

    use_cache = not args.no_cache
    rows = []

    for season in args.seasons:
        try:
            r = analyse_season(season, use_cache=use_cache)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR CL {season}: {e}")

    if not rows:
        print("\nNo data collected — check API key and network.")
        return

    print(f"\n{'='*70}")
    print(f"  {'Season':>8}  {'Group n':>8}  {'Group avg':>10}  {'KO n':>6}  {'KO avg':>8}  {'Factor':>8}")
    print(f"  {'-'*66}")

    total_group_goals = total_group_n = total_ko_goals = total_ko_n = 0
    for r in rows:
        print(
            f"  {r['season']:>8}  {r['group_n']:>8}  {r['group_avg']:>10.3f}"
            f"  {r['knockout_n']:>6}  {r['knockout_avg']:>8.3f}  {r['factor']:>8.4f}"
        )
        total_group_goals += r["group_avg"] * r["group_n"]
        total_group_n     += r["group_n"]
        total_ko_goals    += r["knockout_avg"] * r["knockout_n"]
        total_ko_n        += r["knockout_n"]

    overall_group_avg = total_group_goals / total_group_n
    overall_ko_avg    = total_ko_goals    / total_ko_n
    overall_factor    = overall_ko_avg    / overall_group_avg

    print(f"  {'-'*66}")
    print(
        f"  {'OVERALL':>8}  {total_group_n:>8}  {overall_group_avg:>10.3f}"
        f"  {total_ko_n:>6}  {overall_ko_avg:>8.3f}  {overall_factor:>8.4f}"
    )
    print(f"{'='*70}")
    print()
    from config import KNOCKOUT_GOALS_FACTOR
    print(f"  Current  KNOCKOUT_GOALS_FACTOR = {KNOCKOUT_GOALS_FACTOR}")
    print(f"  Empirical factor ({total_ko_n} knockout matches) = {overall_factor:.4f}")

    delta = overall_factor - KNOCKOUT_GOALS_FACTOR
    direction = "higher" if delta > 0 else "lower"
    print(f"  => {abs(delta):.4f} {direction} than current setting")

    rounded = round(overall_factor * 20) / 20  # round to nearest 0.05
    print()
    print(f"  Recommended: set KNOCKOUT_GOALS_FACTOR = {overall_factor:.2f}  (or {rounded:.2f} rounded to 0.05)")
    print()


if __name__ == "__main__":
    main()
