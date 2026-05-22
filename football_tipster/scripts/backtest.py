"""
scripts/backtest.py — historical backtesting framework.

Replays finished fixtures from football-data.org, runs the full model pipeline
on data available *before* each match, and compares predictions to actual results.

Outputs:
  - Per-market win rate and Brier score
  - Calibration curve (predicted prob bucket vs actual frequency)
  - Over 2.5 bias analysis

Usage:
    python -m scripts.backtest --leagues PL BL1 --seasons 2022 2023
    python -m scripts.backtest --leagues PL --seasons 2023 --min-prob 0.55
    python -m scripts.backtest --leagues PL --seasons 2023 --no-cache

No data leakage: for each fixture the model only sees matches played
strictly before that fixture's utcDate.
"""

import argparse
import sys
import os
import math
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextlib

import fetcher
import analyzer
import markets
from logger import _evaluate_result
from config import LEAGUES, CL_KNOCKOUT_STAGES, DC_RHO


@contextlib.contextmanager
def _patch_rho(rho_override: float):
    """Temporarily set all DC_RHO values to rho_override for a sweep run."""
    orig = dict(analyzer.DC_RHO)
    for k in analyzer.DC_RHO:
        analyzer.DC_RHO[k] = rho_override
    try:
        yield
    finally:
        analyzer.DC_RHO.update(orig)


@contextlib.contextmanager
def _patch_xg_conv(xg_conv: float):
    """Temporarily override the module-level XG_CONV constant in analyzer."""
    orig = analyzer.XG_CONV
    analyzer.XG_CONV = xg_conv
    try:
        yield
    finally:
        analyzer.XG_CONV = orig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _build_standings_from_history(matches: list[dict], before_date: datetime) -> dict:
    """
    Derive lightweight standings from match history up to before_date.
    Returns {team_id: {avg_scored, avg_conceded, played, form_score, position}}.
    This is a best-effort approximation — backtest uses it where live standings
    would have been available on the day.
    """
    scored   = defaultdict(list)
    conceded = defaultdict(list)
    form     = defaultdict(list)  # newest first

    for m in matches:
        try:
            md = _parse_date(m.get("utcDate", ""))
        except (ValueError, TypeError):
            continue
        if md >= before_date:
            continue
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue
        home_id = m.get("homeTeam", {}).get("id")
        away_id = m.get("awayTeam", {}).get("id")
        winner  = m.get("score", {}).get("winner", "")
        if home_id:
            scored[home_id].append(hg)
            conceded[home_id].append(ag)
            form[home_id].append("W" if winner == "HOME_TEAM" else ("D" if winner == "DRAW" else "L"))
        if away_id:
            scored[away_id].append(ag)
            conceded[away_id].append(hg)
            form[away_id].append("W" if winner == "AWAY_TEAM" else ("D" if winner == "DRAW" else "L"))

    standings = {}
    all_teams = set(scored) | set(conceded)
    for tid in all_teams:
        s = scored[tid]
        c = conceded[tid]
        f = form[tid][:6]   # last 6 for form score
        if not s:
            continue
        # Form score using same formula as analyzer._compute_form_score()
        fs = analyzer._compute_form_score("".join(f)) if f else 0.5
        standings[tid] = {
            "id":            tid,
            "played":        len(s),
            "avg_scored":    sum(s) / len(s),
            "avg_conceded":  sum(c) / len(c) if c else 1.2,
            "form_score":    fs,
            "position":      10,   # unknown without live table — use neutral
        }

    return standings


def _filter_before(matches: list[dict], before_date: datetime, team_id: int) -> dict:
    """Return {matches: [...]} containing only this team's matches before before_date."""
    result = []
    for m in matches:
        try:
            md = _parse_date(m.get("utcDate", ""))
        except (ValueError, TypeError):
            continue
        if md >= before_date:
            continue
        home_id = m.get("homeTeam", {}).get("id")
        away_id = m.get("awayTeam", {}).get("id")
        if home_id == team_id or away_id == team_id:
            result.append(m)
    return {"matches": result}


def _brier_score(predictions: list[tuple[float, int]]) -> float:
    """Mean Brier score. predictions = [(prob, outcome)] where outcome=1 for W."""
    if not predictions:
        return float("nan")
    return sum((p - o) ** 2 for p, o in predictions) / len(predictions)


def _calibration_buckets(predictions: list[tuple[float, int]], n_buckets: int = 10):
    """
    Group predictions into n_buckets equal-width probability bands.
    Returns list of (mid_prob, predicted_freq, actual_freq, count).
    """
    buckets = defaultdict(lambda: {"count": 0, "wins": 0, "prob_sum": 0.0})
    width = 1.0 / n_buckets
    for prob, outcome in predictions:
        b = min(int(prob / width), n_buckets - 1)
        buckets[b]["count"]    += 1
        buckets[b]["wins"]     += outcome
        buckets[b]["prob_sum"] += prob

    result = []
    for b in range(n_buckets):
        if buckets[b]["count"] == 0:
            continue
        n      = buckets[b]["count"]
        mid    = buckets[b]["prob_sum"] / n
        actual = buckets[b]["wins"] / n
        result.append((mid, actual, n))
    return result


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def replay_season(league_code: str, season: int, use_cache: bool = True,
                  min_prob: float = 0.0, xg_conv: float = None,
                  rho_override: float = None) -> list[dict]:
    """
    Fetch a full season and replay every finished fixture through the model.
    Returns a list of prediction records — one per pick generated.

    xg_conv / rho_override: when set, temporarily patch the corresponding
    module-level constants in analyzer.py so the sweep doesn't touch live config.

    Each record:
        {league, season, match_id, utc_date, home_id, away_id,
         market, pick, model_prob, actual_result,   # W/L
         correct,   # bool
         brier_input: (prob, outcome)}
    """
    ctx_rho = _patch_rho(rho_override) if rho_override is not None else contextlib.nullcontext()
    ctx_xg  = _patch_xg_conv(xg_conv)  if xg_conv    is not None else contextlib.nullcontext()
    with ctx_rho, ctx_xg:
        return _replay_season_inner(league_code, season, use_cache, min_prob)


def _replay_season_inner(league_code: str, season: int, use_cache: bool,
                         min_prob: float) -> list[dict]:
    print(f"  Fetching {league_code} {season} ...", end="", flush=True)
    raw = fetcher.get_season_matches(league_code, season, use_cache=use_cache)
    all_matches = [m for m in raw.get("matches", []) if m.get("status") == "FINISHED"]
    print(f" {len(all_matches)} finished matches")

    if len(all_matches) < 10:
        print(f"  [skip] Too few matches for {league_code} {season}")
        return []

    # Sort chronologically so we can replay in order
    def _match_date(m):
        try:
            return _parse_date(m.get("utcDate", "1970-01-01T00:00:00Z"))
        except (ValueError, TypeError):
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    all_matches.sort(key=_match_date)
    records = []

    for i, match in enumerate(all_matches):
        match_date = _match_date(match)
        home_id  = match.get("homeTeam", {}).get("id")
        away_id  = match.get("awayTeam", {}).get("id")
        match_id = match.get("id")
        stage    = match.get("stage", "")

        if not home_id or not away_id or not match_id:
            continue

        ft      = match.get("score", {}).get("fullTime", {})
        hg_act  = ft.get("home")
        ag_act  = ft.get("away")
        if hg_act is None or ag_act is None:
            continue

        # Build rolling histories — only matches strictly before this fixture
        home_hist_data = _filter_before(all_matches, match_date, home_id)
        away_hist_data = _filter_before(all_matches, match_date, away_id)

        # Need at least 5 matches per team to produce a meaningful prediction
        if (len(home_hist_data["matches"]) < 5 or
                len(away_hist_data["matches"]) < 5):
            continue

        standings = _build_standings_from_history(all_matches, match_date)
        if home_id not in standings or away_id not in standings:
            continue

        strength_factors = analyzer._compute_strength_factors(standings)
        home_history = analyzer.parse_team_history(
            home_hist_data, home_id, strength_factors,
            reference_date=match_date,
        )
        away_history = analyzer.parse_team_history(
            away_hist_data, away_id, strength_factors,
            reference_date=match_date,
        )

        # H2H: all prior meetings between these two teams before match_date
        h2h_matches = [
            m for m in all_matches
            if _match_date(m) < match_date and (
                (m.get("homeTeam", {}).get("id") == home_id and
                 m.get("awayTeam", {}).get("id") == away_id) or
                (m.get("homeTeam", {}).get("id") == away_id and
                 m.get("awayTeam", {}).get("id") == home_id)
            )
        ]
        h2h = analyzer.parse_h2h({"matches": h2h_matches}, home_team_id=home_id)

        home_standing = {**standings[home_id], "id": home_id}
        away_standing = {**standings[away_id], "id": away_id}

        is_knockout = stage in CL_KNOCKOUT_STAGES

        try:
            probs = analyzer.compute_match_probabilities(
                league_code, home_standing, away_standing,
                home_history, away_history, h2h,
                strength_factors=strength_factors,
                is_knockout=is_knockout,
            )
        except AssertionError:
            continue   # probability normalisation failed — skip

        # Build the fake match result dict for _evaluate_result
        winner = ("HOME_TEAM" if hg_act > ag_act else
                  "AWAY_TEAM" if ag_act > hg_act else "DRAW")
        match_result = {
            "status": "FINISHED",
            "score": {
                "winner": winner,
                "fullTime": {"home": hg_act, "away": ag_act},
            },
        }

        # Evaluate each market pick
        market_picks = []
        market_picks += markets.evaluate_1x2(probs)
        market_picks += markets.evaluate_over_under(
            probs,
            expected_total=probs.get("expected_total"),
        )
        market_picks += markets.evaluate_btts(probs)

        for pick in market_picks:
            prob = pick["model_prob"]
            if prob < min_prob:
                continue
            pick_name  = pick["pick"]
            market     = pick["market"]
            outcome    = _evaluate_result(pick_name, market, match_result)
            if outcome is None:
                continue
            correct = outcome == "W"
            records.append({
                "league":       league_code,
                "season":       season,
                "match_id":     match_id,
                "utc_date":     match.get("utcDate", ""),
                "home_id":      home_id,
                "away_id":      away_id,
                "market":       market,
                "pick":         pick_name,
                "model_prob":   prob,
                "actual":       outcome,
                "correct":      correct,
            })

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_market_summary(records: list[dict]):
    by_market = defaultdict(list)
    for r in records:
        by_market[r["market"]].append(r)

    print(f"\n{'Market':<20} {'Pick':<22} {'N':>5} {'WR%':>6} {'Brier':>7}")
    print("-" * 65)

    for market in ["1X2", "Over/Under", "BTTS"]:
        picks_in_market = defaultdict(list)
        for r in by_market.get(market, []):
            picks_in_market[r["pick"]].append(r)

        for pick_name in sorted(picks_in_market):
            recs = picks_in_market[pick_name]
            n    = len(recs)
            wr   = sum(1 for r in recs if r["correct"]) / n * 100
            bs   = _brier_score([(r["model_prob"], 1 if r["correct"] else 0) for r in recs])
            print(f"  {market:<18} {pick_name:<22} {n:>5} {wr:>6.1f} {bs:>7.4f}")

    print()


def _print_calibration(records: list[dict], market: str = None):
    subset = [r for r in records if market is None or r["market"] == market]
    preds  = [(r["model_prob"], 1 if r["correct"] else 0) for r in subset]
    buckets = _calibration_buckets(preds)

    label = market or "ALL"
    print(f"\nCalibration curve - {label}  (predicted prob -> actual win rate)")
    print(f"  {'Pred prob':>10}  {'Actual WR':>10}  {'N':>6}  {'Deviation':>10}")
    print("  " + "-" * 44)
    for mid, actual, n in buckets:
        if n < 5:
            continue
        dev = actual - mid
        bar = "+" * int(abs(dev) * 40) if dev > 0 else "-" * int(abs(dev) * 40)
        print(f"  {mid:>10.2f}  {actual:>10.2f}  {n:>6}  {dev:>+10.3f}  {bar}")
    print()


def _print_summary(records: list[dict]):
    if not records:
        print("  No predictions generated.")
        return

    total  = len(records)
    wins   = sum(1 for r in records if r["correct"])
    wr     = wins / total * 100
    bs     = _brier_score([(r["model_prob"], 1 if r["correct"] else 0) for r in records])
    leagues = sorted({r["league"] for r in records})
    seasons = sorted({r["season"] for r in records})

    print(f"\n{'='*65}")
    print(f"  BACKTEST SUMMARY")
    print(f"  Leagues : {', '.join(leagues)}")
    print(f"  Seasons : {', '.join(str(s) for s in seasons)}")
    print(f"  Total predictions : {total}")
    print(f"  Overall win rate  : {wr:.1f}%")
    print(f"  Overall Brier     : {bs:.4f}  (perfect=0, random~0.25)")
    print(f"{'='*65}")

    _print_market_summary(records)
    _print_calibration(records, market="Over/Under")
    _print_calibration(records, market="1X2")


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------

def _sweep_summary(param_name: str, param_val: float, records: list[dict]) -> dict:
    """Return a compact summary dict for one sweep point."""
    ou_records = [r for r in records if r["market"] == "Over/Under"]
    all_bs = _brier_score([(r["model_prob"], 1 if r["correct"] else 0) for r in records])
    ou_bs  = _brier_score([(r["model_prob"], 1 if r["correct"] else 0) for r in ou_records])
    ou_wr  = (sum(1 for r in ou_records if r["correct"]) / len(ou_records) * 100
              if ou_records else float("nan"))
    total_wr = (sum(1 for r in records if r["correct"]) / len(records) * 100
                if records else float("nan"))
    return {
        "param": param_name,
        "value": param_val,
        "total_n": len(records),
        "total_wr": total_wr,
        "total_brier": all_bs,
        "ou_n": len(ou_records),
        "ou_wr": ou_wr,
        "ou_brier": ou_bs,
    }


def _run_rho_sweep(leagues: list, seasons: list, use_cache: bool):
    """Test DC_RHO values and report Over/Under Brier + WR for each."""
    rho_values = [-0.10, -0.13, -0.15, -0.18, -0.20, -0.23]
    print(f"\n{'='*70}")
    print(f"  DC_RHO SWEEP  (leagues={leagues}, seasons={seasons})")
    print(f"  Goal: minimise Over/Under Brier score")
    print(f"{'='*70}")
    print(f"\n  {'RHO':>6}  {'OU N':>6}  {'OU WR%':>7}  {'OU Brier':>9}  {'All Brier':>10}")
    print("  " + "-" * 48)

    results = []
    for rho in rho_values:
        all_records = []
        for league in leagues:
            for season in seasons:
                recs = replay_season(league, season, use_cache=use_cache, rho_override=rho)
                all_records.extend(recs)
        s = _sweep_summary("DC_RHO", rho, all_records)
        results.append(s)
        print(f"  {rho:>6.2f}  {s['ou_n']:>6}  {s['ou_wr']:>7.1f}  {s['ou_brier']:>9.4f}  {s['total_brier']:>10.4f}")

    best = min(results, key=lambda r: r["ou_brier"])
    print(f"\n  Best DC_RHO = {best['value']:.2f}  (OU Brier {best['ou_brier']:.4f}, OU WR {best['ou_wr']:.1f}%)")
    print(f"\n  ACTION: update DC_RHO in config.py to {best['value']:.2f} for all leagues")
    print()


def _run_xg_sweep(leagues: list, seasons: list, use_cache: bool):
    """Test XG_CONV values and report Over/Under Brier + WR for each."""
    xg_values = [0.25, 0.28, 0.30, 0.33, 0.35]
    print(f"\n{'='*70}")
    print(f"  XG_CONV SWEEP  (leagues={leagues}, seasons={seasons})")
    print(f"  Goal: minimise Over/Under Brier score")
    print(f"{'='*70}")
    print(f"\n  {'XG_CONV':>8}  {'OU N':>6}  {'OU WR%':>7}  {'OU Brier':>9}  {'All Brier':>10}")
    print("  " + "-" * 50)

    results = []
    for xg in xg_values:
        all_records = []
        for league in leagues:
            for season in seasons:
                recs = replay_season(league, season, use_cache=use_cache, xg_conv=xg)
                all_records.extend(recs)
        s = _sweep_summary("XG_CONV", xg, all_records)
        results.append(s)
        print(f"  {xg:>8.2f}  {s['ou_n']:>6}  {s['ou_wr']:>7.1f}  {s['ou_brier']:>9.4f}  {s['total_brier']:>10.4f}")

    best = min(results, key=lambda r: r["ou_brier"])
    print(f"\n  Best XG_CONV = {best['value']:.2f}  (OU Brier {best['ou_brier']:.4f}, OU WR {best['ou_wr']:.1f}%)")
    print(f"\n  ACTION: update XG_CONV in analyzer.py to {best['value']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest the football model on historical data.")
    parser.add_argument(
        "--leagues", nargs="+", default=["PL"],
        help="League codes to backtest (default: PL)",
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2023],
        help="Season start years to backtest (default: 2023)",
    )
    parser.add_argument(
        "--min-prob", type=float, default=0.0,
        help="Only include picks with model_prob >= this value (default: 0, all picks)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh API calls (ignores cached season data)",
    )
    parser.add_argument(
        "--sweep-rho", action="store_true",
        help="Sweep DC_RHO values (-0.13 to -0.20) to find optimal Over/Under calibration",
    )
    parser.add_argument(
        "--sweep-xg", action="store_true",
        help="Sweep XG_CONV values (0.25 to 0.35) to find optimal xG conversion rate",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.sweep_rho:
        _run_rho_sweep(args.leagues, args.seasons, use_cache)
        return

    if args.sweep_xg:
        _run_xg_sweep(args.leagues, args.seasons, use_cache)
        return

    all_records = []
    for league in args.leagues:
        for season in args.seasons:
            print(f"\n[{league} {season}]")
            records = replay_season(league, season, use_cache=use_cache, min_prob=args.min_prob)
            all_records.extend(records)
            if records:
                _print_market_summary(records)

    _print_summary(all_records)


if __name__ == "__main__":
    main()
