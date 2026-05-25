"""
main.py — Entry point for Football Betting Tipster.

Interactive use (recommended):
    python main.py

Scripted / automated use:
    python main.py --leagues PL BL1 SA         # specific leagues, skip menu
    python main.py --min-edge 3                # lower confidence threshold
    python main.py --no-cache                  # force fresh API calls
    python main.py --output coupon.txt         # save coupon to file
    python main.py --over25                    # Over 2.5 table only
"""

# Fix Windows terminal encoding before any Rich import
import sys
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import os
import subprocess
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

import fetcher
import analyzer
import markets
import coupon
import logger
import ml_calibrator
import odds_fetcher
import match_store
import staking
import drift
import corrections
import warn_log
from config import LEAGUES, DEFAULT_MIN_EDGE, ODDS_API_KEY, CL_KNOCKOUT_STAGES, CACHE_VERSION
from cache import get_cache_age_hours, evict_stale_cache
from scripts.calibrate_baselines import auto_update_baselines

# Status console — stdout so menu text and input() appear on the same stream.
# The coupon output (coupon.py) uses its own recorded console on stdout too,
# so everything ends up in the right place.
status = Console(stderr=False, legacy_windows=False, highlight=False)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

# Statuses that mean the match hasn't kicked off yet
_UNPLAYED = {"SCHEDULED", "TIMED"}


def _pick_sort_key(p):
    """Sort key that puts verified-edge picks above all no-edge picks.

    Picks with positive edge are scored in the range (0.5, 1.0]:
        0.5 + edge_component * 0.375 + prob_component * 0.125
    Picks with no edge (no odds or negative edge) are scored in [0, 0.5):
        prob * 0.4 + certainty * 0.1
    This guarantees any verified-edge pick outranks any no-edge pick.
    """
    prob  = p.get("model_prob", 0)
    edge  = p.get("edge") or 0.0
    unc   = p.get("uncertainty")
    cert  = (1.0 - unc) if unc is not None else 1.0
    if edge > 0:
        edge_norm = min(edge / 20.0, 1.0)
        return 0.5 + edge_norm * 0.375 + prob * 0.125
    else:
        return prob * 0.4 + cert * 0.1




def _today_unplayed(raw_fixtures):
    """Return only matches that haven't started yet (SCHEDULED or TIMED)."""
    matches = raw_fixtures.get("matches", [])
    return [m for m in matches if m.get("status", "") in _UNPLAYED]


def _default_standing():
    return {"form_score": 0.5, "position": 10, "avg_scored": 1.2, "avg_conceded": 1.2}


def _safe_prob_pct(raw) -> str:
    """Render a CSV cell as 'NN%' or '-'.

    Guards against blank/whitespace cells and malformed numerics (e.g. a hand-
    edited 'n/a' value) so the bet-history table never crashes on bad data.
    """
    if raw is None:
        return "-"
    s = str(raw).strip()
    if not s:
        return "-"
    try:
        return f"{float(s) * 100:.0f}%"
    except ValueError:
        return "-"


def _maybe_apply_corrections(drift_rows):
    """Interactive prompt: for each actionable drift row, offer to fit and
    persist an isotonic correction.

    Called only in interactive (menu) mode so batch CLI runs never block on
    input(). Skips rows where a correction already exists and was fitted from
    fewer than 20 bets behind the current count (avoids constant re-fitting on
    every coupon run).
    """
    actionable = [d for d in drift_rows if d["severity"] in ("yellow", "red")]
    if not actionable:
        return

    existing = {(c["market"], c["pick"]): c for c in corrections.list_corrections()}

    candidates = []
    for d in actionable:
        prev = existing.get((d["market"], d["pick"]))
        # Skip re-prompting until at least 20 new bets accumulated since the
        # last fit — avoids prompt fatigue on every run.
        if prev and (d["n"] - (prev.get("n_samples") or 0)) < 20:
            continue
        candidates.append(d)

    if not candidates:
        return

    status.print()
    status.print(
        "  [bold magenta]Correction available[/bold magenta] for "
        f"{len(candidates)} market(s). Applying writes to "
        "[dim]data/isotonic_corrections.json[/dim] and takes effect on the next run."
    )

    for d in candidates:
        prompt = (
            f"  Apply isotonic correction for "
            f"[bold]{d['market']} / {d['pick']}[/bold]  "
            f"(n={d['n']}, gap {d['gap_pp']:+.1f}pp)? [y/N]: "
        )
        status.print(prompt, end="")
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            status.print()
            return
        if answer != "y":
            continue
        entry = corrections.fit_correction(d["market"], d["pick"])
        if entry is None:
            status.print(
                f"  [yellow]Skipped — could not fit (need {corrections._MIN_FIT_SAMPLE}+ "
                f"bets or sklearn unavailable).[/yellow]"
            )
            continue
        corrections.save_correction(entry)
        # Reset the calibrator so the new correction is picked up on the next call
        ml_calibrator.reset_calibrator()
        status.print(
            f"  [green]Saved.[/green] Future picks for "
            f"{d['market']} / {d['pick']} will use the calibrated probability."
        )


def _accas_to_log_rows(accas):
    """Convert acca dicts (from markets.build_cross_fixture_accas) to bets_log
    row dicts that logger.log_bets can write.

    Schema (uses existing columns — no migration):
      match_id   = "ACC:<id1>+<id2>+<id3>"  (settle_bets detects this prefix)
      home       = "Team1 / Team2 / Team3"  (display label)
      away       = "(N-leg acca)"
      league     = "MULTI"
      market     = "AccaCross"
      pick       = "<team>: <pick> @<odds> + <team>: <pick> @<odds> + ..."
      model_prob = joint probability
      odds       = combined decimal odds
      edge       = joint edge in percent

    The per-leg odds are embedded in the pick string so settle_bets can
    recompute the effective payout odds if any leg is voided (postponed).

    Only accas where every leg has real bookmaker odds are logged. Inferred-odds
    accas are display-only — the user's real bet odds will differ from the
    model's fair odds, so any ROI computed on inferred odds would be misleading.
    """
    rows = []
    for acca in accas:
        if not acca.get("verified_edge"):
            continue   # informational only — don't track
        leg_ids  = [str(leg["match_id"]) for leg in acca["legs"]]
        leg_strs = [
            f"{leg['home']}: {leg['pick']} @{leg['odds']:.2f}"
            for leg in acca["legs"]
        ]
        home_label = " / ".join(leg["home"] for leg in acca["legs"])
        rows.append({
            "match_id":   "ACC:" + "+".join(leg_ids),
            "home":       home_label,
            "away":       f"({acca['size']}-leg acca)",
            "league":     "MULTI",
            "market":     "AccaCross",
            "pick":       " + ".join(leg_strs),
            "model_prob": acca["joint_prob"],
            "odds":       acca["joint_odds"],
            "edge":       acca["edge"],
            "stake_units": acca.get("stake_units"),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-league pipeline (with live progress updates)
# ---------------------------------------------------------------------------

def process_league(league_code, today_str, use_cache, progress, task):
    """
    Fetch and analyse all fixtures for one league on today_str (YYYY-MM-DD).
    Updates the Rich progress task at each network step.
    Returns a list of fixture dicts.
    """

    def step(desc):
        progress.update(task, description=f" [cyan]{league_code}[/cyan]  {desc}")

    # ── Fixtures ────────────────────────────────────────────────────────
    step("fetching fixtures ...")
    raw = fetcher.get_fixtures(league_code, today_str, use_cache=use_cache)
    matches = _today_unplayed(raw)

    if not matches:
        return []

    progress.update(task, total=len(matches))

    # ── Standings ───────────────────────────────────────────────────────
    step(f"loading standings  ({len(matches)} fixtures found)")
    standings_raw = fetcher.get_standings(league_code, use_cache=use_cache)
    standings = analyzer.parse_standings(standings_raw, league_code)

    # ── Strength factors (computed once per league from standings) ──────
    # Split home/away indices are added per-fixture after histories are fetched.
    strength_factors = analyzer._compute_strength_factors(standings)

    # ── Live odds (one call per league, graceful fallback) ───────────────
    step("fetching live odds ...")
    odds_map = odds_fetcher.get_odds_for_league(league_code, use_cache=use_cache) if ODDS_API_KEY else {}

    result = []
    total = len(matches)

    for i, match in enumerate(matches, 1):
        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        match_id = match.get("id")

        if not home_id or not away_id or not match_id:
            progress.advance(task)
            continue

        home_name = home_team.get("name", "Home")
        away_name = away_team.get("name", "Away")

        step(f"[{i}/{total}] {home_name} vs {away_name}")

        # ── Team histories + H2H ────────────────────────────────────────
        home_hist_raw = fetcher.get_team_matches(home_id, use_cache=use_cache, league=league_code)
        away_hist_raw = fetcher.get_team_matches(away_id, use_cache=use_cache, league=league_code)
        h2h_raw       = fetcher.get_head2head(match_id, use_cache=use_cache)

        # E.3: Merge fresh API window with full SQLite history (up to 76 matches).
        # This gives the model up to two seasons of data instead of half a season.
        home_api_matches  = home_hist_raw.get("matches", []) if home_hist_raw else []
        away_api_matches  = away_hist_raw.get("matches", []) if away_hist_raw else []
        home_db_matches   = match_store.get_team_match_history(home_id, limit=76)
        away_db_matches   = match_store.get_team_match_history(away_id, limit=76)
        home_merged       = {"matches": match_store.merge_match_history(home_api_matches, home_db_matches)}
        away_merged       = {"matches": match_store.merge_match_history(away_api_matches, away_db_matches)}

        home_history = analyzer.parse_team_history(home_merged, home_id, strength_factors, league=league_code)
        away_history = analyzer.parse_team_history(away_merged, away_id, strength_factors, league=league_code)
        h2h          = analyzer.parse_h2h(h2h_raw, home_team_id=home_id)

        if home_id not in standings:
            warn_log.fallback("home team missing from standings", "_default_standing", league=league_code, match_id=str(match_id))
        if away_id not in standings:
            warn_log.fallback("away team missing from standings", "_default_standing", league=league_code, match_id=str(match_id))
        home_standing = {**(standings.get(home_id) or _default_standing()), "id": home_id, "league": league_code}
        away_standing = {**(standings.get(away_id) or _default_standing()), "id": away_id, "league": league_code}

        # ── Home/Away split strength factors (Tier 1 #2) ────────────────
        # Recompute strength factors with split histories for these two teams.
        # This enriches home_attack/away_attack/home_defence/away_defence
        # indices so the xG model uses venue-specific performance.
        team_histories = {home_id: home_history, away_id: away_history}
        strength_factors_fx = analyzer._compute_strength_factors(standings, team_histories)

        # ── Fatigue check — derive last match date from already-fetched history ──
        _home_matches = home_merged.get("matches", [])
        _away_matches = away_merged.get("matches", [])
        home_last = _home_matches[0].get("utcDate") if _home_matches else None
        away_last = _away_matches[0].get("utcDate") if _away_matches else None

        # ── Knockout stage detection ────────────────────────────────────
        # football-data.org returns a 'stage' field on each match.
        # Knockout legs need suppressed goals and reduced Over/BTTS confidence.
        match_stage = match.get("stage", "")
        is_knockout = match_stage in CL_KNOCKOUT_STAGES

        # ── Referee tendency ───────────────────────────────────────────
        # Extract the main referee name from the fixture and compute a
        # goals-per-game multiplier from historical matches they officiated.
        ref_name = ""
        for ref in match.get("referees", []):
            if ref.get("type", "").upper() == "REFEREE":
                ref_name = ref.get("name", "")
                break
        league_avg_gpg = (strength_factors.get("_league_avg_scored", 1.30) * 2
                          if strength_factors else 2.6)
        referee_factor = analyzer.compute_referee_factor(
            ref_name, home_hist_raw, away_hist_raw,
            league_avg_gpg=league_avg_gpg,
        )

        # ── Probabilities ───────────────────────────────────────────────
        total_teams_n = len(standings) or 20
        probs = analyzer.compute_match_probabilities(
            league_code, home_standing, away_standing,
            home_history, away_history, h2h,
            strength_factors=strength_factors_fx,
            home_last_match_date=home_last,
            away_last_match_date=away_last,
            is_knockout=is_knockout,
            referee_factor=referee_factor,
            total_teams=total_teams_n,
            fixture_date=match.get("utcDate", ""),
        )

        # ── Live odds lookup for this fixture ───────────────────────────
        fx_odds = odds_fetcher.lookup_odds(odds_map, home_name, away_name)

        # ── Market evaluation — value-first sweep ───────────────────────
        # Find the single best-value pick for this fixture: the market where
        # model edge vs bookmaker odds is highest. Falls back to best
        # model_prob pick (flagged as unverified) when no odds are available.
        _best = markets.best_value_pick(
            probs,
            fx_odds,
            expected_total=probs.get("expected_total"),
            league=league_code,
        )
        all_picks = []
        if _best:
            pick = _best["pick"]
            pick["verified_edge"] = _best["verified_edge"]
            all_picks = [pick]

        # ── Context features for ML calibration + logging ──────────────
        home_pos_raw  = home_standing.get("position", 10)
        away_pos_raw  = away_standing.get("position", 10)
        # Normalise position: 1.0 = top of table, 0.0 = bottom
        home_pos_norm = 1.0 - (home_pos_raw - 1) / max(total_teams_n - 1, 1)
        away_pos_norm = 1.0 - (away_pos_raw - 1) / max(total_teams_n - 1, 1)
        home_form_v   = home_history.get("home_form_score") or home_standing.get("form_score", 0.5)
        away_form_v   = away_history.get("away_form_score") or away_standing.get("form_score", 0.5)
        form_adv_v    = home_form_v - away_form_v
        exp_total_v   = (probs.get("expected_home_goals", 0) or 0) + (probs.get("expected_away_goals", 0) or 0)

        # Attach context to every pick for downstream logging
        for pick in all_picks:
            pick["home_position"]  = home_pos_norm
            pick["away_position"]  = away_pos_norm
            pick["form_adv"]       = form_adv_v
            pick["expected_total"] = exp_total_v

        # ── ML calibration ─────────────────────────────────────────────
        cal = ml_calibrator.get_calibrator()
        for pick in all_picks:
            cal_extra = {
                "edge":           pick.get("edge"),
                "home_position":  pick["home_position"],
                "away_position":  pick["away_position"],
                "form_adv":       pick["form_adv"],
                "expected_total": pick["expected_total"],
                "odds_taken":     pick.get("odds"),
            }
            pick["model_prob"] = cal.calibrate(
                pick["model_prob"], pick["market"], league_code, pick["pick"],
                extra=cal_extra,
            )
            seg_data = cal.segment_uncertainty(pick["market"], pick["pick"], league_code)
            if seg_data:
                rate, n = seg_data
                pick["uncertainty"] = ml_calibrator.calibration_uncertainty(rate, n)
            else:
                pick["uncertainty"] = None

        # ── Refresh edge from calibrated probability ────────────────────
        # Edge was first computed in the market evaluators using the raw
        # model_prob. After calibration shifts the probability, the displayed
        # edge, post-cal filter, and Kelly stake math would diverge unless we
        # recompute here, BEFORE the filter runs.
        for pick in all_picks:
            o = pick.get("odds")
            if o is not None and o > 1.0:
                pick["edge"] = (pick["model_prob"] - 1.0 / o) * 100

        # ── Post-calibration filter ─────────────────────────────────────
        # Two checks for a pick to survive:
        #   1. model_prob ≥ market floor (no calibration nuke)
        #   2. for verified-edge picks (real bookmaker odds present), edge
        #      must STILL be positive after the calibration-driven refresh.
        #      Otherwise the bet is no longer a value bet and shouldn't be
        #      shown as a single — even if the original pre-calibration edge
        #      was positive when best_value_pick chose it.
        _POSTCAL_FLOOR = {
            **markets.MIN_PROB,
            "BTTS Yes": 0.62,
            "BTTS No":  0.55,
            "1X (Home or Draw)":  0.60,
            "X2 (Draw or Away)":  0.60,
            "12 (Home or Away)":  0.60,
        }
        _COMBO_FLOOR = 0.30
        filtered = []
        for p in all_picks:
            if p.get("verified_edge"):
                floor = markets._LOW_FLOOR
                # Edge sanity: calibration may have erased the value.
                if (p.get("edge") or 0) <= 0:
                    continue
            elif p["market"] == "Combo":
                floor = _COMBO_FLOOR
            else:
                floor = _POSTCAL_FLOOR.get(p["pick"], _COMBO_FLOOR)
            if p["model_prob"] >= floor:
                filtered.append(p)
        all_picks = filtered

        # ── Stake sizing (quarter Kelly) ────────────────────────────────
        # Computed after the filter so stake recommendations always agree
        # with the (now-positive) calibrated edge.
        for pick in all_picks:
            pick["stake_units"] = staking.compute_stake_units(
                pick.get("model_prob"), pick.get("odds"),
            )

        # ── Reason strings ──────────────────────────────────────────────
        for pick in all_picks:
            pick["reason"] = analyzer.build_reason(
                pick["pick"], pick["market"], probs,
                home_name, away_name,
                home_standing, away_standing, h2h,
            )

        all_picks.sort(key=_pick_sort_key, reverse=True)

        # Accumulator candidates: short-odds high-confidence picks usable as legs
        # in cross-fixture parlays. Pool is built per fixture; combination across
        # fixtures happens once after every league is processed.
        acca_candidates = markets.collect_acca_candidates(probs, fx_odds, league=league_code)

        result.append({
            "league":    league_code,
            "match_id":  match_id,
            "home_name": home_name,
            "away_name": away_name,
            "utc_date":  match.get("utcDate", ""),
            "probs":     probs,
            "picks":     all_picks,
            "acca_candidates": acca_candidates,
        })

        progress.advance(task)

    return result


# ---------------------------------------------------------------------------
# Menu helpers
# ---------------------------------------------------------------------------

def _menu_clear():
    os.system("cls" if sys.platform == "win32" else "clear")


def _menu_header():
    status.print()
    status.print(Rule("[bold yellow]Football Tipster[/bold yellow]", style="yellow"))
    now = datetime.now().strftime("%A %d %B %Y  %H:%M")
    status.print(f"  [dim]{now}[/dim]")
    status.print()


def _menu_prompt(n_options: int) -> str:
    valid = {str(i) for i in range(n_options + 1)}
    while True:
        try:
            choice = input("  Enter choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            status.print()
            return "0"
        if choice in valid:
            return choice
        status.print(f"  [red]Please enter a number between 0 and {n_options}.[/red]")


def _menu_quick_stats():
    """Render compact ROI line. Shows current-model ROI when available so the
    user sees the live cohort's performance, not the lifetime average pulled
    down by retired pre-threshold picks."""
    current = logger.compute_roi_summary(model_version=logger.MODEL_VERSION)
    lifetime = logger.compute_roi_summary()
    parts = []
    if current:
        color = "green" if current["roi_pct"] >= 0 else "red"
        sign  = "+" if current["roi_pct"] >= 0 else ""
        parts.append(
            f"[dim]ROI ({logger.MODEL_VERSION}):[/dim] [{color}]{sign}{current['roi_pct']:.1f}%[/{color}]"
            f" [dim]({current['wins']}W/{current['losses']}L, {current['total']} bets)[/dim]"
        )
    elif lifetime:
        color = "green" if lifetime["roi_pct"] >= 0 else "red"
        sign  = "+" if lifetime["roi_pct"] >= 0 else ""
        parts.append(
            f"[dim]ROI (lifetime):[/dim] [{color}]{sign}{lifetime['roi_pct']:.1f}%[/{color}]"
            f" [dim]({lifetime['wins']}W/{lifetime['losses']}L, {lifetime['total']} bets)[/dim]"
        )
    else:
        parts.append("[dim]No settled bets yet[/dim]")
    status.print("  " + "   ".join(parts))
    status.print()


def _show_bet_history():
    _menu_clear()
    _menu_header()

    rows = logger._load_rows()
    if not rows:
        status.print("  [yellow]No bets logged yet. Generate today's picks first.[/yellow]")
        status.print()
        input("  Press Enter to return...")
        return

    settled_rows   = [r for r in rows if r.get("result", "").strip() in ("W", "L")]
    unsettled_rows = [r for r in rows if not r.get("result", "").strip()]

    roi_lifetime = logger.compute_roi_summary()
    roi_current  = logger.compute_roi_summary(model_version=logger.MODEL_VERSION)
    summary_lines = []
    summary_lines.append(
        f"[bold]Total logged:[/bold] {len(rows)}   "
        f"[green]Settled: {len(settled_rows)}[/green]   "
        f"[yellow]Pending: {len(unsettled_rows)}[/yellow]"
    )

    def _roi_line(label, roi):
        if not roi:
            return f"[dim]{label}: not enough settled bets yet (need 5+)[/dim]"
        color = "green" if roi["roi_pct"] >= 0 else "red"
        sign  = "+" if roi["roi_pct"] >= 0 else ""
        return (
            f"[bold]{label}:[/bold] {roi['wins']}W / {roi['losses']}L   "
            f"ROI [{color}]{sign}{roi['roi_pct']:.1f}%[/{color}]   "
            f"[dim]({roi['total']} bets)[/dim]"
        )

    summary_lines.append(_roi_line(f"Current model ({logger.MODEL_VERSION})", roi_current))
    summary_lines.append(_roi_line("Lifetime (all model versions)", roi_lifetime))
    status.print(Panel("\n".join(summary_lines), title="Performance", border_style="yellow"))
    status.print()

    # Recent bets table (last 20)
    recent = sorted(rows, key=lambda r: r.get("date", ""), reverse=True)[:20]
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Date",   style="dim", width=11)
    table.add_column("Match",  width=28)
    table.add_column("Lg",     width=4)
    table.add_column("Market", width=11)
    table.add_column("Pick",   width=22)
    table.add_column("Prob",   justify="right", width=6)
    table.add_column("Odds",   justify="right", width=6)
    table.add_column("Result", justify="center", width=7)
    table.add_column("ROI",    justify="right", width=7)

    for r in recent:
        result = r.get("result", "").strip()
        result_str = {"W": "[green]W[/green]", "L": "[red]L[/red]"}.get(result, "[yellow]?[/yellow]")
        roi_val = r.get("roi", "").strip()
        if roi_val:
            try:
                rv = float(roi_val)
                roi_str = f"[green]+{rv:.2f}[/green]" if rv >= 0 else f"[red]{rv:.2f}[/red]"
            except ValueError:
                roi_str = roi_val
        else:
            roi_str = "[dim]-[/dim]"
        prob_str = _safe_prob_pct(r.get("model_prob"))
        odds_str = r.get("odds_taken", "").strip() or "[dim]-[/dim]"
        match_str = f"{r.get('home','')[:13]} v {r.get('away','')[:13]}"
        table.add_row(
            r.get("date", ""), match_str, r.get("league", ""),
            r.get("market", ""), r.get("pick", ""),
            prob_str, odds_str, result_str, roi_str,
        )

    status.print(f"  [bold]Recent bets[/bold] [dim](last {len(recent)})[/dim]")
    status.print()
    status.print(table)
    status.print()

    status.print("  [cyan]1[/cyan]  Market breakdown (win rate per market/pick)")
    status.print("  [cyan]2[/cyan]  Pending bets only")
    status.print("  [cyan]0[/cyan]  Back")
    status.print()
    choice = _menu_prompt(2)
    if choice == "1":
        _show_market_breakdown(settled_rows)
        input("\n  Press Enter to return...")
    elif choice == "2":
        _show_pending_bets(unsettled_rows)
        input("\n  Press Enter to return...")


def _show_market_breakdown(settled_rows: list):
    by_market = defaultdict(list)
    for r in settled_rows:
        by_market[r.get("market", "?")].append(r)
    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("Market", width=14)
    table.add_column("Pick",   width=22)
    table.add_column("W",      justify="right", width=5)
    table.add_column("L",      justify="right", width=5)
    table.add_column("WR%",    justify="right", width=6)
    for market in ["1X2", "Over/Under", "BTTS", "Double Chance", "Combo"]:
        if market not in by_market:
            continue
        by_pick = defaultdict(list)
        for r in by_market[market]:
            by_pick[r.get("pick", "?")].append(r)
        for pick, recs in sorted(by_pick.items()):
            w = sum(1 for r in recs if r.get("result", "").strip() == "W")
            l = sum(1 for r in recs if r.get("result", "").strip() == "L")
            n = w + l
            if n == 0:
                continue
            wr = w / n * 100
            color = "green" if wr >= 60 else ("red" if wr < 45 else "yellow")
            table.add_row(market, pick, str(w), str(l), f"[{color}]{wr:.0f}%[/{color}]")
    status.print()
    status.print(table)


def _show_pending_bets(unsettled_rows: list):
    if not unsettled_rows:
        status.print("\n  [green]No pending bets — all settled.[/green]")
        return
    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("Date",  width=11)
    table.add_column("Match", width=28)
    table.add_column("Lg",    width=4)
    table.add_column("Pick",  width=22)
    table.add_column("Prob",  justify="right", width=6)
    status.print()
    for r in unsettled_rows:
        prob_str = _safe_prob_pct(r.get("model_prob"))
        match_str = f"{r.get('home','')[:13]} v {r.get('away','')[:13]}"
        table.add_row(r.get("date", ""), match_str, r.get("league", ""), r.get("pick", ""), prob_str)
    status.print(table)


def _settle_menu():
    _menu_clear()
    _menu_header()
    status.print("  [bold]Settle pending bets[/bold]")
    status.print("  [dim]Fetches results for all unsettled picks from the API.[/dim]")
    status.print()
    rows = logger._load_rows()
    unsettled = [r for r in rows if not r.get("result", "").strip()]
    if not unsettled:
        status.print("  [green]All bets are already settled — nothing to do.[/green]")
        status.print()
        input("  Press Enter to return...")
        return
    status.print(f"  [yellow]{len(unsettled)} pending bet(s) will be checked.[/yellow]")
    status.print()
    settled_count, failed_count = logger.settle_bets(console=status)
    status.print()
    if settled_count:
        status.print(f"  [green]Settled: {settled_count}[/green]")
    if failed_count:
        status.print(f"  [yellow]Still pending (result not out yet): {failed_count}[/yellow]")
    if not settled_count and not failed_count:
        status.print("  [dim]Nothing changed.[/dim]")
    status.print()
    input("  Press Enter to return...")


def _backtest_menu():
    _menu_clear()
    _menu_header()
    status.print("  [bold]Backtest[/bold]")
    status.print("  [dim]Replays a full historical season through the model (~2-3 min).[/dim]")
    status.print()
    status.print("  [cyan]1[/cyan]  Quick  — Premier League 2023 only")
    status.print("  [cyan]2[/cyan]  Full   — PL + SA + BL1 + FL1, 2023")
    status.print("  [cyan]3[/cyan]  Custom — type your own league codes")
    status.print("  [cyan]0[/cyan]  Back")
    status.print()
    choice = _menu_prompt(3)
    if choice == "0":
        return
    if choice == "1":
        leagues = ["PL"]
    elif choice == "2":
        leagues = ["PL", "SA", "BL1", "FL1"]
    else:
        raw = input("  League codes (space-separated, e.g. PL BL1): ").strip().upper()
        leagues = raw.split() if raw else ["PL"]
    status.print()
    subprocess.run([sys.executable, "-m", "scripts.backtest", "--leagues"] + leagues + ["--seasons", "2023"])
    status.print()
    input("  Press Enter to return...")


def _warnings_menu():
    _menu_clear()
    _menu_header()

    # ── Data freshness check (§5.4) ─────────────────────────────────────────
    status.print("  [bold]Data freshness[/bold]  [dim](standings cache age per league)[/dim]")
    status.print()
    STALE_WARN  = 6.0   # hours — yellow
    STALE_ALERT = 20.0  # hours — red (older than any typical matchday cycle)
    any_stale = False
    for lg in LEAGUES:
        age = get_cache_age_hours(f"v{CACHE_VERSION}_standings_{lg}")
        if age is None:
            status.print(f"  [cyan]{lg:<4}[/cyan]  [dim]not cached yet[/dim]")
        elif age >= STALE_ALERT:
            status.print(f"  [cyan]{lg:<4}[/cyan]  [red]{age:.1f}h old — STALE, run today's picks to refresh[/red]")
            any_stale = True
        elif age >= STALE_WARN:
            status.print(f"  [cyan]{lg:<4}[/cyan]  [yellow]{age:.1f}h old[/yellow]")
        else:
            status.print(f"  [cyan]{lg:<4}[/cyan]  [green]{age:.1f}h old[/green]")
    if any_stale:
        status.print()
        status.print("  [red]⚠  One or more leagues have stale standings (>20h). Run option 1 to refresh.[/red]")
    status.print()

    # ── Fallback warnings log ───────────────────────────────────────────────
    status.print("  [bold]Model fallback warnings[/bold]  [dim]Logged whenever the model fell back to a default[/dim]")
    status.print()
    log_path = "warnings.log"
    if not os.path.exists(log_path):
        status.print("  [green]No warnings logged — data pipeline looks clean.[/green]")
        status.print()
        input("  Press Enter to return...")
        return
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        status.print("  [green]warnings.log is empty.[/green]")
    else:
        recent = lines[-30:]
        status.print(f"  [dim]Showing last {len(recent)} of {len(lines)} warning(s)[/dim]")
        status.print()
        for line in recent:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) >= 5:
                date, league, _, reason, fallback = parts[0], parts[1], parts[2], parts[3], parts[4]
                status.print(f"  [dim]{date}[/dim]  [cyan]{league or '-'}[/cyan]  [yellow]{reason}[/yellow]  [dim]-> {fallback}[/dim]")
            else:
                status.print(f"  [dim]{line.strip()}[/dim]")
    status.print()
    input("  Press Enter to return...")


# ---------------------------------------------------------------------------
# Analysis pipeline (called from menu option 1)
# ---------------------------------------------------------------------------

def _run_analysis(leagues: list, use_cache: bool, min_edge: float,
                  mode_full: bool, mode_over25: bool,
                  output_file: str = None, update_baselines: bool = False,
                  interactive: bool = False):
    today_str    = datetime.now().strftime("%Y-%m-%d")
    date_display = datetime.now().strftime("%A %d %B %Y")

    match_store.init_db()
    evict_stale_cache()

    cal = ml_calibrator.get_calibrator()
    cal_label = (
        f"LR ({cal.sample_count} bets)" if cal.uses_lr
        else (f"stats ({cal.sample_count} bets)" if cal.is_active else "off (no history)")
    )
    lr_leagues = cal.league_lr_leagues
    league_lr_label = (
        " ".join(f"{lg}({cal.league_lr_sample_count(lg)})" for lg in sorted(lr_leagues))
        if lr_leagues else "none yet"
    )

    settled, failed = logger.settle_bets(console=status)
    if settled:
        ml_calibrator.reset_calibrator()
        cal = ml_calibrator.get_calibrator()
        cal_label = (
            f"LR ({cal.sample_count} bets)" if cal.uses_lr
            else (f"stats ({cal.sample_count} bets)" if cal.is_active else "off (no history)")
        )
        lr_leagues = cal.league_lr_leagues
        league_lr_label = (
            " ".join(f"{lg}({cal.league_lr_sample_count(lg)})" for lg in sorted(lr_leagues))
            if lr_leagues else "none yet"
        )

    status.print()
    status.print(Rule("[bold yellow]Football Tipster[/bold yellow]", style="yellow"))
    quota = odds_fetcher.get_quota_remaining()
    odds_label = (f"on ({quota} credits left)" if quota is not None else "on") if ODDS_API_KEY else "off (no key)"

    status.print(
        Text.assemble(
            ("  Leagues: ", "dim"), (", ".join(leagues), "cyan"),
            ("   Date: ", "dim"), (date_display, "white"),
            ("   Min edge: ", "dim"), (f"{min_edge:.0f}%", "white"),
            ("   Cache: ", "dim"), ("off" if not use_cache else "on", "white"),
            ("   ML: ", "dim"), (cal_label, "magenta"),
            ("   Per-league LR: ", "dim"), (league_lr_label, "magenta"),
            ("   Odds: ", "dim"), (odds_label, "cyan"),
        )
    )

    if settled or failed:
        parts = []
        if settled:
            parts.append(f"[green]{settled} settled[/green]")
        if failed:
            parts.append(f"[yellow]{failed} still pending[/yellow]")
        status.print(f"  [dim]Bet results:[/dim]  {',  '.join(parts)}")

    # Prefer the current-model cohort; fall back to lifetime if too few bets exist yet.
    roi_data = (logger.compute_roi_summary(model_version=logger.MODEL_VERSION)
                or logger.compute_roi_summary())
    if roi_data:
        roi_color = "green" if roi_data["roi_pct"] >= 0 else "red"
        roi_sign  = "+" if roi_data["roi_pct"] >= 0 else ""
        status.print(
            f"  [dim]ROI ({roi_data['total']} bets):[/dim]"
            f"  [{roi_color}]{roi_sign}{roi_data['roi_pct']:.1f}%[/{roi_color}]"
            f"  [dim]({roi_data['wins']}W / {roi_data['losses']}L)[/dim]"
        )

    if update_baselines:
        for change in auto_update_baselines():
            status.print(f"  [bold cyan]BASELINE UPDATED:[/bold cyan] [cyan]{change}[/cyan]")

    if cal.is_active:
        rows = cal.summary()
        if rows:
            status.print("  [dim]Market history:[/dim]", end="")
            for market, rate, n in rows:
                color = "green" if rate >= 0.55 else ("red" if rate <= 0.45 else "yellow")
                status.print(f"  [{color}]{market} {rate * 100:.0f}% ({n})[/{color}]", end="")
            status.print()
    status.print()

    all_fixtures = []
    league_summary = []

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(bar_width=20), MofNCompleteColumn(),
        console=status, transient=False,
    ) as progress:
        for league_code in leagues:
            task = progress.add_task(f" [cyan]{league_code}[/cyan]  starting ...", total=None)
            try:
                fixtures = process_league(league_code, today_str, use_cache, progress, task)
                all_fixtures.extend(fixtures)
                picks_count = sum(len(f["picks"]) for f in fixtures)
                league_summary.append((league_code, len(fixtures), picks_count, None))
                progress.update(
                    task,
                    description=(
                        f" [green]✓[/green] [cyan]{league_code}[/cyan]"
                        f"  {len(fixtures)} fixture(s)   {picks_count} pick(s)"
                    ),
                    completed=progress.tasks[task].total or 1,
                    total=progress.tasks[task].total or 1,
                )
            except Exception as exc:
                league_summary.append((league_code, 0, 0, str(exc)))
                progress.update(
                    task,
                    description=f" [red]✗[/red] [cyan]{league_code}[/cyan]  [red]{exc}[/red]",
                    completed=1, total=1,
                )

    status.print()
    status.print(Rule("Summary", style="dim"))
    for code, fx, pk, err in league_summary:
        if err:
            status.print(f"  [red]✗[/red] [cyan]{code}[/cyan]  [red]{err}[/red]")
        else:
            status.print(f"  [green]✓[/green] [cyan]{code}[/cyan]  [dim]{fx} fixture(s)   {pk} pick(s)[/dim]")
    status.print()

    if not all_fixtures:
        status.print(f"[yellow]No unplayed fixtures found for today ({today_str}). Check back on a matchday.[/yellow]")
        return

    # Only verified-edge singles count as value picks. Picks without real
    # bookmaker odds are dropped from the singles surface — their useful form
    # is as an acca leg (Tier 2), not as a standalone bet.
    value_picks = []
    for fixture in all_fixtures:
        if not fixture["picks"]:
            continue
        chosen = fixture["picks"][0]
        if not chosen.get("verified_edge"):
            continue
        value_picks.append({
            **chosen,
            "match_id": fixture["match_id"],
            "home":     fixture["home_name"],
            "away":     fixture["away_name"],
            "league":   fixture["league"],
            "utc_date": fixture.get("utc_date", ""),
        })

    # Highest edge first — that is the ranking that matters
    value_picks.sort(key=lambda p: p.get("edge", 0), reverse=True)

    # Build cross-fixture accumulators from the qualifying short-odds picks
    # collected per fixture above. These don't compete with the single picks
    # — they're an additional bet type the user can choose to stake.
    accas = markets.build_cross_fixture_accas(all_fixtures)

    # Stake-size verified accas (real odds on every leg). MODEL accas use
    # inferred prices so we can't recommend a stake reliably.
    for acca in accas:
        if acca.get("verified_edge"):
            acca["stake_units"] = staking.compute_stake_units(
                acca["joint_prob"], acca["joint_odds"], acca["edge"],
            )
        else:
            acca["stake_units"] = None

    acca_log_rows = _accas_to_log_rows(accas)

    if mode_full:
        # No-odds singles are deliberately not rendered or logged — they're not
        # value bets. Useful no-odds picks surface only as acca legs in Tier 2.
        coupon.render_coupon(value_picks, accas=accas, date_str=date_display)
        logger.log_bets(value_picks + acca_log_rows)
        from telegram_notifier import build_telegram_message, send_telegram_message
        msg = build_telegram_message(value_picks, accas)
        send_telegram_message(msg)

        # Drift detector — surface any (market, pick) buckets where the model's
        # average predicted probability has drifted from the actual win rate.
        # When run inside the interactive menu (interactive=True), offer to fit
        # and save an isotonic correction for each actionable bucket.
        drift_rows = drift.compute_drift()
        coupon.render_drift_block(drift_rows)
        if interactive:
            _maybe_apply_corrections(drift_rows)

    if mode_over25:
        coupon.render_over25(all_fixtures, date_str=date_display)

    if output_file:
        text = coupon.export_text()
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(text)
        status.print(f"[dim]Coupon saved to {output_file}[/dim]")

    n_calls = fetcher.get_api_call_count()
    fetcher.reset_api_call_count()
    if n_calls >= fetcher._API_CALL_WARN_THRESHOLD:
        status.print(
            f"[yellow]  Warning: {n_calls} live API calls this run "
            f"(>= {fetcher._API_CALL_WARN_THRESHOLD} — approaching free-tier daily limit).[/yellow]"
        )
    else:
        status.print(f"[dim]  API calls this run: {n_calls}[/dim]")


def _picks_menu(leagues: list, use_cache: bool, min_edge: float):
    _menu_clear()
    _menu_header()
    status.print("  [bold]Today's picks[/bold]\n")
    status.print("  [cyan]1[/cyan]  Best pick per match  [dim](recommended)[/dim]")
    status.print("  [cyan]2[/cyan]  Over 2.5 goals table only")
    status.print("  [cyan]3[/cyan]  Both")
    status.print("  [cyan]0[/cyan]  Back")
    status.print()
    choice = _menu_prompt(3)
    if choice == "0":
        return
    mode_full   = choice in ("1", "3")
    mode_over25 = choice in ("2", "3")
    status.print()
    _run_analysis(leagues, use_cache, min_edge, mode_full, mode_over25, interactive=True)
    status.print()
    input("  Press Enter to return to the menu...")


def _bankroll_menu():
    """Set the user's bankroll. Persisted to data/user_settings.json."""
    _menu_clear()
    _menu_header()
    status.print("  [bold]Bankroll[/bold]")
    status.print()
    status.print(
        f"  [dim]Current bankroll:[/dim] [bold]€{staking.BANKROLL_EUR:.2f}[/bold]   "
        f"[dim]1 unit ({staking.UNIT_PCT*100:.1f}%) = €{staking.UNIT_EUR:.2f}[/dim]"
    )
    status.print(
        f"  [dim]Stakes scale linearly with bankroll. Quarter-Kelly capped at "
        f"{staking.MAX_STAKE_UNITS} units per pick.[/dim]"
    )
    status.print()
    status.print("  Enter a new bankroll in euros, or press Enter to keep current.")
    status.print("  [dim](Press 0 + Enter to return to the menu.)[/dim]")
    status.print()
    try:
        raw = input("  New bankroll €: ").strip()
    except (EOFError, KeyboardInterrupt):
        status.print()
        return
    if not raw or raw == "0":
        return
    raw = raw.replace(",", ".")
    try:
        staking.set_bankroll(raw)
    except ValueError as exc:
        status.print(f"  [red]{exc}[/red]")
        status.print()
        input("  Press Enter to return to the menu...")
        return
    status.print(
        f"  [green]Saved.[/green] Bankroll now [bold]€{staking.BANKROLL_EUR:.2f}[/bold]   "
        f"1 unit = €{staking.UNIT_EUR:.2f}"
    )
    status.print()
    input("  Press Enter to return to the menu...")


def _calibration_menu():
    _menu_clear()
    _menu_header()
    status.print("  [bold]Calibration Analysis[/bold]")
    status.print("  [dim]Predicted vs actual win rate per pick type and probability bucket.[/dim]")
    status.print()
    status.print("  [cyan]1[/cyan]  All markets (summary + detail)")
    status.print("  [cyan]2[/cyan]  Per-league breakdown")
    status.print("  [cyan]3[/cyan]  Single pick type (e.g. 'Over 2.5')")
    status.print("  [cyan]0[/cyan]  Back")
    status.print()
    choice = _menu_prompt(3)
    if choice == "0":
        return
    status.print()
    if choice == "1":
        subprocess.run([sys.executable, "-m", "scripts.calibration_curves"])
    elif choice == "2":
        subprocess.run([sys.executable, "-m", "scripts.calibration_curves", "--by-league"])
    elif choice == "3":
        pick = input("  Pick type (e.g. 'Over 2.5', 'Home Win', 'Under 2.5'): ").strip()
        if pick:
            subprocess.run([sys.executable, "-m", "scripts.calibration_curves", "--pick", pick])
    status.print()
    input("  Press Enter to return...")


# ---------------------------------------------------------------------------
# Main interactive menu + CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Football Betting Tipster — statistical coupon generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--leagues", nargs="+", default=LEAGUES, metavar="CODE",
                        help="League codes to analyse (e.g. PL BL1 SA)")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE, dest="min_edge",
                        help="Minimum edge %% required for a 1X2 pick")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="Bypass local JSON cache and force fresh API calls")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Save the coupon as plain text to FILE")
    parser.add_argument("--over25", action="store_true", dest="over25",
                        help="Show the top 6 fixtures most likely to produce Over 2.5 goals")
    parser.add_argument("--update-baselines", action="store_true", dest="update_baselines",
                        help="Re-compute BASELINES from settled bet history and update config.py")
    args = parser.parse_args()

    use_cache = not args.no_cache
    leagues   = [c.upper() for c in args.leagues]

    # ── Scripted / non-interactive path ─────────────────────────────────
    # Any explicit flag bypasses the menu and runs straight through.
    if args.over25 or args.output or args.update_baselines or not sys.stdin.isatty():
        _run_analysis(
            leagues, use_cache, args.min_edge,
            mode_full=not args.over25,
            mode_over25=args.over25,
            output_file=args.output,
            update_baselines=args.update_baselines,
        )
        return

    # ── Interactive menu loop ────────────────────────────────────────────
    while True:
        _menu_clear()
        _menu_header()
        _menu_quick_stats()
        status.print("  [bold]What would you like to do?[/bold]")
        status.print()
        status.print("  [cyan]1[/cyan]  Today's picks     [dim]Value picks + best available per fixture[/dim]")
        status.print("  [cyan]2[/cyan]  Over 2.5 goals    [dim]Top fixtures most likely to have 3+ goals[/dim]")
        status.print("  [cyan]3[/cyan]  Bet history       [dim]View all logged picks and win rate[/dim]")
        status.print("  [cyan]4[/cyan]  Settle bets       [dim]Fetch results for pending picks[/dim]")
        status.print("  [cyan]5[/cyan]  Run backtest      [dim]Replay a historical season through the model[/dim]")
        status.print("  [cyan]6[/cyan]  Data warnings     [dim]Check if model fell back to defaults[/dim]")
        status.print("  [cyan]7[/cyan]  Calibration       [dim]Predicted vs actual WR per market[/dim]")
        status.print(f"  [cyan]8[/cyan]  Set bankroll      [dim]Currently €{staking.BANKROLL_EUR:.2f} · 1 unit = €{staking.UNIT_EUR:.2f}[/dim]")
        status.print()
        status.print("  [cyan]0[/cyan]  Exit")
        status.print()
        choice = _menu_prompt(8)
        if choice == "0":
            status.print("\n  [dim]Goodbye.[/dim]\n")
            sys.exit(0)
        elif choice == "1":
            _picks_menu(leagues, use_cache, args.min_edge)
        elif choice == "2":
            _run_analysis(leagues, use_cache, args.min_edge,
                          mode_full=False, mode_over25=True)
            input("  Press Enter to return to the menu...")
        elif choice == "3":
            _show_bet_history()
        elif choice == "4":
            _settle_menu()
        elif choice == "5":
            _backtest_menu()
        elif choice == "6":
            _warnings_menu()
        elif choice == "7":
            _calibration_menu()
        elif choice == "8":
            _bankroll_menu()


if __name__ == "__main__":
    main()
