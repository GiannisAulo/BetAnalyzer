from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

import staking

console = Console(record=True, legacy_windows=False, width=140)

def _prob_color(prob):
    """Color band keyed on model_prob: green ≥ 0.70, yellow ≥ 0.60, white ≥ 0.50, dim below."""
    if prob >= 0.70:
        return "bold green"
    if prob >= 0.60:
        return "yellow"
    if prob >= 0.50:
        return "white"
    return "dim"


def _shorten(name, max_len=13):
    """Trim common suffixes then truncate to max_len."""
    for suffix in (" FC", " CF", " SC", " AC", " IF", " BV", " SV", " FK"):
        name = name.replace(suffix, "")
    name = name.strip()
    return name if len(name) <= max_len else name[:max_len - 1] + "."


_LEAGUE_NAMES = {
    "PL":  "Premier League",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
    "PPL": "Primeira Liga",
    "DED": "Eredivisie",
    "ELC": "Championship",
    "BSA": "Brasileirão",
}


def _build_picks_table(picks, show_edge=True):
    """Build a Rich Table for a list of picks, grouped by league."""
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),     # tightened from (0,2) so all columns fit in 140 chars
        show_edge=False,
    )
    table.add_column("Match",  width=40, no_wrap=True)
    table.add_column("Lg",     width=3,  justify="center")
    table.add_column("Pick",   width=22, no_wrap=True)
    table.add_column("Market", width=10, no_wrap=True)
    table.add_column("Conf",   width=9,  justify="right")
    table.add_column("Odds",   width=6,  justify="right")
    table.add_column("Edge",   width=7,  justify="right")
    table.add_column("Stake",  width=14, justify="right")   # widened so €X.XX never truncates

    # Sort by league, then by edge desc (value tier) / model_prob desc (prob tier)
    current_league = None
    for pick in sorted(picks, key=lambda p: (p["league"], -(p.get("edge") or 0), -p["model_prob"])):
        # League divider row
        if pick["league"] != current_league:
            current_league = pick["league"]
            league_name = _LEAGUE_NAMES.get(current_league, current_league)
            table.add_row(
                f"[bold dim]{league_name}[/bold dim]",
                "", "", "", "", "", "", "",
            )

        c = _prob_color(pick.get("model_prob", 0))
        match_str = f"{_shorten(pick['home'])} vs {_shorten(pick['away'])}"

        if pick.get("odds") is not None:
            odds_str = f"[{c}]{pick['odds']:.2f}[/{c}]"
        else:
            fair_odds = 1 / pick["model_prob"] if pick["model_prob"] > 0 else 0
            odds_str = f"[dim]{fair_odds:.2f}~[/dim]"

        if show_edge:
            edge = pick.get("edge")
            if edge is not None and edge > 0:
                edge_color = "green" if edge >= 5 else "yellow"
                edge_str = f"[{edge_color}]+{edge:.1f}%[/{edge_color}]"
            else:
                edge_str = "[dim]—[/dim]"
        else:
            edge_str = "[dim]—[/dim]"

        conf_pct = pick["model_prob"] * 100
        unc = pick.get("uncertainty")
        if unc is not None and unc * 100 > 3:
            conf_str = f"[{c}]{conf_pct:.0f}%±{unc*100:.0f}pp[/{c}]"
        else:
            conf_str = f"[{c}]{conf_pct:.0f}%[/{c}]"

        pick_name = pick["pick"]
        if pick.get("market") == "Combo" and " + " in pick_name:
            leg1, leg2 = pick_name.split(" + ", 1)
            pick_display = f"[{c}]{leg1}[/{c}]\n[{c}]+ {leg2}[/{c}]"
        else:
            pick_display = f"[{c}]{pick_name}[/{c}]"

        stake_u = pick.get("stake_units")
        if stake_u is not None:
            eur = staking.units_to_eur(stake_u)
            stake_str = f"[bold]{stake_u}u[/bold] [dim](€{eur:.2f})[/dim]"
        else:
            stake_str = "[dim]—[/dim]"

        table.add_row(
            match_str,
            pick["league"],
            pick_display,
            f"[dim]{pick['market']}[/dim]",
            conf_str,
            odds_str,
            edge_str,
            stake_str,
        )
        reason = pick.get("reason", "")
        if reason:
            table.add_row(f"[dim italic]{reason}[/dim italic]", "", "", "", "", "", "", "")

    return table


def _render_accas(accas):
    """Render the cross-fixture accumulator section.

    Each acca is a dict with keys: size, joint_odds, joint_prob, edge, legs.
    Each leg has: home, away, league, pick, market, model_prob, odds.
    """
    console.print(
        Rule("[bold cyan]SAFE ACCUMULATORS — LEG-STACKED VALUE[/bold cyan]", style="cyan")
    )
    console.print()
    console.print(
        "  [dim]Short-odds high-confidence picks combined into 2–3 leg accumulators "
        "where the combined edge becomes visible.[/dim]"
    )
    console.print()

    for idx, acca in enumerate(accas, 1):
        joint_prob_pct = acca["joint_prob"] * 100
        col = _prob_color(acca["joint_prob"])
        verified = acca.get("verified_edge", False)
        tag = "[green]VERIFIED[/green]" if verified else "[yellow]MODEL[/yellow]"
        edge_part = (
            f"  ·  Edge [green]+{acca['edge']:.1f}%[/green]"
            if verified else "  ·  [dim]check live odds[/dim]"
        )
        stake_u = acca.get("stake_units")
        if stake_u is not None:
            eur = staking.units_to_eur(stake_u)
            stake_part = f"  ·  Stake [bold]{stake_u}u[/bold] [dim](€{eur:.2f})[/dim]"
        else:
            stake_part = ""
        header = (
            f"[bold]Acca {idx}[/bold] {tag}  ·  {acca['size']} legs  ·  "
            f"Combined [{col}]{acca['joint_odds']:.2f}[/{col}]  ·  "
            f"Prob [{col}]{joint_prob_pct:.0f}%[/{col}]{edge_part}{stake_part}"
        )
        console.print(header)

        table = Table(
            box=box.SIMPLE, show_header=True, header_style="bold dim",
            padding=(0, 2), show_edge=False,
        )
        table.add_column("Match",  width=40, no_wrap=True)
        table.add_column("Lg",     width=4,  justify="center")
        table.add_column("Pick",   width=22, no_wrap=True)
        table.add_column("Market", width=10, no_wrap=True)
        table.add_column("Prob",   width=7,  justify="right")
        table.add_column("Odds",   width=7,  justify="right")
        for leg in acca["legs"]:
            leg_col   = _prob_color(leg["model_prob"])
            match_str = f"{_shorten(leg['home'])} vs {_shorten(leg['away'])}"
            # Tilde suffix marks an odds value derived from the model, not the bookmaker
            odds_disp = (
                f"[{leg_col}]{leg['odds']:.2f}~[/{leg_col}]"
                if leg.get("inferred_odds")
                else f"[{leg_col}]{leg['odds']:.2f}[/{leg_col}]"
            )
            table.add_row(
                match_str,
                leg["league"],
                f"[{leg_col}]{leg['pick']}[/{leg_col}]",
                f"[dim]{leg['market']}[/dim]",
                f"[{leg_col}]{leg['model_prob']*100:.0f}%[/{leg_col}]",
                odds_disp,
            )
        console.print(table)
        console.print()

    console.print(
        "  [dim]VERIFIED = every leg has real bookmaker odds; edge confirmed.  "
        "MODEL = at least one leg uses model fair odds (~); verify on the bookmaker before betting.[/dim]"
    )
    console.print()


def render_drift_block(drift_rows):
    """Render the per-market drift report (yellow/red only) at the foot of a coupon.

    drift_rows: list of dicts produced by drift.compute_drift(). Green rows are
    skipped — only actionable drift is surfaced to avoid noise.
    """
    actionable = [d for d in drift_rows if d["severity"] in ("yellow", "red")]
    if not actionable:
        return
    console.print(
        Rule("[bold magenta]MODEL CALIBRATION DRIFT[/bold magenta]", style="magenta")
    )
    console.print()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  padding=(0, 2), show_edge=False)
    table.add_column("Market",    width=14, no_wrap=True)
    table.add_column("Pick",      width=22, no_wrap=True)
    table.add_column("n",         width=4, justify="right")
    table.add_column("Predicted", width=10, justify="right")
    table.add_column("Actual",    width=8, justify="right")
    table.add_column("Gap",       width=8, justify="right")
    table.add_column("Severity",  width=10, justify="center")
    for d in actionable:
        sev_col = "red" if d["severity"] == "red" else "yellow"
        gap_sign = "+" if d["gap_pp"] >= 0 else ""
        direction = "(over-confident)" if d["gap_pp"] > 0 else "(under-confident)"
        table.add_row(
            d["market"], d["pick"], str(d["n"]),
            f"{d['predicted_pct']:.0f}%",
            f"{d['actual_wr_pct']:.0f}%",
            f"[{sev_col}]{gap_sign}{d['gap_pp']:.1f}pp[/{sev_col}]",
            f"[{sev_col}]{d['severity'].upper()}[/{sev_col}]",
        )
    console.print(table)
    console.print(
        "  [dim]Positive gap = model over-confident (predicts more wins than reality).  "
        "Negative gap = under-confident.[/dim]"
    )
    console.print()


def render_coupon(value_picks, _no_odds_unused=None, accas=None, date_str=None):
    """
    Render the value-only coupon. Two tiers, both genuine value bets:

    Tier 1 — VALUE PICKS: fixtures where the model found positive edge against
    real bookmaker odds. Sorted by edge descending. These are the actionable
    single-bet recommendations.

    Tier 2 — SAFE ACCUMULATORS: cross-fixture combos that stack short-odds
    high-confidence picks into 2–3 leg parlays with calculated joint edge.
    Verified (real odds on every leg) and MODEL (at least one inferred leg)
    are distinguished. Only shown when at least one qualifying acca exists.

    Note: solo picks without real bookmaker odds are deliberately NOT shown.
    Short-odds favourites (1.20–1.45) are not value bets as singles — they
    only become value when combined into accumulators (Tier 2 handles that).
    The `_no_odds_unused` parameter is retained for backward compatibility
    with existing callers but its contents are ignored.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%A %d %B %Y")

    console.print()
    console.print(Rule(f"[bold yellow]BETTING COUPON  —  {date_str}[/bold yellow]", style="yellow"))
    console.print(
        f"  [dim]Bankroll: €{staking.BANKROLL_EUR:.2f}  ·  "
        f"1 unit = €{staking.UNIT_EUR:.2f} ({staking.UNIT_PCT*100:.1f}%)  ·  "
        f"Stakes by quarter-Kelly (max {staking.MAX_STAKE_UNITS}u per pick)[/dim]"
    )
    console.print()

    # ── Tier 1: Value picks ──────────────────────────────────────────────────
    console.print(Rule("[bold green]VALUE PICKS — EDGE CONFIRMED[/bold green]", style="green"))
    console.print()
    if value_picks:
        console.print(_build_picks_table(value_picks, show_edge=True))
    else:
        console.print("[yellow]  No positive-edge picks found today.[/yellow]")
        console.print()

    # ── Tier 2: Safe accumulators ────────────────────────────────────────────
    if accas:
        _render_accas(accas)
    console.print()


def render_over25(fixtures, date_str=None, top_n=6, min_prob=0.45):
    """
    Render a special Over 2.5 goals table.

    Scans all fixtures, reads the model's over_2_5 probability from probs,
    sorts by that probability descending, and shows the top_n matches.

    min_prob: minimum Over 2.5 probability to be included (45% default —
    below this the prediction is not meaningful enough to show).

    Shows a clear message when fewer than top_n qualifying matches exist.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%A %d %B %Y")

    console.print()
    console.print(
        Rule(f"[bold cyan]OVER 2.5 GOALS — TOP PICKS  —  {date_str}[/bold cyan]", style="cyan")
    )
    console.print()

    # Build list of (fixture, over_2_5_prob, over_2_5_odds)
    candidates = []
    for fx in fixtures:
        probs = fx.get("probs", {})
        p25 = probs.get("over_2_5", 0.0)
        if p25 < min_prob:
            continue

        # Find if there's a real Over 2.5 odds pick in the fixture picks
        o25_odds = None
        for pick in fx.get("picks", []):
            if pick.get("pick") == "Over 2.5" and pick.get("odds") is not None:
                o25_odds = pick["odds"]
                break

        candidates.append({
            "home":       fx["home_name"],
            "away":       fx["away_name"],
            "league":     fx["league"],
            "over_2_5":   p25,
            "exp_home":   probs.get("expected_home_goals", 0),
            "exp_away":   probs.get("expected_away_goals", 0),
            "btts":       probs.get("btts_yes", 0),
            "o25_odds":   o25_odds,
        })

    # Sort by Over 2.5 probability descending
    candidates.sort(key=lambda x: x["over_2_5"], reverse=True)
    top = candidates[:top_n]

    if not top:
        console.print(
            f"[yellow]  No matches today with Over 2.5 probability ≥ {min_prob:.0%}. "
            f"Try again on a higher-scoring matchday.[/yellow]"
        )
        console.print()
        return

    if len(candidates) < top_n:
        console.print(
            f"[yellow]  Only {len(candidates)} match(es) qualify "
            f"(Over 2.5 prob ≥ {min_prob:.0%}) — showing all.[/yellow]"
        )
        console.print()

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Match",       width=26, no_wrap=True)
    table.add_column("Lg",          width=4,  justify="center")
    table.add_column("xG",          width=9,  justify="center")
    table.add_column("Over 2.5 %",  width=11, justify="right")
    table.add_column("BTTS %",      width=9,  justify="right")
    table.add_column("Odds",        width=7,  justify="right")

    for c in top:
        p = c["over_2_5"]
        col = _prob_color(p)

        match_str = f"{_shorten(c['home'])} vs {_shorten(c['away'])}"
        xg_str    = f"{c['exp_home']:.1f}–{c['exp_away']:.1f}"
        prob_str  = f"[{col}]{p * 100:.1f}%[/{col}]"
        btts_str  = f"{c['btts'] * 100:.0f}%"

        if c["o25_odds"] is not None:
            odds_str = f"[{col}]{c['o25_odds']:.2f}[/{col}]"
        else:
            fair = 1 / p if p > 0 else 0
            odds_str = f"[dim]{fair:.2f}~[/dim]"

        table.add_row(
            match_str,
            c["league"],
            f"[dim]{xg_str}[/dim]",
            prob_str,
            f"[dim]{btts_str}[/dim]",
            odds_str,
        )

    console.print(table)
    console.print(
        f"  [dim]Ranked by model Over 2.5 probability across all {len(fixtures)} fixture(s) today.[/dim]"
    )
    console.print()


def export_text():
    return console.export_text()
