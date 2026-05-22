from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

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
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Match",  width=50, no_wrap=True)
    table.add_column("Lg",     width=3,  justify="center")
    table.add_column("Pick",   width=24, no_wrap=True)
    table.add_column("Market", width=10, no_wrap=True)
    table.add_column("Conf",   width=10, justify="right")
    table.add_column("Odds",   width=6,  justify="right")
    table.add_column("Edge",   width=7,  justify="right")

    # Sort by league, then by edge desc (value tier) / model_prob desc (prob tier)
    current_league = None
    for pick in sorted(picks, key=lambda p: (p["league"], -(p.get("edge") or 0), -p["model_prob"])):
        # League divider row
        if pick["league"] != current_league:
            current_league = pick["league"]
            league_name = _LEAGUE_NAMES.get(current_league, current_league)
            table.add_row(
                f"[bold dim]{league_name}[/bold dim]",
                "", "", "", "", "", "",
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

        table.add_row(
            match_str,
            pick["league"],
            pick_display,
            f"[dim]{pick['market']}[/dim]",
            conf_str,
            odds_str,
            edge_str,
        )
        reason = pick.get("reason", "")
        if reason:
            table.add_row(f"[dim italic]{reason}[/dim italic]", "", "", "", "", "", "")

    return table


def render_coupon(value_picks, no_odds_picks, date_str=None):
    """
    Render a two-tier coupon.

    Tier 1 — VALUE PICKS: fixtures where the model found positive edge against
    bookmaker odds. Sorted by edge descending. These are the actionable bets.

    Tier 2 — BEST AVAILABLE: fixtures with no bookmaker odds or no positive edge.
    Sorted by model_prob. Shown for reference — check live prices before betting.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%A %d %B %Y")

    console.print()
    console.print(Rule(f"[bold yellow]BETTING COUPON  —  {date_str}[/bold yellow]", style="yellow"))
    console.print()

    # ── Tier 1: Value picks ──────────────────────────────────────────────────
    console.print(Rule("[bold green]VALUE PICKS — EDGE CONFIRMED[/bold green]", style="green"))
    console.print()
    if value_picks:
        console.print(_build_picks_table(value_picks, show_edge=True))
    else:
        console.print("[yellow]  No positive-edge picks found today.[/yellow]")
        console.print()

    # ── Tier 2: Best available (no verified edge) ────────────────────────────
    if no_odds_picks:
        console.print(Rule("[bold yellow]BEST AVAILABLE — NO BOOKMAKER ODDS[/bold yellow]", style="yellow"))
        console.print()
        console.print(_build_picks_table(no_odds_picks, show_edge=False))
        console.print(
            "  [dim]These picks have no verified edge. Check live odds before betting.[/dim]"
        )
        console.print()
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
