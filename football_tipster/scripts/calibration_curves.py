"""scripts/calibration_curves.py — Per-market calibration analysis.

Reads bets_log.csv, bins model_prob into 5pp buckets per pick type, and
reports actual win rate vs predicted probability.  When ≥20 settled bets
exist in a pick type, isotonic regression is also applied and shown as the
corrected expected probability.

Usage:
    python -m scripts.calibration_curves
    python -m scripts.calibration_curves --league PL
    python -m scripts.calibration_curves --pick "Over 2.5"
    python -m scripts.calibration_curves --by-league    # breakdown per league

Bias column:
    green  = actual within ±5pp of predicted  (well-calibrated)
    red    = actual below predicted            (model over-confident)
    yellow = actual above predicted            (model under-confident)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# Fix Windows terminal encoding before Rich initialises
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

console = Console(legacy_windows=False)

_BET_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bets_log.csv"
)

# Minimum settled bets in a pick type before isotonic regression is attempted.
_MIN_ISOTONIC = 20
# Width of each probability bucket.
_BIN = 0.05


def _bucket(prob: float) -> float:
    return round(int(prob / _BIN) * _BIN, 3)


def _load(path: str, league_filter=None, pick_filter=None) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("result") not in ("W", "L", "D"):
                continue
            if league_filter and r.get("league") != league_filter:
                continue
            if pick_filter and r.get("pick") != pick_filter:
                continue
            try:
                prob = float(r["model_prob"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "pick":       r["pick"],
                "market":     r["market"],
                "league":     r["league"],
                "model_prob": prob,
                "win":        1 if r["result"] == "W" else 0,
            })
    return rows


def _try_isotonic(probs: list, wins: list) -> dict:
    """Fit isotonic regression; return bucket-mid → corrected-prob mapping."""
    if len(probs) < _MIN_ISOTONIC:
        return {}
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression

        X = np.array(probs)
        y = np.array(wins, dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X, y)
        midpoints = [_bucket(p) + _BIN / 2 for p in probs]
        preds = iso.predict(midpoints)
        # Map bucket lower bound → median corrected value in that bucket
        by_bucket = defaultdict(list)
        for p, pred in zip(probs, preds):
            by_bucket[_bucket(p)].append(pred)
        return {lo: sum(vs) / len(vs) for lo, vs in by_bucket.items()}
    except ImportError:
        return {}


def _render_pick(pick_name: str, rows: list, show_isotonic: bool):
    """Render one calibration table for a single pick type."""
    n_total = len(rows)
    n_wins  = sum(r["win"] for r in rows)
    overall = n_wins / n_total if n_total else 0.0

    # Build bucket summary
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[_bucket(r["model_prob"])].append(r["win"])

    # Isotonic regression (full dataset, not per-bucket)
    iso_map = {}
    if show_isotonic and n_total >= _MIN_ISOTONIC:
        probs = [r["model_prob"] for r in rows]
        wins  = [r["win"] for r in rows]
        iso_map = _try_isotonic(probs, wins)

    has_iso = bool(iso_map)

    wr_color = "green" if overall >= 0.60 else ("yellow" if overall >= 0.50 else "red")
    table = Table(
        title=f"[bold]{pick_name}[/bold]  "
              f"[dim]({n_total} bets · [{wr_color}]{overall:.0%} WR[/{wr_color}])[/dim]",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Predicted",  width=12, justify="center")
    table.add_column("n",          width=4,  justify="right")
    table.add_column("Actual WR",  width=10, justify="center")
    table.add_column("Bias",       width=10, justify="center")
    if has_iso:
        table.add_column("Isotonic",  width=10, justify="center")
        table.add_column("Adj Bias",  width=10, justify="center")

    for lo in sorted(buckets.keys()):
        bucket_wins = buckets[lo]
        n       = len(bucket_wins)
        actual  = sum(bucket_wins) / n
        mid     = lo + _BIN / 2
        bias    = actual - mid

        pred_range = f"{lo:.0%}–{lo + _BIN:.0%}"

        # Bias coloring: ±5pp = green, negative = red (over-confident), positive = yellow
        if abs(bias) < 0.05:
            bias_str = f"[green]{bias:+.1%}[/green]"
        elif bias < 0:
            bias_str = f"[red]{bias:+.1%}[/red]"
        else:
            bias_str = f"[yellow]{bias:+.1%}[/yellow]"

        # Actual WR: red when worse than predicted by >5pp
        wr_color_cell = "green" if actual >= mid - 0.05 else "red"
        wr_str = f"[{wr_color_cell}]{actual:.0%}[/{wr_color_cell}]"

        row = [pred_range, str(n), wr_str, bias_str]

        if has_iso:
            iso_val = iso_map.get(lo)
            if iso_val is not None:
                iso_bias = actual - iso_val
                iso_str  = f"{iso_val:.0%}"
                adj_color = "green" if abs(iso_bias) < 0.05 else ("red" if iso_bias < 0 else "yellow")
                adj_str  = f"[{adj_color}]{iso_bias:+.1%}[/{adj_color}]"
            else:
                iso_str = "—"
                adj_str = "—"
            row += [iso_str, adj_str]

        table.add_row(*row)

    console.print(table)

    if has_iso:
        console.print(
            "  [dim]Isotonic: calibrated expected probability. "
            "Adj Bias = actual - isotonic (near zero = well-fit).[/dim]"
        )
    elif n_total < _MIN_ISOTONIC:
        needed = _MIN_ISOTONIC - n_total
        console.print(
            f"  [dim]Isotonic: needs {needed} more bets (>={_MIN_ISOTONIC} required).[/dim]"
        )
    console.print()


def _summary_table(rows: list):
    """One-line summary per pick type — quick overview of every market."""
    by_pick: dict = defaultdict(list)
    for r in rows:
        by_pick[r["pick"]].append(r)

    table = Table(
        title="[bold]Market Summary[/bold]",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Pick",         width=28, no_wrap=True)
    table.add_column("n",            width=5,  justify="right")
    table.add_column("Actual WR",    width=10, justify="center")
    table.add_column("Avg Pred",     width=10, justify="center")
    table.add_column("Bias",         width=10, justify="center")
    table.add_column("Status",       width=18, justify="left")

    for pick_name in sorted(by_pick.keys()):
        pick_rows = by_pick[pick_name]
        n       = len(pick_rows)
        actual  = sum(r["win"] for r in pick_rows) / n
        avg_pred = sum(r["model_prob"] for r in pick_rows) / n
        bias    = actual - avg_pred

        wr_color = "green" if actual >= 0.60 else ("yellow" if actual >= 0.50 else "red")
        wr_str   = f"[{wr_color}]{actual:.0%}[/{wr_color}]"

        pred_str = f"{avg_pred:.0%}"

        if abs(bias) < 0.05:
            bias_str   = f"[green]{bias:+.1%}[/green]"
            status_str = "[green]OK[/green]"
        elif bias < -0.10:
            bias_str   = f"[red]{bias:+.1%}[/red]"
            status_str = "[red]OVER-CONFIDENT[/red]"
        elif bias < 0:
            bias_str   = f"[red]{bias:+.1%}[/red]"
            status_str = "[yellow]slightly over[/yellow]"
        else:
            bias_str   = f"[yellow]{bias:+.1%}[/yellow]"
            status_str = "[yellow]under-confident[/yellow]"

        if n < 10:
            status_str = "[dim]too few bets[/dim]"

        table.add_row(pick_name, str(n), wr_str, pred_str, bias_str, status_str)

    console.print(table)
    console.print()


def _by_league_breakdown(rows: list, pick_filter=None):
    """Per-league WR for each pick type."""
    pick_types = sorted({r["pick"] for r in rows})
    leagues    = sorted({r["league"] for r in rows})

    if pick_filter:
        pick_types = [p for p in pick_types if p == pick_filter]

    for pick_name in pick_types:
        pick_rows = [r for r in rows if r["pick"] == pick_name]
        n_total   = len(pick_rows)
        n_wins    = sum(r["win"] for r in pick_rows)
        overall   = n_wins / n_total if n_total else 0.0

        wr_color = "green" if overall >= 0.60 else ("yellow" if overall >= 0.50 else "red")
        table = Table(
            title=f"[bold]{pick_name}[/bold]  "
                  f"[dim]({n_total} total · [{wr_color}]{overall:.0%} WR[/{wr_color}])[/dim]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            padding=(0, 2),
            show_edge=False,
        )
        table.add_column("League",    width=6,  justify="left")
        table.add_column("n",         width=5,  justify="right")
        table.add_column("Actual WR", width=10, justify="center")
        table.add_column("Avg Pred",  width=10, justify="center")
        table.add_column("Bias",      width=10, justify="center")

        for lg in leagues:
            lg_rows = [r for r in pick_rows if r["league"] == lg]
            if not lg_rows:
                continue
            n      = len(lg_rows)
            actual = sum(r["win"] for r in lg_rows) / n
            avg_pred = sum(r["model_prob"] for r in lg_rows) / n
            bias   = actual - avg_pred

            wr_c  = "green" if actual >= 0.60 else ("yellow" if actual >= 0.50 else "red")
            wr_s  = f"[{wr_c}]{actual:.0%}[/{wr_c}]"
            b_c   = "green" if abs(bias) < 0.05 else ("red" if bias < 0 else "yellow")
            b_s   = f"[{b_c}]{bias:+.1%}[/{b_c}]"

            table.add_row(lg, str(n), wr_s, f"{avg_pred:.0%}", b_s)

        console.print(table)
        console.print()


def run(league_filter=None, pick_filter=None, by_league=False):
    """Entry point callable from menu or CLI."""
    rows = _load(_BET_LOG, league_filter=league_filter, pick_filter=pick_filter)

    if not rows:
        console.print("[red]No settled bets found matching those filters.[/red]")
        return

    title = "Calibration Analysis"
    if league_filter:
        title += f" — {league_filter}"
    if pick_filter:
        title += f" — {pick_filter}"

    console.print()
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))
    console.print(
        f"  [dim]{len(rows)} settled bets  |  "
        f"isotonic regression requires >={_MIN_ISOTONIC} per pick type[/dim]"
    )
    console.print()

    if by_league:
        console.print(Rule("[cyan]Per-League Breakdown[/cyan]", style="dim"))
        _by_league_breakdown(rows, pick_filter)
        return

    # Summary table first
    _summary_table(rows)

    # Per-pick detailed calibration
    if not pick_filter:
        console.print(Rule("[cyan]Detailed Calibration by Probability Bucket[/cyan]", style="dim"))
        console.print()

    by_pick: dict = defaultdict(list)
    for r in rows:
        by_pick[r["pick"]].append(r)

    for pick_name in sorted(by_pick.keys()):
        _render_pick(pick_name, by_pick[pick_name], show_isotonic=True)


def main():
    parser = argparse.ArgumentParser(
        description="Per-market calibration analysis for Football Tipster"
    )
    parser.add_argument("--league",    default=None, metavar="CODE",
                        help="Filter to one league (e.g. PL)")
    parser.add_argument("--pick",      default=None, metavar="NAME",
                        help="Filter to one pick type (e.g. 'Over 2.5')")
    parser.add_argument("--by-league", action="store_true", dest="by_league",
                        help="Show per-league WR breakdown instead of bucket detail")
    args = parser.parse_args()
    run(league_filter=args.league, pick_filter=args.pick, by_league=args.by_league)


if __name__ == "__main__":
    main()
