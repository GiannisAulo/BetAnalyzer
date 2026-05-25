"""
staking.py — Quarter-Kelly stake sizing for value-bet recommendations.

Each pick with real bookmaker odds and positive edge gets a recommended stake
in units. One unit = 1% of bankroll (0.10 EUR on the default 10 EUR bankroll).
Stakes are capped to keep any single bet at most 10% of bankroll, which is
quarter-Kelly's natural safety ceiling for typical edges.

Why quarter Kelly: full Kelly maximises long-run growth but has very high
variance. Quarter Kelly captures ~85% of the growth rate while reducing
drawdown depth by ~75%. On a small bankroll where ruin is a real concern,
that trade is correct.

This module never touches the model. It reads (prob, odds) and outputs units.
Picks without real bookmaker odds get None — we don't recommend stakes when
we can't verify the bet's price.

Bankroll is persisted in `data/user_settings.json` and can be changed via the
menu (or programmatically with `set_bankroll(eur)`). The default is 10 EUR.
"""

import json
import os

# ── Static parameters ───────────────────────────────────────────────────────
UNIT_PCT         = 0.01     # 1 unit = 1% of bankroll
KELLY_FRACTION   = 0.25     # quarter Kelly
MAX_STAKE_UNITS  = 10       # safety cap: at most 10% of bankroll per pick

# ── Bankroll persistence ────────────────────────────────────────────────────
_DEFAULT_BANKROLL = 10.0
_SETTINGS_FILE    = os.path.join("data", "user_settings.json")
_MIN_BANKROLL     = 1.0
_MAX_BANKROLL     = 1_000_000.0


def _read_settings():
    if not os.path.exists(_SETTINGS_FILE):
        return {}
    try:
        with open(_SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _load_bankroll():
    """Read the user's saved bankroll, falling back to the default."""
    raw = _read_settings().get("bankroll_eur")
    try:
        v = float(raw)
        if _MIN_BANKROLL <= v <= _MAX_BANKROLL:
            return v
    except (TypeError, ValueError):
        pass
    return _DEFAULT_BANKROLL


def set_bankroll(eur):
    """Persist a new bankroll and update derived values.

    Raises ValueError when out of range; callers should display the error.
    Mutates module-level BANKROLL_EUR and UNIT_EUR so the next call to
    units_to_eur() / coupon rendering picks up the new value.
    """
    try:
        v = float(eur)
    except (TypeError, ValueError):
        raise ValueError(f"bankroll must be a number, got {eur!r}")
    if not (_MIN_BANKROLL <= v <= _MAX_BANKROLL):
        raise ValueError(
            f"bankroll must be between €{_MIN_BANKROLL:.0f} and €{_MAX_BANKROLL:,.0f}"
        )

    settings = _read_settings()
    settings["bankroll_eur"] = v
    os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, sort_keys=True)

    global BANKROLL_EUR, UNIT_EUR
    BANKROLL_EUR = v
    UNIT_EUR     = BANKROLL_EUR * UNIT_PCT


# Loaded at import; mutated by set_bankroll()
BANKROLL_EUR = _load_bankroll()
UNIT_EUR     = BANKROLL_EUR * UNIT_PCT


def units_to_eur(units):
    """Convert a unit count to EUR using the current bankroll."""
    if units is None:
        return None
    return units * UNIT_EUR


def compute_stake_units(prob, odds, edge=None):
    """Return recommended stake in units, or None when staking is not advised.

    prob:  model probability of the pick winning (0..1)
    odds:  decimal bookmaker odds (e.g. 1.85). When None, returns None — we
           don't recommend stakes for picks lacking a real bookmaker price.
    edge:  optional pre-computed edge in percent. When None we recompute from
           prob and odds. Used as a sanity gate: non-positive edge → no stake.

    Quarter-Kelly formula for decimal odds (b = odds - 1, q = 1 - p):
        f* = (b * p - q) / b
        stake_fraction = KELLY_FRACTION * max(f*, 0)
        units          = round(stake_fraction / UNIT_PCT)
    Capped above at MAX_STAKE_UNITS. When the natural Kelly stake rounds to
    less than 1 unit, returns None — the edge is too small to act on at the
    current unit size, and force-clamping up to 1 unit would over-stake.

    Returns an integer number of units, or None if the pick shouldn't be staked.
    """
    if odds is None or odds <= 1.0:
        return None
    if prob is None or prob <= 0 or prob >= 1:
        return None

    # Quick edge sanity gate
    if edge is None:
        edge = (prob - 1.0 / odds) * 100
    if edge <= 0:
        return None

    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    if f_star <= 0:
        return None   # Kelly says don't bet

    stake_fraction = KELLY_FRACTION * f_star
    units = round(stake_fraction / UNIT_PCT)

    if units < 1:
        return None   # edge too small at current unit size; no recommendation
    if units > MAX_STAKE_UNITS:
        units = MAX_STAKE_UNITS
    return int(units)


def format_stake(units):
    """Render a units integer as 'N u (€X.XX)'. Returns '—' when units is None."""
    if units is None:
        return "—"
    eur = units_to_eur(units)
    return f"{units}u (€{eur:.2f})"
