"""
Market evaluators: 1X2, Double Chance, Over/Under, BTTS, Combo.

When odds are supplied the edge is computed; when absent, picks are flagged
purely on model confidence so the coupon is still useful without live odds.

Every game produces at least one pick — the goal is to give the user a
recommendation for every fixture so they can build a coupon regardless of
whether bookmaker prices are available.

Minimum confidence thresholds for no-odds picks prevent noise (e.g. showing
a 32% Away Win with no odds), but there is no upper cap — a 77% Home Win is
a strong tip whether or not we have the live price.
"""

from config import EXPECTED_TOTAL_GATES_BY_LEAGUE, EXPECTED_TOTAL_CAPS_BY_LEAGUE
import warn_log

# Maximum gap between model probability and de-vigged market probability before
# a pick is suppressed. A gap > 15pp means the model disagrees sharply with the
# efficient market — more likely a model calibration error than genuine edge.
_SHARP_CONSENSUS_MAX_GAP = 0.15

# Minimum odds we consider worth showing. Picks below this are skipped entirely.
MIN_FAIR_ODDS = 1.60

# Absolute probability floor used in value-sweep mode.
# Below this nothing is worth recommending regardless of edge.
_LOW_FLOOR = 0.35

# Absolute minimum model probability to consider a pick, even with positive edge.
# Prevents the model from recommending longshots with tiny win probability
# (e.g. Away Win at 11% or Home Win at 25%) where edge is just noise.
# Applied to BOTH odds-based and no-odds paths.
MIN_PROB = {
    "Home Win":  0.62,   # raised: live WR 44% at avg model_prob 0.55 — need stronger conviction
    "Away Win":  0.65,   # raised: 0W/6L live WR at 0.52 — model has no edge on away picks below 0.65
    "Draw":      0.45,
    "Over 2.5":  0.65,   # raised: 52% live WR at 0.64 avg model_prob — 12pp calibration gap
    "Under 2.5": 0.65,   # raised: 46% live WR at 0.61 avg — model cannot pick Under 2.5 below 0.65
    "Over 3.5":  0.65,   # raised: backtest 44% WR at 0.45 — losing market
    "Under 3.5": 0.50,
}

# Minimum model probability to show a pick when no live odds are available.
# Higher than MIN_PROB because without odds we can't confirm edge — require more confidence.
_MIN_PROB_NO_ODDS = {
    "Home Win":  0.65,
    "Away Win":  0.65,
    "Draw":      0.52,
    "Over 1.5":  0.75,
    "Under 1.5": 0.70,
    "Over 2.5":  0.67,
    "Under 2.5": 0.67,
    "Over 3.5":  0.65,
    "Under 3.5": 0.58,
}

# Contextual gates for Over/Under: expected_total must support the pick direction.
# Prevents Over 2.5 picks in low-xG fixtures and Under 2.5 in high-xG ones.
_EXPECTED_TOTAL_GATES = {
    "Over 2.5":  2.90,   # raised: xG trend fix deflates expected_total; require more signal
    "Over 3.5":  3.50,
}
_EXPECTED_TOTAL_CAPS = {
    "Under 2.5": 2.50,   # tightened: Under 2.5 only when model clearly supports low scoring
    "Under 3.5": 4.00,   # reject Under 3.5 if expected_total > 4.0
}


def evaluate_1x2(probs, odds_home=None, odds_draw=None, odds_away=None, min_edge=5.0, _value_sweep=False):
    picks = []
    markets_list = [
        ("Home Win", "home", odds_home),
        ("Draw",     "draw", odds_draw),
        ("Away Win", "away", odds_away),
    ]
    for name, key, odds in markets_list:
        model_prob = probs[key]
        floor = _LOW_FLOOR if _value_sweep else MIN_PROB.get(name, 0)
        if model_prob < floor:
            continue
        if odds is not None:
            # Sharp consensus filter stays active even in value-sweep mode.
            if odds_home is not None and odds_draw is not None and odds_away is not None:
                other = [o for o in [odds_home, odds_draw, odds_away] if o is not odds]
                sharp_prob = _devig_prob(odds, *other)
                if sharp_prob is not None and abs(model_prob - sharp_prob) > _SHARP_CONSENSUS_MAX_GAP:
                    warn_log.fallback(
                        f"sharp consensus filter suppressed {name}: "
                        f"model={model_prob:.2f} market={sharp_prob:.2f} "
                        f"gap={abs(model_prob - sharp_prob):.2f}",
                        "pick suppressed",
                    )
                    continue
            implied = 1 / odds
            edge = (model_prob - implied) * 100
            if _value_sweep or edge >= min_edge * 0.6:
                picks.append(_pick("1X2", name, model_prob, implied, odds, edge))
        elif _value_sweep or model_prob >= 0.65:
            picks.append(_pick("1X2", name, model_prob, None, None, None))
    return picks


def evaluate_double_chance(probs, odds_1x=None, odds_x2=None, odds_12=None, min_edge=3.0, _value_sweep=False):
    picks = []
    dc_markets = [
        ("1X (Home or Draw)", probs["home"] + probs["draw"], odds_1x),
        ("X2 (Draw or Away)", probs["draw"] + probs["away"], odds_x2),
        ("12 (Home or Away)", probs["home"] + probs["away"], odds_12),
    ]
    for name, model_prob, odds in dc_markets:
        if odds is not None:
            if odds < MIN_FAIR_ODDS:
                continue
            implied = 1 / odds
            edge = (model_prob - implied) * 100
            if _value_sweep or edge >= min_edge * 0.6:
                picks.append(_pick("Double Chance", name, model_prob, implied, odds, edge))
        elif _value_sweep or model_prob >= 0.60:
            if _value_sweep and model_prob < _LOW_FLOOR:
                continue
            picks.append(_pick("Double Chance", name, model_prob, None, None, None))
    return picks


def evaluate_over_under(probs, odds_over15=None, odds_under15=None,
                         odds_over25=None, odds_under25=None,
                         odds_over35=None, odds_under35=None, min_edge=4.0,
                         expected_total=None, league=None, _value_sweep=False):
    """Evaluate Over/Under markets.

    expected_total: model's predicted total goals for the fixture (exp_home + exp_away).
    When supplied, acts as a contextual gate — prevents Over 2.5 picks in low-scoring
    fixtures and Under 2.5 picks in high-scoring ones, regardless of edge.
    league: when supplied, per-league gate/cap values override the global defaults.
    """
    picks = []

    # paired_odds: the complementary line used for de-vig (over ↔ under same threshold).
    ou_markets = [
        ("Over 1.5",  "over_1_5",  odds_over15,  odds_under15),
        ("Under 1.5", "under_1_5", odds_under15, odds_over15),
        ("Over 2.5",  "over_2_5",  odds_over25,  odds_under25),
        ("Under 2.5", "under_2_5", odds_under25, odds_over25),
        ("Over 3.5",  "over_3_5",  odds_over35,  odds_under35),
        ("Under 3.5", "under_3_5", odds_under35, odds_over35),
    ]
    for name, key, odds, paired_odds in ou_markets:
        model_prob = probs[key]

        floor = _LOW_FLOOR if _value_sweep else MIN_PROB.get(name, 0)
        if model_prob < floor:
            continue

        # Contextual gate: expected_total must support the pick direction.
        # Per-league values take precedence over global fallbacks.
        if expected_total is not None:
            gate = (EXPECTED_TOTAL_GATES_BY_LEAGUE.get(name, {}).get(league)
                    if league else None) or _EXPECTED_TOTAL_GATES.get(name)
            if gate is not None and expected_total < gate:
                continue
            cap = (EXPECTED_TOTAL_CAPS_BY_LEAGUE.get(name, {}).get(league)
                   if league else None) or _EXPECTED_TOTAL_CAPS.get(name)
            if cap is not None and expected_total > cap:
                continue

        if odds is not None:
            # Sharp consensus filter stays active even in value-sweep mode.
            sharp_prob = _devig_prob(odds, paired_odds) if paired_odds is not None else None
            if sharp_prob is not None and abs(model_prob - sharp_prob) > _SHARP_CONSENSUS_MAX_GAP:
                warn_log.fallback(
                    f"sharp consensus filter suppressed {name}: "
                    f"model={model_prob:.2f} market={sharp_prob:.2f} "
                    f"gap={abs(model_prob - sharp_prob):.2f}",
                    "pick suppressed",
                )
                continue
            implied = 1 / odds
            edge = (model_prob - implied) * 100
            if _value_sweep or edge >= min_edge * 0.6:
                picks.append(_pick("Over/Under", name, model_prob, implied, odds, edge))
        elif _value_sweep or model_prob >= _MIN_PROB_NO_ODDS.get(name, 1.0):
            picks.append(_pick("Over/Under", name, model_prob, None, None, None))
    return picks


def evaluate_btts(probs, odds_yes=None, odds_no=None, min_edge=4.0, _value_sweep=False):
    picks = []
    # BTTS No disabled: backtest 49% WR across 4 leagues (near-random).
    # Clean-sheet model is not reliable enough to pick against both teams scoring.
    btts_markets = [
        ("BTTS Yes", "btts_yes", odds_yes),
    ]
    for name, key, odds in btts_markets:
        model_prob = probs[key]

        if _value_sweep and model_prob < _LOW_FLOOR:
            continue

        if odds is not None:
            implied = 1 / odds
            edge = (model_prob - implied) * 100
            if _value_sweep or edge >= min_edge * 0.6:
                picks.append(_pick("BTTS", name, model_prob, implied, odds, edge))
        elif _value_sweep or model_prob >= 0.62:
            picks.append(_pick("BTTS", name, model_prob, None, None, None))
    return picks


def best_value_pick(probs, fx_odds: dict, expected_total=None, league=None):
    """
    Sweep every market for a fixture and return the single best-value pick.

    Strategy:
    - Evaluate all markets with _value_sweep=True (low floors, no edge minimum).
    - Among picks that have real bookmaker odds and positive edge, return the one
      with the highest edge — that is the pick where the model most disagrees with
      the bookmaker in our favour.
    - If no market shows positive edge, return the highest model_prob pick as the
      best available option (clearly flagged as unverified).

    Returns:
        {"pick": <pick dict>, "verified_edge": bool}
        or None if no candidates pass the absolute floor.
    """
    all_candidates = []
    all_candidates += evaluate_1x2(
        probs,
        odds_home=fx_odds.get("home_odds"),
        odds_draw=fx_odds.get("draw_odds"),
        odds_away=fx_odds.get("away_odds"),
        min_edge=0,
        _value_sweep=True,
    )
    all_candidates += evaluate_double_chance(
        probs,
        min_edge=0,
        _value_sweep=True,
    )
    all_candidates += evaluate_over_under(
        probs,
        odds_over15=fx_odds.get("over_1.5"),
        odds_under15=fx_odds.get("under_1.5"),
        odds_over25=fx_odds.get("over_2.5"),
        odds_under25=fx_odds.get("under_2.5"),
        odds_over35=fx_odds.get("over_3.5"),
        odds_under35=fx_odds.get("under_3.5"),
        min_edge=0,
        expected_total=expected_total,
        league=league,
        _value_sweep=True,
    )
    all_candidates += evaluate_btts(
        probs,
        odds_yes=fx_odds.get("btts_yes"),
        min_edge=0,
        _value_sweep=True,
    )
    all_candidates += evaluate_combos(all_candidates, probs=probs)

    # Absolute floor — remove garbage below _LOW_FLOOR
    all_candidates = [c for c in all_candidates if c["model_prob"] >= _LOW_FLOOR]
    if not all_candidates:
        return None

    # Picks with real bookmaker odds and positive edge
    positive_edge = [
        c for c in all_candidates
        if c.get("odds") is not None and (c.get("edge") or 0) > 0
    ]

    if positive_edge:
        best = max(positive_edge, key=lambda p: p.get("edge", 0))
        return {"pick": best, "verified_edge": True}

    # No positive edge anywhere — return highest model_prob as best-available
    best = max(all_candidates, key=lambda p: p["model_prob"])
    return {"pick": best, "verified_edge": False}


def _joint_prob_from_matrix(matrix, leg1_name, leg2_name):
    """
    Compute exact joint probability for a two-leg combo using the score matrix.
    Returns None when either leg isn't a matrix-computable outcome.

    Supported leg names (must match pick names used in evaluate_*):
      1X2:        Home Win, Draw, Away Win
      Over/Under: Over 1.5, Under 1.5, Over 2.5, Under 2.5, Over 3.5, Under 3.5
      BTTS:       BTTS Yes, BTTS No
      DC:         1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away)
    """
    if matrix is None:
        return None

    _GRID = len(matrix)

    def cell_matches(i, j, leg):
        if leg == "Home Win":
            return i > j
        if leg == "Draw":
            return i == j
        if leg == "Away Win":
            return i < j
        if leg == "Over 1.5":
            return i + j > 1
        if leg == "Under 1.5":
            return i + j < 2
        if leg == "Over 2.5":
            return i + j > 2
        if leg == "Under 2.5":
            return i + j < 3
        if leg == "Over 3.5":
            return i + j > 3
        if leg == "Under 3.5":
            return i + j < 4
        if leg == "BTTS Yes":
            return i > 0 and j > 0
        if leg == "BTTS No":
            return i == 0 or j == 0
        if leg == "1X (Home or Draw)":
            return i >= j
        if leg == "X2 (Draw or Away)":
            return i <= j
        if leg == "12 (Home or Away)":
            return i != j
        return None  # unknown leg — can't compute from matrix

    total = 0.0
    for i in range(_GRID):
        for j in range(_GRID):
            l1 = cell_matches(i, j, leg1_name)
            l2 = cell_matches(i, j, leg2_name)
            if l1 is None or l2 is None:
                return None  # leg not supported — fall back to multiplication
            if l1 and l2:
                total += matrix[i][j]
    return total


def evaluate_combos(all_picks, probs=None):
    """
    Build logical double combos from the picks already generated.

    When probs contains a 'score_matrix' (from compute_match_probabilities),
    joint probabilities are derived directly from the score grid — capturing
    exact statistical correlations without hand-tuned discount factors.
    Falls back to independent multiplication when the matrix is unavailable.

    Rules:
    - Legs must come from different markets.
    - Combined odds must be >= 1.60.
    - Impossible combinations excluded (BTTS Yes + Under 1.5/2.5).
    - Returns up to 5 combos sorted by model_prob descending.
    """
    matrix = (probs or {}).get("score_matrix")

    by_name = {p["pick"]: p for p in all_picks}

    def _odds(p):
        if p.get("odds") is not None:
            return p["odds"]
        mp = p.get("model_prob", 0)
        return (1 / mp) if mp > 0 else None

    _COMBO_DEFS = [
        ("Home Win",           "Over 1.5"),
        ("Home Win",           "Over 2.5"),
        ("Home Win",           "BTTS Yes"),
        ("Home Win",           "BTTS No"),
        ("1X (Home or Draw)",  "Over 1.5"),
        ("1X (Home or Draw)",  "Under 2.5"),
        ("X2 (Draw or Away)",  "Under 2.5"),
        ("X2 (Draw or Away)",  "Over 1.5"),
        ("Away Win",           "Over 1.5"),
        ("Away Win",           "Over 2.5"),
        ("BTTS Yes",           "Over 2.5"),
        ("Draw",               "Under 2.5"),
        ("Draw",               "BTTS Yes"),
    ]

    _IMPOSSIBLE = {
        frozenset(["BTTS Yes", "Under 1.5"]),
        frozenset(["BTTS Yes", "Under 2.5"]),
    }

    combos = []
    for leg1_name, leg2_name in _COMBO_DEFS:
        p1 = by_name.get(leg1_name)
        p2 = by_name.get(leg2_name)
        if p1 is None or p2 is None:
            continue

        pair = frozenset([leg1_name, leg2_name])
        if pair in _IMPOSSIBLE:
            continue

        o1 = _odds(p1)
        o2 = _odds(p2)
        if o1 is None or o2 is None:
            continue

        combo_odds = o1 * o2
        if combo_odds < 1.60:
            continue

        # Use score matrix for exact joint probability when available.
        # Falls back to independent multiplication if leg isn't matrix-computable.
        joint_prob = _joint_prob_from_matrix(matrix, leg1_name, leg2_name)
        if joint_prob is None:
            joint_prob = p1["model_prob"] * p2["model_prob"]

        if joint_prob < 0.30:
            continue

        has_real_odds = p1.get("odds") is not None and p2.get("odds") is not None
        edge = (joint_prob - 1 / combo_odds) * 100 if has_real_odds else None

        combos.append({
            "market": "Combo",
            "pick": f"{leg1_name} + {leg2_name}",
            "model_prob": joint_prob,
            "implied_prob": 1 / combo_odds if has_real_odds else None,
            "odds": combo_odds if has_real_odds else None,
            "edge": edge,
        })

    combos.sort(key=lambda c: c["model_prob"], reverse=True)
    return combos[:5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _devig_prob(own_price: float, *other_prices: float) -> float | None:
    """Proportional de-vig: remove bookmaker margin and return true implied probability.

    own_price and other_prices are decimal odds.  Returns None if any price is
    missing or <= 1.0 (which would imply a guaranteed outcome).
    """
    all_prices = [own_price, *other_prices]
    if any(p is None or p <= 1.0 for p in all_prices):
        return None
    raw = [1.0 / p for p in all_prices]
    return raw[0] / sum(raw)


def _pick(market, name, model_prob, implied, odds, edge):
    return {
        "market": market,
        "pick": name,
        "model_prob": model_prob,
        "implied_prob": implied,
        "odds": odds,
        "edge": edge,
    }
