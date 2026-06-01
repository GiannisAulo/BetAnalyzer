#!/usr/bin/env python3
"""
WC2026 / specific-game prediction engine.

Pipeline (per match):
  ELO lambda (65%) + normalised attack/defence lambda (30%)
  + form multiplier (5%) → Dixon-Coles matrix → de-vigged edge detection.

Pure self-contained module — no config.py / API_KEY dependency.
"""

import csv
import json
import math
from datetime import date, datetime
from pathlib import Path

_HERE = Path(__file__).parent
BETS_LOG = _HERE / "bets_log.csv"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

WC_TOTAL_GOALS    = 2.65    # WC 2014-2022 group stage average total goals/match
WC_HALF           = WC_TOTAL_GOALS / 2          # per-team neutral base
ATK_MEAN          = 1.194   # mean attack_strength across 48 WC2026 teams
DEF_MEAN          = 0.996   # mean defense_strength across 48 WC2026 teams
WC_MEAN_ELO       = 1583    # mean ELO across 48 WC2026 teams
ELO_DIVISOR       = 400.0

ELO_WEIGHT        = 0.65
GOALS_WEIGHT      = 0.30
FORM_WEIGHT       = 0.05

RHO_WC            = -0.10   # DC low-score correction for international football
FORM_DECAY_K      = 0.50    # per-month decay
H2H_DECAY_K       = 0.35    # per-year decay
H2H_MAX_AGE_DAYS  = 5 * 365
H2H_MIN_WEIGHT    = 1.5
H2H_MAX_NUDGE     = 0.03    # ±3% cap on H2H nudge

FORM_MULT_CLAMP   = (0.92, 1.08)
LAM_HARD_CAP      = 3.5
HOME_BOOST        = 0.10    # +10% home lambda when not neutral ground

MISSING_OUT       = 0.025   # lambda reduction per "Out" player
MISSING_MAJOR     = 0.012   # per "Major doubt"
MISSING_MINOR     = 0.005   # per "Minor doubt"
MISSING_CAP       = 0.12

EDGE_THRESHOLDS   = {"HIGH": 5.0, "MEDIUM": 8.0, "LOW": 12.0}
EDGE_ANOMALY      = 25.0

# Draw scale by ELO-difference bucket (descending threshold)
_ELO_DRAW_SCALE = [(400, 0.40), (300, 0.55), (200, 0.70), (100, 0.85), (0, 1.00)]

MODEL_VERSION = "WC_v1"

# Mirrors logger._MAX_SETTLE_ATTEMPTS — specific-game rows are written with this
# value so the daily auto-settler skips them (no API id to settle against).
_MAX_SETTLE_ATTEMPTS = 10

NON_WC_ELO: dict[str, int] = {
    "Italy": 1670, "Denmark": 1640, "Serbia": 1480, "Chile": 1520,
    "Ukraine": 1530, "Nigeria": 1460, "Slovakia": 1450, "Romania": 1440,
    "Russia": 1440, "Iceland": 1430, "Wales": 1430, "Ireland": 1400,
    "Cameroon": 1400, "Finland": 1400, "Northern Ireland": 1370,
    "Albania": 1370, "Montenegro": 1370, "Bolivia": 1340, "Jamaica": 1330,
    "Angola": 1310, "Guatemala": 1300, "China": 1380, "Venezuela": 1380,
    "Zimbabwe": 1240, "Zambia": 1290, "Equatorial Guinea": 1270,
    "Mauritania": 1270, "Kazakhstan": 1280, "Latvia": 1280, "Cyprus": 1230,
    "Azerbaijan": 1240, "Botswana": 1200, "Faroe Islands": 1150,
    "Puerto Rico": 1150, "Luxembourg": 1120, "Malta": 1100,
    "Liechtenstein": 1000, "Gibraltar": 950, "Bermuda": 1050,
    "San Marino": 900, "Eswatini": 1050,
}

# ---------------------------------------------------------------------------
# Dixon-Coles maths (self-contained, no analyzer.py import needed)
# ---------------------------------------------------------------------------

_GRID = 8


def _poisson(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float) -> list[list[float]]:
    matrix = []
    for i in range(_GRID):
        row = []
        for j in range(_GRID):
            p = _poisson(lam_h, i) * _poisson(lam_a, j) * _tau(i, j, lam_h, lam_a, RHO_WC)
            row.append(max(p, 0.0))
        matrix.append(row)
    total = sum(p for row in matrix for p in row)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _probs(matrix: list[list[float]]) -> dict:
    home_win = draw = away_win = 0.0
    btts = o15 = o25 = o35 = 0.0
    for i in range(_GRID):
        for j in range(_GRID):
            p = matrix[i][j]
            if i > j:   home_win += p
            elif i == j: draw     += p
            else:        away_win += p
            if i > 0 and j > 0:  btts += p
            if i + j > 1: o15 += p
            if i + j > 2: o25 += p
            if i + j > 3: o35 += p
    return dict(home_win=home_win, draw=draw, away_win=away_win,
                btts=btts, o15=o15, o25=o25, o35=o35)


# ---------------------------------------------------------------------------
# Staking (inline — avoids staking.py's module-level bankroll state)
# ---------------------------------------------------------------------------

KELLY_FRACTION  = 0.25
UNIT_PCT        = 0.01
MAX_STAKE_UNITS = 10

def _load_bankroll() -> float:
    try:
        settings = (_HERE / "settings.json")
        if settings.exists():
            return float(json.loads(settings.read_text(encoding="utf-8")).get("bankroll_eur", 10.0))
    except Exception:
        pass
    return 10.0

def _stake_units(prob: float, odds: float, edge: float) -> int | None:
    if edge <= 0 or odds <= 1:
        return None
    f_star = (prob * odds - 1) / (odds - 1)
    if f_star <= 0:
        return None
    units = round(KELLY_FRACTION * f_star / UNIT_PCT)
    if units < 1:
        return None
    return min(units, MAX_STAKE_UNITS)

def _units_to_eur(units: int) -> float:
    return units * _load_bankroll() * UNIT_PCT

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()

def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_DIVISOR))

def _draw_scale(elo_diff: float) -> float:
    for threshold, scale in _ELO_DRAW_SCALE:
        if elo_diff >= threshold:
            return scale
    return 1.00

def _opponent_elo(name: str, wc_index: dict) -> float:
    td = wc_index.get(name)
    if td:
        return float(td.get("elo_rating") or WC_MEAN_ELO)
    return float(NON_WC_ELO.get(name, WC_MEAN_ELO))

# ---------------------------------------------------------------------------
# Form signal
# ---------------------------------------------------------------------------

def _form_signal(team: dict, ref_date: date, wc_index: dict) -> float | None:
    matches = team.get("recent_matches", [])
    if not matches:
        return None
    num = denom = 0.0
    for m in matches:
        try:
            age_days = (ref_date - _parse_date(m["date"])).days
        except Exception:
            continue
        if age_days < 0:
            continue
        decay = math.exp(-FORM_DECAY_K * age_days / 30)
        comp_w = float(m.get("weight", 0.8))
        opp_elo = _opponent_elo(m.get("opponent", ""), wc_index)
        oqf = _clamp(opp_elo / WC_MEAN_ELO, 0.5, 1.8)
        w = decay * comp_w * oqf
        num   += w * float(m.get("goals_scored", 0))
        denom += w
    return (num / denom) if denom > 0.01 else None

# ---------------------------------------------------------------------------
# Lambda computation
# ---------------------------------------------------------------------------

def _lambdas(home: dict, away: dict, ref_date: date, neutral: bool, wc_index: dict):
    elo_a = float(home.get("elo_rating") or WC_MEAN_ELO)
    elo_b = float(away.get("elo_rating") or WC_MEAN_ELO)

    p_a = _elo_win_prob(elo_a, elo_b)
    lam_elo_a = WC_TOTAL_GOALS * p_a
    lam_elo_b = WC_TOTAL_GOALS * (1.0 - p_a)

    atk_a = float(home.get("attack_strength") or 1.0) / ATK_MEAN
    atk_b = float(away.get("attack_strength") or 1.0) / ATK_MEAN
    def_a = float(home.get("defense_strength") or 1.0) / DEF_MEAN
    def_b = float(away.get("defense_strength") or 1.0) / DEF_MEAN

    lam_goals_a = WC_HALF * atk_a * def_b
    lam_goals_b = WC_HALF * atk_b * def_a

    # ELO_WEIGHT + GOALS_WEIGHT < 1 (form is a separate multiplier, not additive),
    # so normalise here to avoid systematically deflating every lambda.
    _blend_sum = ELO_WEIGHT + GOALS_WEIGHT
    base_a = (ELO_WEIGHT * lam_elo_a + GOALS_WEIGHT * lam_goals_a) / _blend_sum
    base_b = (ELO_WEIGHT * lam_elo_b + GOALS_WEIGHT * lam_goals_b) / _blend_sum

    fs_a = _form_signal(home, ref_date, wc_index)
    fs_b = _form_signal(away, ref_date, wc_index)
    mult_a = _clamp(fs_a / WC_HALF, *FORM_MULT_CLAMP) if fs_a is not None else 1.0
    mult_b = _clamp(fs_b / WC_HALF, *FORM_MULT_CLAMP) if fs_b is not None else 1.0

    la = base_a * (1.0 + FORM_WEIGHT * (mult_a - 1.0))
    lb = base_b * (1.0 + FORM_WEIGHT * (mult_b - 1.0))

    if not neutral:
        la *= (1.0 + HOME_BOOST)
        lb *= max(0.1, 1.0 - HOME_BOOST * 0.5)

    return _clamp(la, 0.1, LAM_HARD_CAP), _clamp(lb, 0.1, LAM_HARD_CAP)

# ---------------------------------------------------------------------------
# Missing player penalty
# ---------------------------------------------------------------------------

def _missing_penalty(home: dict, away: dict, la: float, lb: float):
    def pen(team: dict) -> float:
        total = 0.0
        for s in team.get("key_players_missing", []):
            sl = s.lower()
            # match only the status tag, e.g. "(Out)", "(Major doubt)", "(Minor doubt)"
            if "major doubt" in sl:        total += MISSING_MAJOR
            elif "minor doubt" in sl:      total += MISSING_MINOR
            elif "doubt" in sl:            total += MISSING_MAJOR
            elif "(out)" in sl or "(out " in sl or sl.endswith("out"): total += MISSING_OUT
            else:                          total += MISSING_OUT  # bare name = assume out
        return min(total, MISSING_CAP)
    return (
        _clamp(la * (1.0 - pen(home)), 0.1, LAM_HARD_CAP),
        _clamp(lb * (1.0 - pen(away)), 0.1, LAM_HARD_CAP),
    )

# ---------------------------------------------------------------------------
# H2H adjustment
# ---------------------------------------------------------------------------

def _h2h_adjust(home: dict, away: dict, la: float, lb: float, ref_date: date):
    away_name = away.get("team", "")
    home_name = home.get("team", "")
    records = home.get("h2h", {}).get(away_name) or away.get("h2h", {}).get(home_name)
    from_home = bool(home.get("h2h", {}).get(away_name))
    if not records:
        return la, lb

    sn = sd = cn = cd = 0.0
    for m in records:
        try:
            age_days = (ref_date - _parse_date(m["date"])).days
        except Exception:
            continue
        if age_days < 0 or age_days > H2H_MAX_AGE_DAYS:
            continue
        w = math.exp(-H2H_DECAY_K * age_days / 365.25) * float(m.get("weight", 0.8))
        sn += w * float(m.get("goals_scored", 0));  sd += w
        cn += w * float(m.get("goals_conceded", 0)); cd += w

    if sd < H2H_MIN_WEIGHT:
        return la, lb

    h2h_s = sn / sd
    h2h_c = cn / cd

    if from_home:
        nudge_a = _clamp((h2h_s - la) * 0.5, -H2H_MAX_NUDGE * la, H2H_MAX_NUDGE * la)
        nudge_b = _clamp((h2h_c - lb) * 0.5, -H2H_MAX_NUDGE * lb, H2H_MAX_NUDGE * lb)
    else:
        nudge_a = _clamp((h2h_c - la) * 0.5, -H2H_MAX_NUDGE * la, H2H_MAX_NUDGE * la)
        nudge_b = _clamp((h2h_s - lb) * 0.5, -H2H_MAX_NUDGE * lb, H2H_MAX_NUDGE * lb)

    return (
        _clamp(la + nudge_a, 0.1, LAM_HARD_CAP),
        _clamp(lb + nudge_b, 0.1, LAM_HARD_CAP),
    )

# ---------------------------------------------------------------------------
# Draw correction
# ---------------------------------------------------------------------------

def _draw_correct(ph: float, pd: float, pa: float, elo_diff: float):
    scale = _draw_scale(abs(elo_diff))
    pd2 = pd * scale
    removed = pd - pd2
    wl = ph + pa
    if wl < 1e-9:
        return ph, pd2, pa
    ph2 = ph + removed * (ph / wl)
    pa2 = pa + removed * (pa / wl)
    t = ph2 + pd2 + pa2
    return ph2 / t, pd2 / t, pa2 / t

# ---------------------------------------------------------------------------
# Value detection helpers
# ---------------------------------------------------------------------------

def _devig3(oh: float, od: float, oa: float):
    if min(oh, od, oa) <= 0:
        return 1/3, 1/3, 1/3
    t = 1/oh + 1/od + 1/oa
    return (1/oh)/t, (1/od)/t, (1/oa)/t

def _devig2(oy: float, on_: float):
    if min(oy, on_) <= 0:
        return 0.5, 0.5
    t = 1/oy + 1/on_
    return (1/oy)/t, (1/on_)/t

# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _confidence(home: dict, away: dict, max_edge: float) -> float:
    score = 1.0
    fa = len(home.get("recent_matches", [])) >= 3
    fb = len(away.get("recent_matches", [])) >= 3
    if not fa and not fb:   score -= 0.25
    elif not fa or not fb:  score -= 0.10
    if home.get("avg_goals_scored") is None and away.get("avg_goals_scored") is None:
        score -= 0.15
    elif home.get("avg_goals_scored") is None or away.get("avg_goals_scored") is None:
        score -= 0.05
    kp = len(home.get("key_players_missing", [])) + len(away.get("key_players_missing", []))
    if kp >= 3:   score -= 0.12
    elif kp >= 1: score -= 0.04
    if max_edge > EDGE_ANOMALY:
        return 0.0
    return max(0.10, min(1.0, score))

def _conf_label(score: float) -> str:
    if score >= 0.80: return "HIGH"
    if score >= 0.60: return "MEDIUM"
    return "LOW"

# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_match(home: dict, away: dict, match_info: dict, wc_index: dict | None = None) -> dict:
    wc_index = wc_index or {}

    try:
        ref_date = _parse_date(match_info.get("date", str(date.today())))
    except Exception:
        ref_date = date.today()

    neutral = bool(match_info.get("neutral_ground", True))

    la, lb = _lambdas(home, away, ref_date, neutral, wc_index)
    la, lb = _missing_penalty(home, away, la, lb)
    la, lb = _h2h_adjust(home, away, la, lb, ref_date)

    matrix = _score_matrix(la, lb)
    p = _probs(matrix)

    elo_a = float(home.get("elo_rating") or WC_MEAN_ELO)
    elo_b = float(away.get("elo_rating") or WC_MEAN_ELO)
    ph, pd, pa = _draw_correct(p["home_win"], p["draw"], p["away_win"], elo_a - elo_b)

    # Top scorelines
    cells = sorted(
        [((i, j), matrix[i][j]) for i in range(_GRID) for j in range(_GRID)],
        key=lambda x: x[1], reverse=True
    )

    # Market edges
    odds = match_info.get("odds", {})
    markets: dict[str, dict] = {}

    def _od(key: str) -> float:
        v = odds.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    o_h, o_d, o_a = _od("home_win"), _od("draw"), _od("away_win")
    if o_h > 1 and o_d > 1 and o_a > 1:
        fh, fd, fa = _devig3(o_h, o_d, o_a)
        markets["home_win"] = dict(model=ph, fair=fh, odds=o_h,
                                   pick=home.get("team","Home"), edge=(ph-fh)*100)
        markets["draw"]     = dict(model=pd, fair=fd, odds=o_d,
                                   pick="Draw", edge=(pd-fd)*100)
        markets["away_win"] = dict(model=pa, fair=fa, odds=o_a,
                                   pick=away.get("team","Away"), edge=(pa-fa)*100)

    o_o, o_u = _od("over_2_5"), _od("under_2_5")
    if o_o > 1 and o_u > 1:
        fo, fu = _devig2(o_o, o_u)
        markets["over_2_5"]  = dict(model=p["o25"], fair=fo, odds=o_o,
                                    pick="Over 2.5", edge=(p["o25"]-fo)*100)
        markets["under_2_5"] = dict(model=1-p["o25"], fair=fu, odds=o_u,
                                    pick="Under 2.5", edge=((1-p["o25"])-fu)*100)

    o_by, o_bn = _od("btts_yes"), _od("btts_no")
    if o_by > 1 and o_bn > 1:
        fy, fn = _devig2(o_by, o_bn)
        markets["btts_yes"] = dict(model=p["btts"], fair=fy, odds=o_by,
                                   pick="BTTS Yes", edge=(p["btts"]-fy)*100)
        markets["btts_no"]  = dict(model=1-p["btts"], fair=fn, odds=o_bn,
                                   pick="BTTS No", edge=((1-p["btts"])-fn)*100)

    max_edge = max((abs(v["edge"]) for v in markets.values()), default=0.0)
    conf = _confidence(home, away, max_edge)
    conf_lbl = _conf_label(conf)
    threshold = EDGE_THRESHOLDS[conf_lbl]

    value_picks = []
    for mkt_key, mkt in markets.items():
        if mkt["edge"] >= threshold and max_edge < EDGE_ANOMALY:
            units = _stake_units(mkt["model"], mkt["odds"], mkt["edge"])
            if units:
                value_picks.append(dict(
                    market_key=mkt_key,
                    pick=mkt["pick"],
                    model_prob=mkt["model"],
                    odds=mkt["odds"],
                    edge=mkt["edge"],
                    units=units,
                    eur=_units_to_eur(units),
                ))

    return dict(
        match_id     = match_info.get("match_id", "SPECIFIC"),
        competition  = match_info.get("competition", ""),
        date         = ref_date.isoformat(),
        neutral      = neutral,
        home         = home.get("team", "Home"),
        away         = away.get("team", "Away"),
        lam_a        = round(la, 3),
        lam_b        = round(lb, 3),
        p_home       = round(ph, 4),
        p_draw       = round(pd, 4),
        p_away       = round(pa, 4),
        p_btts       = round(p["btts"], 4),
        p_over_15    = round(p["o15"], 4),
        p_over_25    = round(p["o25"], 4),
        p_over_35    = round(p["o35"], 4),
        top3         = [{"score": f"{s[0]}-{s[1]}", "prob": round(pr, 4)} for s, pr in cells[:3]],
        markets      = markets,
        value_picks  = value_picks,
        confidence   = round(conf, 2),
        conf_label   = conf_lbl,
        data_anomaly = max_edge > EDGE_ANOMALY,
    )

# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

W = 60

def print_prediction(r: dict) -> None:
    neutral_str = "Neutral" if r["neutral"] else "Home/Away"
    print()
    print("═" * W)
    print(f"  {r['home']}  vs  {r['away']}")
    print(f"  {r['competition']}  │  {r['date']}  │  {neutral_str}")
    print("═" * W)

    if r["data_anomaly"]:
        print("\n  ⚠  DATA ANOMALY — edge >25pp. Review odds before placing bets.\n")
        return

    print(f"\n  xG:  {r['home']} {r['lam_a']:.2f}  vs  {r['away']} {r['lam_b']:.2f}\n")

    threshold = EDGE_THRESHOLDS[r["conf_label"]]

    # ── Headline recommendation — the single best value-hunting pick ──────────
    picks_sorted = sorted(r["value_picks"], key=lambda v: v["edge"], reverse=True)
    print("  " + "─" * (W - 2))
    if picks_sorted:
        best = picks_sorted[0]
        eur_str = f"  (€{best['eur']:.2f})" if best["eur"] else ""
        print(f"  ★ BET:  {best['pick']}  @ {best['odds']}")
        print(f"          edge {best['edge']:+.1f}pp  ·  {best['units']} unit(s){eur_str}  "
              f"·  conf {r['conf_label']}")
    else:
        # No value — still surface the closest market so the read isn't blank
        best_edge_key = max(r["markets"], key=lambda k: r["markets"][k]["edge"], default=None)
        if best_edge_key:
            bm = r["markets"][best_edge_key]
            print(f"  ✗ NO BET  ·  best edge {bm['pick']} {bm['edge']:+.1f}pp "
                  f"(< {threshold:.0f}pp {r['conf_label']} threshold)")
        else:
            print("  ✗ NO BET  ·  no odds supplied")
    print("  " + "─" * (W - 2))
    print()

    hdr = f"  {'Market':<18} {'Model':>7} {'Fair':>7} {'Edge':>9}"
    print(hdr)
    print("  " + "─" * (W - 2))

    order = [
        ("home_win",  f"{r['home']} Win"),
        ("draw",      "Draw"),
        ("away_win",  f"{r['away']} Win"),
        ("over_2_5",  "Over 2.5"),
        ("under_2_5", "Under 2.5"),
        ("btts_yes",  "BTTS Yes"),
        ("btts_no",   "BTTS No"),
    ]
    for key, label in order:
        m = r["markets"].get(key)
        if not m:
            continue
        flag = "  ✓ VALUE" if m["edge"] >= threshold else ""
        print(f"  {label:<18} {m['model']:>6.1%} {m['fair']:>6.1%} {m['edge']:>+8.1f}pp{flag}")

    print()
    if r["value_picks"]:
        print(f"  VALUE PICKS  (confidence: {r['conf_label']}  {r['confidence']:.0%})")
        print("  " + "─" * (W - 2))
        for vp in r["value_picks"]:
            eur_str = f"  €{vp['eur']:.2f}" if vp["eur"] else ""
            print(f"  ✓  {vp['pick']:<18} @ {vp['odds']:<6}  "
                  f"edge {vp['edge']:+.1f}pp  │  {vp['units']} unit(s){eur_str}")
    else:
        print(f"  No value picks  (confidence: {r['conf_label']}, threshold: {threshold:.0f}pp)")

    print()
    top3_str = "   ".join(f"{t['score']} ({t['prob']:.1%})" for t in r["top3"])
    print(f"  Top scorelines:  {top3_str}")
    print("═" * W)
    print()

# ---------------------------------------------------------------------------
# Bets log
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "match_id","date","home","away","league","market","pick",
    "model_prob","odds_taken","edge","result","roi","settle_attempts",
    "home_position","away_position","form_adv","expected_total",
    "model_version","stake_units",
]

def log_picks(r: dict) -> None:
    if not r["value_picks"] or r["data_anomaly"]:
        return
    write_header = not BETS_LOG.exists() or BETS_LOG.stat().st_size == 0
    with BETS_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if write_header:
            w.writeheader()
        for vp in r["value_picks"]:
            mkt = {"home_win":"1X2","draw":"1X2","away_win":"1X2",
                   "over_2_5":"Over/Under","under_2_5":"Over/Under",
                   "btts_yes":"BTTS","btts_no":"BTTS"}.get(vp["market_key"], vp["market_key"])
            w.writerow({
                "match_id":       r["match_id"],
                "date":           r["date"],
                "home":           r["home"],
                "away":           r["away"],
                "league":         r["competition"],
                "market":         mkt,
                "pick":           vp["pick"],
                "model_prob":     round(vp["model_prob"], 4),
                "odds_taken":     vp["odds"],
                "edge":           round(vp["edge"], 1),
                # Pre-mark so the daily auto-settler (which settles only rows with
                # result=="") skips these — specific/friendly games have no API id.
                "result":         "MANUAL",
                "roi":            "",
                "settle_attempts": _MAX_SETTLE_ATTEMPTS,
                "home_position":  "",
                "away_position":  "",
                "form_adv":       "",
                "expected_total": round(r["lam_a"] + r["lam_b"], 2),
                "model_version":  MODEL_VERSION,
                "stake_units":    vp["units"],
            })
    print(f"  → {len(r['value_picks'])} pick(s) logged to bets_log.csv")

# ---------------------------------------------------------------------------
# Specific game entry point
# ---------------------------------------------------------------------------

def _load_wc_index() -> dict:
    index: dict[str, dict] = {}
    teams_dir = Path(__file__).parent / "wc2026" / "teams"
    for tf in teams_dir.glob("*.json"):
        try:
            td = json.loads(tf.read_text(encoding="utf-8"))
            index[td["team"]] = td
        except Exception:
            pass
    return index


def predict_specific_game(json_path: Path | None = None) -> dict | None:
    if json_path is None:
        json_path = Path(__file__).parent / "specific_game.json"

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ✗  Cannot read {json_path}: {e}")
        return None

    home = data.get("home_team", {})
    away = data.get("away_team", {})
    match_info = data.get("match", {})

    # Validation
    missing = []
    if not home.get("team"):       missing.append("home_team.team")
    if not away.get("team"):       missing.append("away_team.team")
    if not home.get("elo_rating"): missing.append("home_team.elo_rating")
    if not away.get("elo_rating"): missing.append("away_team.elo_rating")
    if missing:
        print(f"  ✗  Missing required fields: {', '.join(missing)}")
        return None

    odds = match_info.get("odds", {})
    if all(v == 0.0 for v in odds.values()):
        print("  ✗  All odds are 0.0 — fill in real bookmaker odds before predicting.")
        return None

    wc_index = _load_wc_index()
    result = predict_match(home, away, match_info, wc_index)
    print_prediction(result)
    log_picks(result)
    return result
