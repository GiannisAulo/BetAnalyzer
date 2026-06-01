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

# Value gate is on EV% (expected return = model_prob × odds − 1), NOT on
# probability-point edge. pp-edge under-rates longshot value: a +3pp edge at
# 4.40 odds is +14% EV, while +3pp at 1.20 is barely +0.5%. EV% is the metric
# that actually drives ROI, so the threshold and anomaly guard both use it.
EV_THRESHOLDS     = {"HIGH": 3.0, "MEDIUM": 5.0, "LOW": 8.0}   # min EV% to bet
EV_ANOMALY        = 50.0   # EV% above this ⇒ likely odds/data error, no bet
# Don't bet outcomes the model rates below this probability. Tail estimates are
# unreliable — a tiny error there inflates EV on longshots into false value
# (e.g. an 11% underdog at 13.0 odds). Mirrors main.py's MIN_PROB floors.
MIN_PICK_PROB     = 0.15

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

def _confidence(home: dict, away: dict, max_ev: float) -> float:
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
    if max_ev > EV_ANOMALY:
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

    # Edge is computed vs the RAW implied probability (1/odds), matching
    # staking.py — this is the true EV gate (model_prob > 1/odds). The de-vigged
    # "fair" line is kept only as an informational market-consensus reference.
    def _mkt(model_p: float, odds_v: float, fair_p: float, pick: str) -> dict:
        implied = 1.0 / odds_v
        return dict(model=model_p, fair=fair_p, implied=implied, odds=odds_v,
                    pick=pick, edge=(model_p - implied) * 100,
                    ev=(model_p * odds_v - 1.0) * 100)

    o_h, o_d, o_a = _od("home_win"), _od("draw"), _od("away_win")
    if o_h > 1 and o_d > 1 and o_a > 1:
        fh, fd, fa = _devig3(o_h, o_d, o_a)
        markets["home_win"] = _mkt(ph, o_h, fh, home.get("team", "Home"))
        markets["draw"]     = _mkt(pd, o_d, fd, "Draw")
        markets["away_win"] = _mkt(pa, o_a, fa, away.get("team", "Away"))

    o_o, o_u = _od("over_2_5"), _od("under_2_5")
    if o_o > 1 and o_u > 1:
        fo, fu = _devig2(o_o, o_u)
        markets["over_2_5"]  = _mkt(p["o25"],     o_o, fo, "Over 2.5")
        markets["under_2_5"] = _mkt(1 - p["o25"], o_u, fu, "Under 2.5")

    o_by, o_bn = _od("btts_yes"), _od("btts_no")
    if o_by > 1 and o_bn > 1:
        fy, fn = _devig2(o_by, o_bn)
        markets["btts_yes"] = _mkt(p["btts"],     o_by, fy, "BTTS Yes")
        markets["btts_no"]  = _mkt(1 - p["btts"], o_bn, fn, "BTTS No")

    # Anomaly detection considers only reliable outcomes (model prob ≥ floor):
    # a huge EV on a trustworthy outcome signals a real odds error, whereas a
    # huge EV on a sub-floor longshot is just tail noise (handled by the floor).
    max_ev = max((v["ev"] for v in markets.values() if v["model"] >= MIN_PICK_PROB),
                 default=0.0)
    conf = _confidence(home, away, max_ev)
    conf_lbl = _conf_label(conf)
    threshold = EV_THRESHOLDS[conf_lbl]

    candidates = []
    for mkt_key, mkt in markets.items():
        # Skip unreliable tail outcomes — model probability too low to trust.
        if mkt["model"] < MIN_PICK_PROB:
            continue
        if mkt["ev"] >= threshold and mkt["ev"] < EV_ANOMALY:
            units = _stake_units(mkt["model"], mkt["odds"], mkt["edge"])
            if units:
                candidates.append(dict(
                    market_key=mkt_key,
                    pick=mkt["pick"],
                    model_prob=mkt["model"],
                    odds=mkt["odds"],
                    edge=mkt["edge"],
                    ev=mkt["ev"],
                    units=units,
                    eur=_units_to_eur(units),
                ))
    # One single best value bet per match (highest EV) — same as main.py, which
    # surfaces only the top-edge market per fixture. No dutching two outcomes.
    candidates.sort(key=lambda v: v["ev"], reverse=True)
    value_picks = candidates[:1]

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
        data_anomaly = max_ev > EV_ANOMALY,
    )

# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

W = 60

def print_prediction(r: dict) -> None:
    print()
    print(f"  {r['home']} vs {r['away']}  ·  {r['date']}")

    if r["data_anomaly"]:
        print(f"  ⚠  NO BET — a reliable outcome shows >{EV_ANOMALY:.0f}% EV, "
              f"which usually means the odds are misaligned with the fixture. "
              f"Double-check the odds.\n")
        return

    if r["value_picks"]:
        for vp in r["value_picks"]:
            eur_str = f" (€{vp['eur']:.2f})" if vp["eur"] else ""
            print(f"  ★ {vp['pick']} @ {vp['odds']}  "
                  f"·  +{vp['ev']:.1f}% EV  ·  {vp['units']}u{eur_str}")
    else:
        print("  ✗ No value bet")
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

def _market_label(key: str) -> str:
    return {"home_win": "1X2", "draw": "1X2", "away_win": "1X2",
            "over_2_5": "Over/Under", "under_2_5": "Over/Under",
            "btts_yes": "BTTS", "btts_no": "BTTS"}.get(key, key)


def log_picks(r: dict) -> None:
    if not r["value_picks"] or r["data_anomaly"]:
        return

    # Load existing rows, then drop any with this match_id so re-running the same
    # fixture overrides its previous picks instead of appending duplicates.
    existing: list[dict] = []
    if BETS_LOG.exists() and BETS_LOG.stat().st_size > 0:
        with BETS_LOG.open("r", newline="", encoding="utf-8") as f:
            existing = [row for row in csv.DictReader(f)
                        if row.get("match_id") != r["match_id"]]

    new_rows = []
    for vp in r["value_picks"]:
        new_rows.append({
            "match_id":        r["match_id"],
            "date":            r["date"],
            "home":            r["home"],
            "away":            r["away"],
            "league":          r["competition"],
            "market":          _market_label(vp["market_key"]),
            "pick":            vp["pick"],
            "model_prob":      round(vp["model_prob"], 4),
            "odds_taken":      vp["odds"],
            "edge":            round(vp["edge"], 1),
            # Pre-mark so the daily auto-settler (which settles only rows with
            # result=="") skips these — specific/friendly games have no API id.
            "result":          "MANUAL",
            "roi":             "",
            "settle_attempts": _MAX_SETTLE_ATTEMPTS,
            "home_position":   "",
            "away_position":   "",
            "form_adv":        "",
            "expected_total":  round(r["lam_a"] + r["lam_b"], 2),
            "model_version":   MODEL_VERSION,
            "stake_units":     vp["units"],
        })

    with BETS_LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerows(new_rows)

    print(f"  → {len(new_rows)} pick(s) written to bets_log.csv (match_id {r['match_id']})")

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
