import math
from datetime import datetime, timezone
from config import (BASELINES, FORM_WEIGHTS, HOME_ADV,
                    KNOCKOUT_GOALS_FACTOR, KNOCKOUT_OVER_PENALTY, KNOCKOUT_BTTS_PENALTY,
                    DC_RHO, DC_RHO_DEFAULT, FORM_DECAY_K, H2H_DECAY_K,
                    LEAGUE_SEASON_CONFIG, SEASON_CONF_RAMP,
                    FATIGUE_SEVERE_DAYS, FATIGUE_MODERATE_DAYS,
                    FATIGUE_PENALTY_FULL, FATIGUE_PENALTY_MOD,
                    XG_CONV_BY_LEAGUE,
                    MOMENTUM_THRESHOLD, MOMENTUM_BOOST, MOMENTUM_PENALTY,
                    STREAK_MIN_LENGTH, STREAK_MAX_LENGTH,
                    STREAK_CS_DEFENCE_BOOST, STREAK_DROUGHT_ATK_PENALTY)
import warn_log

# Decay constants are defined in config.py — see FORM_DECAY_K and H2H_DECAY_K.

# xG proxy conversion: shots-on-target × XG_CONV = xG estimate.
# Module-level so the backtest sweep can patch it without touching the production API.
XG_CONV = 0.33


# ---------------------------------------------------------------------------
# Standings parsing
# ---------------------------------------------------------------------------

def parse_standings(standings_data, league_code):
    """Return dict keyed by team_id with standing stats."""
    teams = {}
    if not standings_data or "standings" not in standings_data:
        return teams

    for group in standings_data["standings"]:
        if group.get("type") == "TOTAL":
            for entry in group.get("table", []):
                team = entry.get("team", {})
                team_id = team.get("id")
                if not team_id:
                    continue

                played = entry.get("playedGames") or 1
                goals_for = entry.get("goalsFor", 0)
                goals_against = entry.get("goalsAgainst", 0)
                form_str = entry.get("form") or ""

                teams[team_id] = {
                    "name": team.get("name", "Unknown"),
                    "position": entry.get("position", 10),
                    "points": entry.get("points", 0),
                    "played": played,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "avg_scored": goals_for / played,
                    "avg_conceded": goals_against / played,
                    "form_str": form_str,
                    "form_score": _compute_form_score(form_str),
                    "league": league_code,
                }
            break

    return teams


def _compute_form_score(form_str):
    """Weighted form score normalised to 0–1."""
    if not form_str:
        return 0.5

    results = list(form_str.upper())[-6:]
    score = 0.0
    max_score = 0.0

    for i, result in enumerate(results):
        w = FORM_WEIGHTS[i] if i < len(FORM_WEIGHTS) else 0.05
        if result == "W":
            score += 3 * w
        elif result == "D":
            score += 1 * w
        max_score += 3 * w

    return score / max_score if max_score > 0 else 0.5


# ---------------------------------------------------------------------------
# Team match history parsing
# ---------------------------------------------------------------------------

def _form_decay_weight(utc_date_str: str, reference: datetime = None) -> float:
    """
    Return exponential decay weight for a team match: exp(-FORM_DECAY_K × age_days / 30).
    reference: optional anchor date (for backtesting); defaults to today.
    Matches with no parseable date get weight 1.
    """
    if not utc_date_str:
        return 1.0
    try:
        dt = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
        now = reference or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age_days = max((now - dt).days, 0)
        return math.exp(-FORM_DECAY_K * age_days / 30)
    except (ValueError, TypeError):
        return 1.0


def parse_team_history(matches_data, team_id, strength_factors=None, reference_date=None, league=None):
    """
    Parse last 30 finished matches into split home/away rolling stats.
    A.1: Goals and rates are weighted by recency — recent matches count more.
    Weight = exp(-0.5 × age_days / 30), so a 30-day-old match has ~61% the
    influence of a match played today.
    C.3: BTTS and Over/Under rates are additionally weighted by opponent defensive
    strength. Scoring against a top defence (defence index ~0.7) earns more credit
    than scoring against a weak defence (defence index ~1.5).
    Opponent weight = 1 / opponent_defence_index (stronger defence → higher weight).
    Clamped to [0.5, 2.0] to prevent extreme single-match distortion.
    """
    # Per-league SoT→xG conversion; falls back to the module-level global for sweeps.
    xg_conv = XG_CONV_BY_LEAGUE.get(league, XG_CONV) if league else XG_CONV

    # Weighted accumulators: (weighted_sum, weight_total)
    # Scored/conceded use opponent-adjusted denominators (MODEL-03).
    home_scored_w = home_conceded_w = 0.0
    away_scored_w = away_conceded_w = 0.0
    home_scored_tot = home_conceded_tot = 0.0   # opponent-adjusted denominators
    away_scored_tot = away_conceded_tot = 0.0
    home_cs_w = away_cs_w = 0.0
    home_w_total = away_w_total = 0.0           # recency-only (for CS rate)

    # C1/Phase2: xG proxy accumulators from shots-on-target data.
    # xG_proxy = shots_on_target × xg_conv (per-league value, patchable for sweeps).
    home_xg_w = home_xg_conceded_w = 0.0
    away_xg_w = away_xg_conceded_w = 0.0
    home_xg_total = away_xg_total = 0.0   # separate weight total (only matches with shot data)

    btts_w = over25_w = over35_w = all_w_total = 0.0

    if not matches_data or "matches" not in matches_data:
        warn_log.fallback("no match history", "default_team_stats")
        return _default_team_stats()

    for match in matches_data["matches"]:
        ft = match.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue

        w = _form_decay_weight(match.get("utcDate", ""), reference=reference_date)
        total = hg + ag

        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")

        # C1/Phase2: extract shots-on-target for xG proxy.
        # Matches may carry shot data from SQLite (home_shots_on_target field)
        # or from the API response (statistics dict).
        h_sot = match.get("home_shots_on_target")
        a_sot = match.get("away_shots_on_target")
        if h_sot is None or a_sot is None:
            stats = match.get("statistics", {})
            if isinstance(stats, dict):
                h_sot = h_sot if h_sot is not None else stats.get("home", {}).get("shotsOnTarget")
                a_sot = a_sot if a_sot is not None else stats.get("away", {}).get("shotsOnTarget")
        has_shots = h_sot is not None and a_sot is not None

        # C.3 / MODEL-03: opponent-difficulty weights.
        # scored_w:   1/opp_def — scoring vs strong defence earns more credit.
        # conceded_w: 1/opp_atk — conceding vs strong attack is expected (less blame).
        # Both clamped to [0.5, 2.0] to prevent single-match distortion.
        opp_scored_w = opp_conceded_w = 1.0
        if strength_factors:
            opp_sf = None
            if home_id == team_id:
                opp_sf = strength_factors.get(away_id)
            elif away_id == team_id:
                opp_sf = strength_factors.get(home_id)
            if opp_sf:
                opp_def = opp_sf.get("defence", 1.0)
                opp_atk = opp_sf.get("attack", 1.0)
                opp_scored_w   = max(0.5, min(2.0, 1.0 / opp_def)) if opp_def > 0 else 1.0
                opp_conceded_w = max(0.5, min(2.0, 1.0 / opp_atk)) if opp_atk > 0 else 1.0
        opp_w = opp_scored_w   # BTTS/Over rates use scored-side weighting (existing behaviour)

        combined_w = w * opp_w

        btts_w    += combined_w * (1 if hg > 0 and ag > 0 else 0)
        over25_w  += combined_w * (1 if total > 2.5 else 0)
        over35_w  += combined_w * (1 if total > 3.5 else 0)
        all_w_total += combined_w

        if home_id == team_id:
            home_scored_w    += w * opp_scored_w   * hg
            home_scored_tot  += w * opp_scored_w
            home_conceded_w  += w * opp_conceded_w * ag
            home_conceded_tot += w * opp_conceded_w
            home_cs_w        += w * (1 if ag == 0 else 0)
            home_w_total     += w
            if has_shots:
                home_xg_w          += w * (h_sot * xg_conv)
                home_xg_conceded_w += w * (a_sot * xg_conv)
                home_xg_total      += w
        elif away_id == team_id:
            away_scored_w    += w * opp_scored_w   * ag
            away_scored_tot  += w * opp_scored_w
            away_conceded_w  += w * opp_conceded_w * hg
            away_conceded_tot += w * opp_conceded_w
            away_cs_w        += w * (1 if hg == 0 else 0)
            away_w_total     += w
            if has_shots:
                away_xg_w          += w * (a_sot * xg_conv)
                away_xg_conceded_w += w * (h_sot * xg_conv)
                away_xg_total      += w

    def wavg(num, denom):
        return num / denom if denom > 0 else 0.0

    # C.1: League position trend — last 5 matches points vs season PPG.
    # Counts raw wins/draws/losses from the 5 most recent finished matches
    # (regardless of home/away split) to derive recent_ppg.
    recent_results = []   # list of "W"/"D"/"L" from team's perspective, newest first
    for m in matches_data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")
        winner = m.get("score", {}).get("winner", "")
        if h_id == team_id:
            if winner == "HOME_TEAM":
                recent_results.append("W")
            elif winner == "DRAW":
                recent_results.append("D")
            else:
                recent_results.append("L")
        elif a_id == team_id:
            if winner == "AWAY_TEAM":
                recent_results.append("W")
            elif winner == "DRAW":
                recent_results.append("D")
            else:
                recent_results.append("L")
        if len(recent_results) >= 5:
            break   # only need the 5 most recent

    if len(recent_results) >= 5:
        pts = sum(3 if r == "W" else (1 if r == "D" else 0) for r in recent_results)
        recent_ppg = pts / len(recent_results)   # points per game over last 5
    else:
        recent_ppg = None

    # A.5: Goals variance as xG proxy.
    # Collect raw goals-scored values for variance calculation.
    # High variance (e.g. 0,0,4,0,3) → team creates bursts of chances;
    # low variance (1,1,2,1,1) → steady but not explosive.
    # We compute std(goals_scored) across all finished matches and store it
    # so compute_match_probabilities() can apply a small variance nudge.
    goals_scored_list = []
    for m in matches_data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")
        if h_id == team_id:
            goals_scored_list.append(hg)
        elif a_id == team_id:
            goals_scored_list.append(ag)

    if len(goals_scored_list) >= 5:
        mean_g = sum(goals_scored_list) / len(goals_scored_list)
        variance = sum((g - mean_g) ** 2 for g in goals_scored_list) / len(goals_scored_list)
        goals_std = math.sqrt(variance)
    else:
        goals_std = None   # not enough data

    # B1: venue-split form scores — last 5 home results and last 5 away results.
    # More predictive than the standings form string which mixes both venues.
    # Also collect all_results (any venue, chronological) for momentum signal.
    home_results = []
    away_results = []
    all_results  = []   # last 6 regardless of venue — for momentum computation
    for m in matches_data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        if ft.get("home") is None or ft.get("away") is None:
            continue
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")
        winner = m.get("score", {}).get("winner", "")
        result_for_team = None
        if h_id == team_id:
            result_for_team = "W" if winner == "HOME_TEAM" else ("D" if winner == "DRAW" else "L")
            if len(home_results) < 6:
                home_results.append(result_for_team)
        elif a_id == team_id:
            result_for_team = "W" if winner == "AWAY_TEAM" else ("D" if winner == "DRAW" else "L")
            if len(away_results) < 6:
                away_results.append(result_for_team)
        if result_for_team is not None and len(all_results) < 6:
            all_results.append(result_for_team)

    def _split_form(results):
        if not results:
            return None
        score = max_score = 0.0
        for i, r in enumerate(results):
            w = FORM_WEIGHTS[i] if i < len(FORM_WEIGHTS) else 0.05
            if r == "W":
                score += 3 * w
            elif r == "D":
                score += 1 * w
            max_score += 3 * w
        return score / max_score if max_score > 0 else 0.5

    home_form_score = _split_form(home_results)
    away_form_score = _split_form(away_results)

    # Form momentum: compare last-3 score vs last-6 score (any venue).
    # Positive delta = upswing; negative = downswing. None when < 3 results.
    if len(all_results) >= 3:
        recent_3 = _split_form(all_results[:3])
        recent_6 = _split_form(all_results)     # up to 6
        momentum = (recent_3 or 0.5) - (recent_6 or 0.5)
    else:
        momentum = None

    # C1/Phase2: xG proxy averages from shots-on-target.
    # Only populated when enough shot-data weight accumulates.
    # 2.0 = ~3 matches within 20 days. Previous value of 3.5 required 4+ matches
    # over a 30-day span (weight ~2.89), so xG almost never activated.
    _MIN_XG_WEIGHT = 2.0
    xg_home = wavg(home_xg_w, home_xg_total) if home_xg_total >= _MIN_XG_WEIGHT else None
    xg_conceded_home = wavg(home_xg_conceded_w, home_xg_total) if home_xg_total >= _MIN_XG_WEIGHT else None
    xg_away = wavg(away_xg_w, away_xg_total) if away_xg_total >= _MIN_XG_WEIGHT else None
    xg_conceded_away = wavg(away_xg_conceded_w, away_xg_total) if away_xg_total >= _MIN_XG_WEIGHT else None

    # Alert when shot data is sparse: if >30% of matches lack shots, xG is unreliable.
    # home_xg_total / home_w_total = fraction of home-match weight that had shot data.
    _SHOT_COVERAGE_MIN = 0.70
    if home_w_total > 0 and (home_xg_total / home_w_total) < _SHOT_COVERAGE_MIN:
        warn_log.fallback(
            f"low home shot-data coverage ({home_xg_total:.1f}/{home_w_total:.1f} = {home_xg_total/home_w_total:.0%})",
            "xG unavailable; raw goals used",
            match_id=str(team_id),
        )
    if away_w_total > 0 and (away_xg_total / away_w_total) < _SHOT_COVERAGE_MIN:
        warn_log.fallback(
            f"low away shot-data coverage ({away_xg_total:.1f}/{away_w_total:.1f} = {away_xg_total/away_w_total:.0%})",
            "xG unavailable; raw goals used",
            match_id=str(team_id),
        )

    # POTENTIAL-06: streak detection (clean sheets and scoring droughts).
    # Walk the ordered match list tail-first; stop as soon as the streak is broken.
    def _streak(matches, team_id_local, condition_fn):
        count = 0
        for m in matches:
            ft = m.get("score", {}).get("fullTime", {})
            hg = ft.get("home")
            ag = ft.get("away")
            if hg is None or ag is None:
                continue
            h_id = m.get("homeTeam", {}).get("id")
            a_id = m.get("awayTeam", {}).get("id")
            if h_id != team_id_local and a_id != team_id_local:
                continue
            is_home = h_id == team_id_local
            if condition_fn(hg, ag, is_home):
                count += 1
            else:
                break
        return count

    all_matches = matches_data.get("matches", [])
    cs_streak   = _streak(all_matches, team_id, lambda hg, ag, ih: (ag == 0 if ih else hg == 0))
    drought_streak = _streak(all_matches, team_id, lambda hg, ag, ih: (hg == 0 if ih else ag == 0))

    return {
        "avg_scored_home":      wavg(home_scored_w,    home_scored_tot  or home_w_total),
        "avg_conceded_home":    wavg(home_conceded_w,  home_conceded_tot or home_w_total),
        "avg_scored_away":      wavg(away_scored_w,    away_scored_tot  or away_w_total),
        "avg_conceded_away":    wavg(away_conceded_w,  away_conceded_tot or away_w_total),
        "clean_sheet_rate_home": wavg(home_cs_w,       home_w_total),
        "clean_sheet_rate_away": wavg(away_cs_w,       away_w_total),
        "btts_rate":            wavg(btts_w,           all_w_total),
        "over_2_5_rate":        wavg(over25_w,         all_w_total),
        "over_3_5_rate":        wavg(over35_w,         all_w_total),
        "goals_std":            goals_std,
        "recent_ppg":           recent_ppg,
        "home_form_score":      home_form_score,   # B1: venue-split form (None if no data)
        "away_form_score":      away_form_score,
        "momentum":             momentum,          # recent_3_form - recent_6_form delta
        "cs_streak":            cs_streak,         # consecutive clean sheets (any venue)
        "drought_streak":       drought_streak,    # consecutive matches without scoring
        # C1/Phase2: xG proxy from shots-on-target (None if insufficient data)
        "xg_scored_home":       xg_home,
        "xg_conceded_home":     xg_conceded_home,
        "xg_scored_away":       xg_away,
        "xg_conceded_away":     xg_conceded_away,
        # Use raw match counts for the split guard (not weighted sums)
        "home_games": sum(
            1 for m in matches_data.get("matches", [])
            if m.get("score", {}).get("fullTime", {}).get("home") is not None
            and m.get("homeTeam", {}).get("id") == team_id
        ),
        "away_games": sum(
            1 for m in matches_data.get("matches", [])
            if m.get("score", {}).get("fullTime", {}).get("home") is not None
            and m.get("awayTeam", {}).get("id") == team_id
        ),
    }


def build_reason(pick_name, market, probs, home_name, away_name,
                 home_standing, away_standing, h2h):
    """
    Build a one-line human-readable reason string for a pick.
    Uses data already computed in compute_match_probabilities().
    """
    parts = []

    # 1. Goals context — always first
    exp_h = probs.get("expected_home_goals", 0)
    exp_a = probs.get("expected_away_goals", 0)
    if exp_h or exp_a:
        parts.append(f"xG {exp_h:.1f}–{exp_a:.1f}")

    # 2. Market-specific detail — directly relevant to the pick
    if market in ("Over/Under", "BTTS"):
        cs_h = probs.get("home_cs_rate", 0)
        cs_a = probs.get("away_cs_rate", 0)
        parts.append(f"CS {cs_h:.0%}/{cs_a:.0%}")
    if market == "1X2":
        hs = probs.get("home_avg_scored", 0)
        as_ = probs.get("away_avg_scored", 0)
        parts.append(f"Avg scored {hs:.1f}/{as_:.1f}")

    # 3. Fatigue / knockout flags — actionable warnings, must not get cut off
    if probs.get("away_fatigue"):
        parts.append("⚠ away fatigue")
    if probs.get("home_fatigue"):
        parts.append("⚠ home fatigue")
    if probs.get("is_knockout"):
        parts.append("⚠ knockout leg")

    # 4. H2H summary — useful context
    meetings = h2h.get("meetings", 0) if h2h else 0
    if meetings >= 5:
        hw = h2h.get("home_wins", 0)
        dr = h2h.get("draws", 0)
        aw = h2h.get("away_wins", 0)
        parts.append(f"H2H {meetings}g: {hw:.0f}W-{dr:.0f}D-{aw:.0f}L")

    # 5. Form strings — last; already encoded in xG, least critical if truncated
    hf = home_standing.get("form_str", "")
    af = away_standing.get("form_str", "")
    if hf or af:
        parts.append(f"Form {hf or '?'} vs {af or '?'}")

    return "  ·  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Referee tendency
# ---------------------------------------------------------------------------

# Minimum matches to consider a referee's stats meaningful.
_MIN_REF_MATCHES = 8


def compute_referee_factor(ref_name, home_hist_raw, away_hist_raw, league_avg_gpg=2.6):
    """
    Compute a goals-per-game multiplier for a specific referee.

    Scans the home and away team's recent match history (the raw API dicts)
    for finished matches officiated by ref_name, then computes the referee's
    average total goals per game relative to the league average.

    Returns a float multiplier clamped to [0.85, 1.15]:
      > 1.0 → this referee's games tend to have more goals (lenient, few fouls)
      < 1.0 → fewer goals (strict, cards-heavy, cautious play)
      1.0   → no adjustment (unknown referee or insufficient data)

    Parameters
    ----------
    ref_name:         Name string from the fixture's referees list.
    home_hist_raw:    Raw API response from get_team_matches() for home team.
    away_hist_raw:    Raw API response from get_team_matches() for away team.
    league_avg_gpg:   League average total goals per game (default 2.6).
    """
    if not ref_name:
        return 1.0

    ref_lower = ref_name.strip().lower()
    total_goals = []

    for hist_raw in [home_hist_raw, away_hist_raw]:
        if not hist_raw or "matches" not in hist_raw:
            continue
        for m in hist_raw["matches"]:
            refs = m.get("referees", [])
            match_ref = ""
            for r in refs:
                if r.get("type", "").upper() == "REFEREE":
                    match_ref = (r.get("name") or "").strip().lower()
                    break
            if match_ref != ref_lower:
                continue

            ft = m.get("score", {}).get("fullTime", {})
            hg = ft.get("home")
            ag = ft.get("away")
            if hg is None or ag is None:
                continue
            total_goals.append(hg + ag)

    # Deduplicate: if both teams played the same match with this ref,
    # it would appear twice. Use match_id to deduplicate.
    seen_ids = set()
    deduped_goals = []
    for hist_raw in [home_hist_raw, away_hist_raw]:
        if not hist_raw or "matches" not in hist_raw:
            continue
        for m in hist_raw["matches"]:
            mid = m.get("id")
            if mid in seen_ids:
                continue
            refs = m.get("referees", [])
            match_ref = ""
            for r in refs:
                if r.get("type", "").upper() == "REFEREE":
                    match_ref = (r.get("name") or "").strip().lower()
                    break
            if match_ref != ref_lower:
                continue
            ft = m.get("score", {}).get("fullTime", {})
            hg = ft.get("home")
            ag = ft.get("away")
            if hg is None or ag is None:
                continue
            seen_ids.add(mid)
            deduped_goals.append(hg + ag)

    if len(deduped_goals) < _MIN_REF_MATCHES:
        return 1.0

    ref_avg = sum(deduped_goals) / len(deduped_goals)
    if league_avg_gpg <= 0:
        return 1.0

    factor = ref_avg / league_avg_gpg
    return max(0.85, min(1.15, factor))


def _default_team_stats():
    return {
        "avg_scored_home": 1.1,
        "avg_conceded_home": 1.1,
        "avg_scored_away": 1.1,
        "avg_conceded_away": 1.1,
        "clean_sheet_rate_home": 0.25,
        "clean_sheet_rate_away": 0.20,
        "btts_rate": 0.50,
        "over_2_5_rate": 0.50,
        "over_3_5_rate": 0.30,
        "goals_std": None,
        "recent_ppg": None,
        "home_form_score": None,
        "away_form_score": None,
        "xg_scored_home": None,
        "xg_conceded_home": None,
        "xg_scored_away": None,
        "xg_conceded_away": None,
        "home_games": 0,
        "away_games": 0,
    }


# ---------------------------------------------------------------------------
# H2H parsing
# ---------------------------------------------------------------------------

def _h2h_decay_weight(utc_date_str: str) -> float:
    """
    Return exponential decay weight for a H2H match: exp(-H2H_DECAY_K × age_days / 30).
    Uses the same formula as _form_decay_weight so k values are directly comparable.
    Matches with no parseable date get weight 1.
    """
    if not utc_date_str:
        return 1.0
    try:
        dt = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_days = max((now - dt).days, 0)
        return math.exp(-H2H_DECAY_K * age_days / 30)
    except (ValueError, TypeError):
        return 1.0


def parse_h2h(h2h_data, home_team_id=None):
    """
    Parse head-to-head API response into summary stats.
    Outcomes are weighted by recency (exponential decay) so older meetings
    have less influence than recent ones.

    home_team_id: when supplied, only H2H matches where this team played at
    home are counted (venue split). This prevents road wins from polluting
    the home-advantage signal. Falls back to all meetings when no ID given
    or fewer than 3 venue-specific matches exist.
    """
    result = {
        "meetings": 0,
        "home_wins": 0,
        "draws": 0,
        "away_wins": 0,
        "total_goals": [],
        "btts_count": 0,
        "weight_total": 0.0,
    }

    if not h2h_data:
        return result

    matches = h2h_data.get("matches", [])

    def _process(match_list, venue_split=False):
        r = {
            "meetings": 0, "home_wins": 0.0, "draws": 0.0, "away_wins": 0.0,
            "total_goals": [], "btts_count": 0.0, "weight_total": 0.0,
            "venue_split": venue_split,   # A2: track whether goals come from venue-filtered data
        }
        for match in match_list:
            ft = match.get("score", {}).get("fullTime", {})
            hg = ft.get("home")
            ag = ft.get("away")
            if hg is None or ag is None:
                continue
            r["meetings"] += 1
            w = _h2h_decay_weight(match.get("utcDate", ""))
            r["weight_total"] += w
            r["total_goals"].append(hg + ag)
            if hg > 0 and ag > 0:
                r["btts_count"] += w
            winner = match.get("score", {}).get("winner")
            if winner == "HOME_TEAM":
                r["home_wins"] += w
            elif winner == "DRAW":
                r["draws"]     += w
            elif winner == "AWAY_TEAM":
                r["away_wins"] += w
        return r

    # Venue-split: only use matches where home_team_id was the home side
    if home_team_id is not None:
        venue_matches = [
            m for m in matches
            if m.get("homeTeam", {}).get("id") == home_team_id
        ]
        if len(venue_matches) >= 3:
            return _process(venue_matches, venue_split=True)
        warn_log.fallback(
            f"H2H venue-split thin ({len(venue_matches)} matches)",
            "all H2H meetings (orientation-mixed)",
        )

    # Fallback: use all meetings — goals list is orientation-mixed, flag it
    return _process(matches, venue_split=False)


# ---------------------------------------------------------------------------
# Probability model
# ---------------------------------------------------------------------------

def poisson_prob(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def _tau(x, y, lam_h, lam_a, rho):
    """
    Dixon-Coles correction factor for low-score cells.
    Adjusts the independent Poisson joint probability for
    (0,0), (1,0), (0,1), (1,1) scorelines.
    rho=0  → no correction (pure Poisson).
    rho≈−0.13 is the empirically estimated value from the original paper.
    """
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


# Goals grid size — 8 covers >99.9% of real scorelines
_GRID = 8


def _score_matrix(lam_h, lam_a, rho=DC_RHO_DEFAULT):
    """
    Build an 8×8 matrix of corrected joint scoreline probabilities.
    rho: Dixon-Coles correlation parameter (league-specific via DC_RHO config).
    Returns matrix[home_goals][away_goals].
    """
    matrix = []
    for i in range(_GRID):
        row = []
        for j in range(_GRID):
            p = poisson_prob(lam_h, i) * poisson_prob(lam_a, j) * _tau(i, j, lam_h, lam_a, rho)
            row.append(max(p, 0.0))   # tau can be slightly negative for extreme lambdas
        matrix.append(row)

    # Renormalise so probabilities sum to 1 (grid truncation + tau corrections)
    total = sum(p for row in matrix for p in row)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _probs_from_matrix(matrix):
    """Derive 1X2, BTTS, and over/under probabilities from a score matrix."""
    home_win = draw = away_win = 0.0
    btts = over_15 = over_25 = over_35 = 0.0

    for i in range(_GRID):
        for j in range(_GRID):
            p = matrix[i][j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i > 0 and j > 0:
                btts += p
            if i + j > 1:
                over_15 += p
            if i + j > 2:
                over_25 += p
            if i + j > 3:
                over_35 += p

    return home_win, draw, away_win, btts, over_15, over_25, over_35


def _compute_strength_factors(standings: dict, team_histories: dict = None) -> dict:
    """
    Compute per-team attack and defence strength indices relative to the league mean.

    Returns {
        team_id: {
            "attack":         float,   # overall (from standings)
            "defence":        float,   # overall (from standings)
            "home_attack":    float,   # home-specific (from split history if available)
            "home_defence":   float,   # home-specific
            "away_attack":    float,   # away-specific
            "away_defence":   float,   # away-specific
        },
        "_league_avg_scored":   float,
        "_league_avg_conceded": float,
    }

    Home/away split indices use team_histories (parsed split stats per team_id).
    Falls back to overall index when split history is unavailable or thin.

    attack  > 1  → scores more than average
    defence < 1  → concedes less than average  (lower = better)
    """
    if not standings:
        return {}

    avgs_scored   = [v["avg_scored"]   for v in standings.values() if v.get("avg_scored")]
    avgs_conceded = [v["avg_conceded"] for v in standings.values() if v.get("avg_conceded")]

    if not avgs_scored or not avgs_conceded:
        return {}

    league_avg_scored   = sum(avgs_scored)   / len(avgs_scored)
    league_avg_conceded = sum(avgs_conceded) / len(avgs_conceded)

    factors: dict = {
        "_league_avg_scored":   league_avg_scored,
        "_league_avg_conceded": league_avg_conceded,
    }

    # Build split-based indices from team histories when available.
    # The divisor is always the full-league overall average (computed from all
    # standings entries above) — NOT a 2-team sample from the current fixture.
    # Typical home/away scoring ratios: home teams score ~15% more than overall
    # average; away teams ~15% less. Using the full-league avg as denominator
    # for both overall and split indices keeps everything on the same scale.
    _MIN_SPLIT = 4   # minimum games to trust a split average

    for team_id, s in standings.items():
        # Clamp all indices to [0.40, 2.00] — prevents elite attack × poor defence
        # from producing unrealistic xG (e.g. 4.9 goals). In practice the best
        # attackers in top leagues sit around 1.7–1.8× league avg; 2.0 is a hard ceiling.
        def _clamp_idx(v): return max(0.40, min(2.00, v))

        overall_attack  = _clamp_idx(s["avg_scored"]   / league_avg_scored)   if league_avg_scored   > 0 else 1.0
        overall_defence = _clamp_idx(s["avg_conceded"] / league_avg_conceded) if league_avg_conceded > 0 else 1.0

        hist = (team_histories or {}).get(team_id, {})

        # Home-specific indices: team's home avg vs full-league overall avg.
        # This correctly captures "this team scores X% more/less than the typical
        # team when playing at home" without polluting the reference with a 2-team mean.
        if hist.get("home_games", 0) >= _MIN_SPLIT and league_avg_scored > 0:
            home_attack  = _clamp_idx(hist["avg_scored_home"]   / league_avg_scored)
            home_defence = _clamp_idx(hist["avg_conceded_home"] / league_avg_conceded) if league_avg_conceded > 0 else overall_defence
        else:
            home_attack  = overall_attack
            home_defence = overall_defence

        # Away-specific indices: same logic for away venue.
        if hist.get("away_games", 0) >= _MIN_SPLIT and league_avg_scored > 0:
            away_attack  = _clamp_idx(hist["avg_scored_away"]   / league_avg_scored)
            away_defence = _clamp_idx(hist["avg_conceded_away"] / league_avg_conceded) if league_avg_conceded > 0 else overall_defence
        else:
            away_attack  = overall_attack
            away_defence = overall_defence

        factors[team_id] = {
            "attack":       overall_attack,
            "defence":      overall_defence,
            "home_attack":  home_attack,
            "home_defence": home_defence,
            "away_attack":  away_attack,
            "away_defence": away_defence,
        }
    return factors


def prob_over(lam_total, threshold=2.5):
    return 1.0 - sum(poisson_prob(lam_total, k) for k in range(int(threshold) + 1))


def prob_btts(lam_home, lam_away):
    return (1 - poisson_prob(lam_home, 0)) * (1 - poisson_prob(lam_away, 0))


def compute_motivation_factor(standing: dict, total_teams: int = 20) -> float:
    """
    Estimate a team's motivation multiplier based on league position and
    games remaining in the season.

    Returns a float in [0.85, 1.12]:
      1.12 → must-win desperation (only wins can still save/advance)
      1.10 → title race / relegation six-pointer
      1.06 → European spots still in play
      1.00 → neutral mid-table
      0.92 → already safe with nothing to play for
      0.88 → title sealed (no pressure)
      0.85 → already relegated (season over mentally)

    Requires standing to include a "league" key matching LEAGUE_SEASON_CONFIG.
    """
    position  = standing.get("position", total_teams // 2)
    points    = standing.get("points", 0)
    played    = standing.get("played", 20)
    league    = standing.get("league", "")

    cfg = LEAGUE_SEASON_CONFIG.get(league, {"season_games": 38, "safety_pts": 36, "relegated": 3})
    season_length = cfg["season_games"]
    safety_pts    = cfg["safety_pts"]
    relegated     = cfg["relegated"]

    games_left = max(0, season_length - played)
    max_pts    = points + games_left * 3

    relegation_cutoff = total_teams - relegated + 1   # position >= this = danger zone

    # --- Dead rubbers (motivation killers) ---
    if position >= relegation_cutoff and max_pts < safety_pts - 5:
        return 0.85   # mathematically relegated

    if position == 1 and games_left <= 3 and points >= safety_pts * 2:
        return 0.88   # title sealed

    if 8 <= position <= total_teams - 4 and points >= safety_pts + 8 and games_left <= 6:
        return 0.92   # safe mid-table, nothing to play for

    # --- Must-win: a draw is not enough ---
    # Relegation: in danger zone, ≤4 games left, points deficit can only be closed by wins
    pts_behind_safety = safety_pts - points
    if (position >= relegation_cutoff and games_left <= 4
            and pts_behind_safety > games_left):       # draws can't close the gap
        return 1.12

    # Title: top 2, ≤3 games left, within 2 pts — must win every game
    if position <= 2 and games_left <= 3 and points >= safety_pts * 1.8:
        return 1.12

    # --- High-motivation ---
    if position <= 3 and games_left >= 3:
        return 1.10   # title race

    if position <= 6 and games_left >= 4:
        return 1.06   # European spots

    if position >= total_teams - relegated - 1 and points < safety_pts + 3:
        return 1.08   # relegation battle, still winnable

    return 1.00


def _days_since(utc_date_str: str) -> float | None:
    """Return days elapsed since utc_date_str, or None if unparseable."""
    if not utc_date_str:
        return None
    try:
        dt = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except (ValueError, TypeError):
        return None


def compute_match_probabilities(league_code, home_standing, away_standing,
                                 home_history, away_history, h2h,
                                 strength_factors=None,
                                 away_last_match_date=None,
                                 home_last_match_date=None,
                                 is_knockout=False,
                                 referee_factor=1.0,
                                 total_teams=20):
    """Compute 1X2 and goals market probabilities for a fixture.

    strength_factors:       output of _compute_strength_factors(standings).
    away_last_match_date:   utcDate string of the away team's last match.
    home_last_match_date:   utcDate string of the home team's last match.
    is_knockout:            True for CL/cup knockout-stage legs. Suppresses
                            expected goals and penalises Over/Under + BTTS
                            confidence to reflect defensive tactical setups.
    referee_factor:         Goals-per-game multiplier for the assigned referee.
                            > 1.0 = more goals expected; < 1.0 = fewer.
    Fatigue penalty applied when a team played within 3 days.
    """
    if league_code not in BASELINES:
        warn_log.fallback("league not in BASELINES", "global default {home:0.45, draw:0.27, away:0.28}", league=league_code)
    baseline = BASELINES.get(league_code, {"home": 0.45, "draw": 0.27, "away": 0.28})

    home_prob = baseline["home"]
    draw_prob = baseline["draw"]
    away_prob = baseline["away"]

    # --- Season-stage confidence scaling ---
    # Adjustments are unreliable early in the season. Scale them linearly
    # from 0 (0 games played) to 1.0 (10+ games played).
    games_played = min(
        home_standing.get("played", 10),
        away_standing.get("played", 10),
    )
    season_conf = min(1.0, games_played / SEASON_CONF_RAMP)

    # --- Form adjustment ---
    # B1: prefer venue-split form when available — home team's home form and
    # away team's away form are far more predictive than mixed overall form.
    # Fall back to standings form_score when split history is thin.
    home_form = home_history.get("home_form_score") or home_standing.get("form_score", 0.5)
    away_form = away_history.get("away_form_score") or away_standing.get("form_score", 0.5)
    form_adv = home_form - away_form

    # B6: adjust home and away independently; draw is reduced only when one side
    # is clearly dominant (large form gap). Draw is NOT a residual of the two moves —
    # it is anchored to the league baseline and only explicitly reduced.
    if form_adv > 0.30:
        home_prob += 0.05 * season_conf
        away_prob -= 0.05 * season_conf
        draw_prob -= 0.02 * season_conf   # dominant home side → fewer draws
    elif form_adv < -0.30:
        away_prob += 0.05 * season_conf
        home_prob -= 0.05 * season_conf
        draw_prob -= 0.02 * season_conf   # dominant away side → fewer draws

    # --- Position adjustment ---
    pos_diff = away_standing.get("position", 10) - home_standing.get("position", 10)
    if pos_diff > 5:
        home_prob += 0.03 * season_conf
        away_prob -= 0.03 * season_conf
        draw_prob -= 0.01 * season_conf
    elif pos_diff < -5:
        away_prob += 0.03 * season_conf
        home_prob -= 0.03 * season_conf
        draw_prob -= 0.01 * season_conf

    # --- C.1: Position trend modifier ---
    # A team's table position is a lagging indicator. Compare recent_ppg (last 5)
    # to season_ppg (standings). If recent form is much better than season avg,
    # nudge their probability up slightly (and vice versa).
    # Max nudge: ±0.02 per team (small — position is already accounted for above).
    _C1_SCALE = 0.02
    home_played = home_standing.get("played", 0)
    away_played = away_standing.get("played", 0)
    if home_played >= 5:
        season_ppg_h = home_standing.get("points", 0) / home_played
        recent_ppg_h = home_history.get("recent_ppg")
        if recent_ppg_h is not None and season_ppg_h > 0:
            trend_h = min(1.5, max(0.5, recent_ppg_h / season_ppg_h))  # clamp ratio
            nudge_h = (trend_h - 1.0) * _C1_SCALE * season_conf
            home_prob += nudge_h
            away_prob -= nudge_h * 0.6
            draw_prob -= nudge_h * 0.4

    if away_played >= 5:
        season_ppg_a = away_standing.get("points", 0) / away_played
        recent_ppg_a = away_history.get("recent_ppg")
        if recent_ppg_a is not None and season_ppg_a > 0:
            trend_a = min(1.5, max(0.5, recent_ppg_a / season_ppg_a))
            nudge_a = (trend_a - 1.0) * _C1_SCALE * season_conf
            away_prob += nudge_a
            home_prob -= nudge_a * 0.6
            draw_prob -= nudge_a * 0.4

    # Normalise
    total = home_prob + draw_prob + away_prob
    if total <= 0:
        home_prob, draw_prob, away_prob = 1/3, 1/3, 1/3
    else:
        home_prob /= total
        draw_prob /= total
        away_prob /= total
    assert abs(home_prob + draw_prob + away_prob - 1.0) < 1e-6, f"1X2 probs don't sum to 1 after baseline normalisation: {home_prob + draw_prob + away_prob}"

    # --- H2H modifier (only if ≥5 meetings, using decay-weighted rates) ---
    if h2h and h2h.get("meetings", 0) >= 5:
        weight_total = h2h.get("weight_total") or h2h["meetings"]
        h2h_home_rate = h2h["home_wins"] / weight_total
        h2h_home_diff = h2h_home_rate - baseline["home"]
        nudge = min(abs(h2h_home_diff) * 0.5, 0.05) * season_conf

        if h2h_home_diff > 0.10:
            home_prob += nudge
            away_prob -= nudge * 0.7
            draw_prob -= nudge * 0.3
        elif h2h_home_diff < -0.10:
            away_prob += nudge
            home_prob -= nudge * 0.7
            draw_prob -= nudge * 0.3

        total = home_prob + draw_prob + away_prob
        if total <= 0:
            home_prob, draw_prob, away_prob = 1/3, 1/3, 1/3
        else:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        assert abs(home_prob + draw_prob + away_prob - 1.0) < 1e-6, f"1X2 probs don't sum to 1 after H2H blend: {home_prob + draw_prob + away_prob}"

    # --- Goals model ---
    # Use home/away split stats only when the sample is large enough (≥4 games).
    # With fewer games the split averages are too noisy; fall back to standing avg.
    _MIN_SPLIT = 4

    if home_history.get("home_games", 0) >= _MIN_SPLIT:
        home_avg_scored   = home_history["avg_scored_home"]
        home_avg_conceded = home_history["avg_conceded_home"]
        home_has_split = True
    else:
        warn_log.fallback(
            f"home split thin ({home_history.get('home_games', 0)} home games)",
            "standing avg_scored/avg_conceded",
            league=league_code,
            match_id=str(home_standing.get("id", "")),
        )
        home_avg_scored   = home_standing.get("avg_scored", 1.2)
        home_avg_conceded = home_standing.get("avg_conceded", 1.2)
        home_has_split = False

    if away_history.get("away_games", 0) >= _MIN_SPLIT:
        away_avg_scored   = away_history["avg_scored_away"]
        away_avg_conceded = away_history["avg_conceded_away"]
    else:
        warn_log.fallback(
            f"away split thin ({away_history.get('away_games', 0)} away games)",
            "standing avg_scored/avg_conceded",
            league=league_code,
            match_id=str(away_standing.get("id", "")),
        )
        away_avg_scored   = away_standing.get("avg_scored", 1.0)
        away_avg_conceded = away_standing.get("avg_conceded", 1.4)

    # Save raw (pre-xG-blend) split averages for use in the trend factor below.
    # The trend must compare like-for-like units (raw goals vs raw goals).
    # Mixing xG-blended values with raw season averages skews the ratio.
    raw_home_avg_scored = home_avg_scored
    raw_away_avg_scored = away_avg_scored

    # C1/Phase2: Blend xG proxy from shots-on-target into scoring averages.
    # SoT × fixed_factor is a rough proxy — no shot location, body part, or assist type.
    # Giving it 60% weight caused systematic over-prediction of expected goals (O/U WR 52%
    # at avg model_prob 0.64 — 12pp calibration gap). Reduced to 35% so raw goals remain
    # the primary signal; xG acts as a smoothing correction rather than the dominant input.
    _XG_BLEND = 0.35
    if home_history.get("xg_scored_home") is not None:
        home_avg_scored   = _XG_BLEND * home_history["xg_scored_home"]   + (1 - _XG_BLEND) * home_avg_scored
        home_avg_conceded = _XG_BLEND * home_history["xg_conceded_home"] + (1 - _XG_BLEND) * home_avg_conceded
    else:
        warn_log.fallback("home xG unavailable", "raw goals avg", league=league_code, match_id=str(home_standing.get("id", "")))
    if away_history.get("xg_scored_away") is not None:
        away_avg_scored   = _XG_BLEND * away_history["xg_scored_away"]   + (1 - _XG_BLEND) * away_avg_scored
        away_avg_conceded = _XG_BLEND * away_history["xg_conceded_away"] + (1 - _XG_BLEND) * away_avg_conceded
    else:
        warn_log.fallback("away xG unavailable", "raw goals avg", league=league_code, match_id=str(away_standing.get("id", "")))

    # A.3: Proper Poisson regression goals model.
    # Formula: exp_goals = league_avg × attacker_strength × opponent_defence_strength
    # This avoids the averaging bias: a 3.0 attacker vs 0.5 defence team gave
    # (3.0+0.5)/2 = 1.75 with old formula; correct answer is 3.0 × 0.5 = 1.5 × league_avg.
    # strength_factors: {team_id: {"attack": float, "defence": float}}
    # attack > 1 means above-average scoring; defence < 1 means below-average conceding.
    home_id = home_standing.get("id")
    away_id = away_standing.get("id")

    if strength_factors and home_id in strength_factors and away_id in strength_factors:
        home_sf = strength_factors[home_id]
        away_sf = strength_factors[away_id]
        league_avg = strength_factors.get("_league_avg_scored", 1.35)

        # Home/away split Poisson regression (Tier 1 #2):
        # exp_home = league_avg × home_team_HOME_attack × away_team_AWAY_defence
        # exp_away = league_avg × away_team_AWAY_attack × home_team_HOME_defence
        # This captures venue-specific tendencies: a team that attacks well at
        # home vs a team that defends poorly away gives a much sharper estimate
        # than using overall (mixed) indices for both teams.
        exp_home = league_avg * home_sf["home_attack"] * away_sf["away_defence"]
        exp_away = league_avg * away_sf["away_attack"] * home_sf["home_defence"]
    else:
        warn_log.fallback(
            "strength_factors missing or team not in factors",
            "averaging formula (home_avg_scored + away_avg_conceded) / 2",
            league=league_code,
            match_id=f"{home_id}v{away_id}",
        )
        exp_home = (home_avg_scored + away_avg_conceded) / 2
        exp_away = (away_avg_scored + home_avg_conceded) / 2

    # --- C.2: Goals-per-game trend factor ---
    # Compares the team's raw home/away split average to an adjusted season
    # baseline so the ratio captures true recent form, not the home-advantage
    # effect (which strength_factors already encode).
    #
    # Denominator = home_season_avg * home_adv_ratio, where home_adv_ratio is
    # HOME_ADV for the league.  This approximates the "expected" home split avg
    # for an average team in form, giving a denominator on the same venue scale
    # as the numerator and avoiding double-counting with home_attack in A.3.
    #
    # Uses raw (pre-xG-blend) split averages so units are consistent with the
    # raw-goals season average from standings.
    _TREND_CLAMP = (0.85, 1.20)   # tighter clamp — trend is a fine-tuning signal

    home_season_avg = home_standing.get("avg_scored", 0)
    if home_season_avg > 0.2 and home_history.get("home_games", 0) >= _MIN_SPLIT:
        # Normalise by expected home scoring rate (season avg × home adv) so the
        # ratio is near 1.0 for a team in typical form and only deviates when
        # recent scoring is genuinely above/below their venue-adjusted baseline.
        home_adj_baseline = home_season_avg * HOME_ADV.get(league_code, 1.10)
        home_trend = max(_TREND_CLAMP[0], min(_TREND_CLAMP[1], raw_home_avg_scored / home_adj_baseline))
        exp_home *= home_trend

    away_season_avg = away_standing.get("avg_scored", 0)
    if away_season_avg > 0.2 and away_history.get("away_games", 0) >= _MIN_SPLIT:
        # Away teams typically score ~15% less than their overall avg away from
        # home; 1/HOME_ADV approximates that expected away rate.
        away_adj_baseline = away_season_avg / HOME_ADV.get(league_code, 1.10)
        away_trend = max(_TREND_CLAMP[0], min(_TREND_CLAMP[1], raw_away_avg_scored / away_adj_baseline))
        exp_away *= away_trend

    # --- A.5: Goals variance nudge ---
    # A team with high goals variance creates more total chances than their mean
    # suggests (boom/bust pattern). Use std(goals_scored) as a proxy for shot
    # volume and apply a small upward nudge to expected goals.
    # Typical Poisson std ≈ sqrt(mean). Excess std above that = extra volatility.
    # Nudge = clamp((actual_std - poisson_std) / poisson_std, -0.10, +0.10)
    # so the adjustment is always relative and bounded.
    _VAR_NUDGE_MAX = 0.10   # cap at ±10%

    home_std = home_history.get("goals_std")
    if home_std is not None and exp_home > 0:
        poisson_std_h = math.sqrt(exp_home)
        excess_h = (home_std - poisson_std_h) / poisson_std_h
        nudge_h = max(-_VAR_NUDGE_MAX, min(_VAR_NUDGE_MAX, excess_h * 0.5))
        exp_home *= (1.0 + nudge_h)

    away_std = away_history.get("goals_std")
    if away_std is not None and exp_away > 0:
        poisson_std_a = math.sqrt(exp_away)
        excess_a = (away_std - poisson_std_a) / poisson_std_a
        nudge_a = max(-_VAR_NUDGE_MAX, min(_VAR_NUDGE_MAX, excess_a * 0.5))
        exp_away *= (1.0 + nudge_a)

    # --- Home advantage multiplier (A.2) ---
    # Only apply when home_avg_scored came from venue-split history (home games only).
    # When it fell back to standings avg (mixed home+away), home advantage is already
    # implicit in that average — applying HOME_ADV on top would double-count it.
    home_adv = HOME_ADV.get(league_code, 1.10)
    if home_has_split:
        exp_home *= home_adv

    # --- Motivation factor (Tier 1 #3) ---
    # Teams fighting for the title, European spots, or survival play harder.
    # Dead-rubber teams (already safe/relegated/champions) play with less intensity.
    # Applied as a multiplier on each team's expected goals: a highly motivated
    # team scores more AND concedes less (opponent's xG is divided by their motivation).
    home_motiv = compute_motivation_factor(home_standing, total_teams=total_teams)
    away_motiv = compute_motivation_factor(away_standing, total_teams=total_teams)
    exp_home *= home_motiv
    exp_away *= away_motiv

    # --- Referee tendency ---
    # referee_factor > 1 means this ref's matches average more goals.
    # Apply symmetrically to both teams' expected goals.
    if referee_factor != 1.0:
        exp_home *= referee_factor
        exp_away *= referee_factor

    exp_total = exp_home + exp_away

    # --- Fatigue penalty (threshold model, §4.4) ---
    # Step-function based on research consensus that <3 days rest is the meaningful
    # threshold. The old continuous exp(-days/7) applied penalty all the way to 7 days
    # (4.4% at 7 days) with no real evidence for that tail.
    #   ≤2 days: full penalty (back-to-back schedule, squad rotation typical)
    #   3 days:  moderate penalty (midweek turnaround, standard busy period)
    #   ≥4 days: no penalty (adequate recovery)
    away_fatigue = home_fatigue = False

    def _fatigue_penalty(days: float | None) -> float:
        if days is None or days >= FATIGUE_MODERATE_DAYS + 1:
            return 0.0
        if days <= FATIGUE_SEVERE_DAYS:
            return FATIGUE_PENALTY_FULL
        return FATIGUE_PENALTY_MOD   # exactly 3 days

    away_days = _days_since(away_last_match_date)
    penalty_a = _fatigue_penalty(away_days)
    if penalty_a > 0:
        exp_away *= (1.0 - penalty_a)
        exp_home *= (1.0 + penalty_a * 0.5)
        away_fatigue = True

    home_days = _days_since(home_last_match_date)
    penalty_h = _fatigue_penalty(home_days)
    if penalty_h > 0:
        exp_home *= (1.0 - penalty_h)
        exp_away *= (1.0 + penalty_h * 0.5)
        home_fatigue = True

    # Form momentum adjustment (POTENTIAL-02).
    # Apply a small attack boost/penalty based on recent trend (last 3 vs last 6 results).
    home_momentum = home_history.get("momentum")
    away_momentum = away_history.get("momentum")
    if home_momentum is not None:
        if home_momentum > MOMENTUM_THRESHOLD:
            exp_home *= MOMENTUM_BOOST
        elif home_momentum < -MOMENTUM_THRESHOLD:
            exp_home *= MOMENTUM_PENALTY
    if away_momentum is not None:
        if away_momentum > MOMENTUM_THRESHOLD:
            exp_away *= MOMENTUM_BOOST
        elif away_momentum < -MOMENTUM_THRESHOLD:
            exp_away *= MOMENTUM_PENALTY

    # Streak adjustments (POTENTIAL-06).
    # Clean sheet streak: team is conceding very little recently → reduce opponent's expected goals.
    # Scoring drought: team hasn't scored recently → reduce their own expected goals.
    def _streak_factor(streak: int, per_match_delta: float) -> float:
        if streak < STREAK_MIN_LENGTH:
            return 1.0
        effective = min(streak, STREAK_MAX_LENGTH) - STREAK_MIN_LENGTH + 1
        return 1.0 - (effective * per_match_delta)

    home_cs     = home_history.get("cs_streak", 0)
    away_cs     = away_history.get("cs_streak", 0)
    home_drought = home_history.get("drought_streak", 0)
    away_drought = away_history.get("drought_streak", 0)

    exp_away *= _streak_factor(home_cs,      STREAK_CS_DEFENCE_BOOST)
    exp_home *= _streak_factor(away_cs,      STREAK_CS_DEFENCE_BOOST)
    exp_home *= _streak_factor(home_drought, STREAK_DROUGHT_ATK_PENALTY)
    exp_away *= _streak_factor(away_drought, STREAK_DROUGHT_ATK_PENALTY)

    exp_total = exp_home + exp_away

    # Blend with H2H avg goals — scale both legs proportionally so the
    # score matrix stays consistent with exp_total.
    # A2: only blend when goals come from venue-split data (home_team always on left).
    # Mixed (home+away legs combined) goals averages are orientation-ambiguous and
    # can nudge total goals in the wrong direction.
    if h2h and h2h.get("meetings", 0) >= 5 and h2h.get("total_goals") and h2h.get("venue_split"):
        h2h_avg      = sum(h2h["total_goals"]) / len(h2h["total_goals"])
        exp_total_new = exp_total * 0.7 + h2h_avg * 0.3
        if exp_total > 0:
            ratio    = exp_total_new / exp_total
            exp_home *= ratio
            exp_away *= ratio
        exp_total = exp_total_new

    # --- Knockout stage adjustment ---
    # CL/cup knockout legs: teams play conservatively (defend first, play for draw).
    # Suppress expected goals and apply market penalties after probability derivation.
    if is_knockout:
        exp_home *= KNOCKOUT_GOALS_FACTOR
        exp_away *= KNOCKOUT_GOALS_FACTOR
        exp_total = exp_home + exp_away

    # Hard cap: no realistic match has more than 3.5 xG per team.
    # Prevents runaway multiplier chains from producing absurd scorelines.
    exp_home = min(exp_home, 3.5)
    exp_away = min(exp_away, 3.5)
    exp_total = exp_home + exp_away

    # --- Dixon-Coles score matrix ---
    matrix = _score_matrix(exp_home, exp_away, rho=DC_RHO.get(league_code, DC_RHO_DEFAULT))
    dc_home, dc_draw, dc_away, dc_btts, dc_over_15, dc_over_25, dc_over_35 = _probs_from_matrix(matrix)

    # Blend DC 1X2 with the form/position-adjusted probs (DC improves low-score
    # accuracy; form/position carry context the goal model can't see directly).
    # A.4: raise blend to 0.8 — DC draw probability is the most accurate part.
    blend = 0.8
    home_prob = home_prob * (1 - blend) + dc_home * blend
    draw_prob = draw_prob * (1 - blend) + dc_draw * blend
    away_prob = away_prob * (1 - blend) + dc_away * blend
    total = home_prob + draw_prob + away_prob
    if total <= 0:
        home_prob, draw_prob, away_prob = 1/3, 1/3, 1/3
    else:
        home_prob /= total
        draw_prob /= total
        away_prob /= total
    assert abs(home_prob + draw_prob + away_prob - 1.0) < 1e-6, f"1X2 probs don't sum to 1 after Dixon-Coles blend: {home_prob + draw_prob + away_prob}"

    # Goals probabilities from the corrected matrix
    p_over_15 = dc_over_15
    p_over_25 = dc_over_25
    p_over_35 = dc_over_35
    p_btts    = dc_btts

    # Reinforce with rolling rates (weighted blend)
    rolling_btts = (home_history.get("btts_rate", 0.5) + away_history.get("btts_rate", 0.5)) / 2

    # B.1: three-way BTTS blend — DC matrix + rolling rate + CS model.
    # cs_model estimates P(BTTS No) = P(home CS) + P(away CS) - P(both CS).
    # We use it as a BTTS No weight, so BTTS Yes weight = 1 - cs_no_prob.
    cs_h = home_history.get("clean_sheet_rate_home", 0.25)
    cs_a = away_history.get("clean_sheet_rate_away", 0.20)
    cs_btts_no_prob = cs_h + cs_a - cs_h * cs_a   # P(at least one CS)
    cs_btts_yes_prob = 1.0 - cs_btts_no_prob
    p_btts = 0.5 * dc_btts + 0.3 * rolling_btts + 0.2 * cs_btts_yes_prob

    rolling_over25 = (home_history.get("over_2_5_rate", 0.5) + away_history.get("over_2_5_rate", 0.5)) / 2
    p_over_25 = p_over_25 * 0.7 + rolling_over25 * 0.3

    rolling_over35 = (home_history.get("over_3_5_rate", 0.3) + away_history.get("over_3_5_rate", 0.3)) / 2
    p_over_35 = p_over_35 * 0.7 + rolling_over35 * 0.3

    # Apply knockout market penalties after all blending is done.
    # Reduces Over/Under and BTTS confidence to reflect defensive setups.
    if is_knockout:
        p_over_15 *= KNOCKOUT_OVER_PENALTY
        p_over_25 *= KNOCKOUT_OVER_PENALTY
        p_over_35 *= KNOCKOUT_OVER_PENALTY
        p_btts    *= KNOCKOUT_BTTS_PENALTY

    return {
        "home": home_prob,
        "draw": draw_prob,
        "away": away_prob,
        "over_1_5": p_over_15,
        "under_1_5": 1 - p_over_15,
        "over_2_5": p_over_25,
        "under_2_5": 1 - p_over_25,
        "over_3_5": p_over_35,
        "under_3_5": 1 - p_over_35,
        "btts_yes": p_btts,
        "btts_no": 1 - p_btts,
        "expected_home_goals": exp_home,
        "expected_away_goals": exp_away,
        "expected_total": exp_total,
        "form_adv": form_adv,
        "home_form": home_form,
        "away_form": away_form,
        "home_cs_rate": home_history.get("clean_sheet_rate_home", 0.25),
        "away_cs_rate": away_history.get("clean_sheet_rate_away", 0.20),
        "home_avg_scored": home_avg_scored,
        "away_avg_scored": away_avg_scored,
        "away_fatigue": away_fatigue,
        "home_fatigue": home_fatigue,
        "is_knockout":  is_knockout,
        "score_matrix": matrix,
    }
