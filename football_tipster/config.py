import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. "
        "Add API_KEY=<your_key> to a .env file in the project root."
    )
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"

_leagues_env = os.getenv("LEAGUES", "")
LEAGUES = [l.strip().upper() for l in _leagues_env.split(",") if l.strip()] \
    if _leagues_env else ["PL", "PD", "BL1", "SA", "FL1", "CL", "PPL", "DED", "ELC", "BSA"]

BASELINES = {
    "PL":  {"home": 0.46, "draw": 0.25, "away": 0.29},
    "PD":  {"home": 0.46, "draw": 0.26, "away": 0.28},
    "BL1": {"home": 0.44, "draw": 0.24, "away": 0.32},
    "SA":  {"home": 0.44, "draw": 0.30, "away": 0.26},
    "FL1": {"home": 0.45, "draw": 0.28, "away": 0.27},
    "CL":  {"home": 0.43, "draw": 0.27, "away": 0.30},
    "PPL": {"home": 0.45, "draw": 0.27, "away": 0.28},
    "DED": {"home": 0.45, "draw": 0.26, "away": 0.29},
    "ELC": {"home": 0.42, "draw": 0.27, "away": 0.32},   # English Championship
    "BSA": {"home": 0.49, "draw": 0.28, "away": 0.23},   # Campeonato Brasileiro Série A
}

# Home advantage multiplier for expected goals.
# Fitted from multi-season scoring rates; >1 means home team scores more.
# These are applied in compute_match_probabilities() to exp_home.
HOME_ADV = {
    "PL":  1.12,
    "PD":  1.10,
    "BL1": 1.13,
    "SA":  1.10,
    "FL1": 1.11,
    "CL":  1.08,
    "PPL": 1.11,
    "DED": 1.12,
    "ELC": 1.16,   # Championship: home 1.38 vs away 1.18 goals → ratio 1.17
    "BSA": 1.50,   # Brazil: home 1.56 vs away 1.04 goals → very strong home advantage
}

# CL/cup knockout stages: teams are tactically conservative — defend first,
# play for the draw, avoid conceding. The goals model needs adjusted parameters.
# stage strings returned by football-data.org API:
CL_KNOCKOUT_STAGES = {
    "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL",
    "ROUND_OF_16",   # alternate API spelling
}

# In knockout games: apply a mild goals suppression for tactical conservatism.
# Empirical data (CL 2023/24, 29 knockout + 96 group-stage matches via
# scripts/validate_knockout_factor.py): knockout avg 3.35 vs group avg 3.08
# — ratio 1.08, i.e. knockouts were NOT more defensive in that season.
# Previous value of 0.78 was set from 4 bets (textbook overfitting).
# Reverted to near-neutral (0.93) pending more seasons of data.
# Re-run validate_knockout_factor.py once API access to 2019-2022 is available.
KNOCKOUT_GOALS_FACTOR = 0.93   # mild suppression — empirical basis: 1 CL season (29 KO matches)
KNOCKOUT_OVER_PENALTY = 0.95   # reduce Over 2.5 prob by 5%
KNOCKOUT_BTTS_PENALTY = 0.95   # reduce BTTS Yes prob by 5%

# Dixon-Coles correlation parameter ρ per league.
# Negative ρ inflates 0-0/1-0/0-1 cells and deflates 1-1 relative to pure Poisson.
# High-scoring leagues (BL1, DED) need a smaller correction; defensive leagues (SA) more.
# Values derived from multi-season 0-0 / 1-1 scoreline frequency analysis.
DC_RHO = {
    "PL":  -0.13,   # English — original Dixon-Coles estimate, well-validated
    "PD":  -0.12,   # La Liga — slightly more open than PL
    "BL1": -0.10,   # Bundesliga — high-scoring, fewer 0-0 draws
    "SA":  -0.15,   # Serie A — defensive, more 0-0/1-0 outcomes
    "FL1": -0.13,   # Ligue 1 — similar to PL
    "CL":  -0.11,   # Champions League — high quality, fewer 0-0 than domestic
    "PPL": -0.13,
    "DED": -0.10,   # Eredivisie — high-scoring like BL1
    "ELC": -0.14,   # Championship — lower quality, more 0-0 per game
    "BSA": -0.11,   # Brazil — open attacking style
}
DC_RHO_DEFAULT = -0.13

FORM_WEIGHTS = [0.30, 0.22, 0.18, 0.13, 0.10, 0.07]

DEFAULT_MIN_EDGE = 5.0
RATE_LIMIT_SLEEP = 6  # seconds between API requests

# Recency decay constants — both use the formula exp(-k × age_days / 30).
# FORM_DECAY_K applies to team rolling stats (last 30 matches).
#   k=0.5 → a match 30 days ago counts 61%, 90 days ago counts 22%.
#   Aggressive decay is correct: a team's form from 3 months ago is stale.
# H2H_DECAY_K applies to head-to-head history (up to 10 meetings, often years apart).
#   k=0.05 → a meeting 30 days ago counts 95%, 1 year ago counts 83%, 3 years ago counts 60%.
#   Slow decay is correct: fixture-pair tendencies (tactical matchups, venue effects)
#   are stable over multiple seasons. Using k=0.5 would make any H2H match >6 months
#   old near-worthless, destroying the signal.
# Both constants should be re-evaluated once backtest.py (2.1) is built —
# sweep k ∈ {0.3, 0.4, 0.5} for form and k ∈ {0.03, 0.05, 0.08} for H2H,
# pick the values that minimise Brier score per market.
FORM_DECAY_K = 0.5
H2H_DECAY_K  = 0.05

# Per-league season configuration for the motivation factor.
# season_games: total games per team in the regular season.
# safety_pts:   points typically needed to avoid relegation (empirical).
# relegated:    number of teams relegated (affects relegation zone width).
LEAGUE_SEASON_CONFIG = {
    "PL":  {"season_games": 38, "safety_pts": 36, "relegated": 3},
    "PD":  {"season_games": 38, "safety_pts": 38, "relegated": 3},
    "BL1": {"season_games": 34, "safety_pts": 33, "relegated": 3},  # 18 teams, 34 games
    "SA":  {"season_games": 38, "safety_pts": 36, "relegated": 3},
    "FL1": {"season_games": 34, "safety_pts": 33, "relegated": 3},  # 18 teams, 34 games
    "CL":  {"season_games": 6,  "safety_pts": 9,  "relegated": 0},  # group stage
    "PPL": {"season_games": 34, "safety_pts": 33, "relegated": 3},  # 18 teams, 34 games
    "DED": {"season_games": 34, "safety_pts": 30, "relegated": 3},  # Eredivisie, 18 teams
    "ELC": {"season_games": 46, "safety_pts": 52, "relegated": 3},  # Championship, 24 teams
    "BSA": {"season_games": 38, "safety_pts": 38, "relegated": 4},  # 20 teams, 4 relegated
}

# Fatigue threshold model (4.4).
# Research consensus: <3 days between matches is the meaningful recovery threshold.
# Step-function: ≤2 days = full penalty (back-to-back), 3 days = moderate,
# ≥4 days = no fatigue at all.
# Full penalty: 10% attack reduction. Moderate (3 days): 5%.
# Opponent gets half the penalty as a counter-boost.
# Season confidence ramp (§4.3).
# Adjustments (form, position, trend) are scaled from 0 → 1 over the first
# SEASON_CONF_RAMP games. Linear ramp; set from observation that league standings
# start reflecting true team quality around game 12-15 (not 10 as originally assumed).
# Increase this value → more conservative picks early in season.
SEASON_CONF_RAMP = 15

# Fatigue threshold model (4.4).
# Research consensus: <3 days rest is the meaningful threshold.
# Step-function: ≤2 days = full penalty (back-to-back schedule, squad rotation typical)
#                3 days  = moderate penalty (midweek turnaround)
#                ≥4 days = no penalty (adequate recovery)
FATIGUE_SEVERE_DAYS    = 2    # ≤ this → full penalty
FATIGUE_MODERATE_DAYS  = 3    # == this → moderate penalty
FATIGUE_PENALTY_FULL   = 0.10 # attack reduction for ≤2-day rest
FATIGUE_PENALTY_MOD    = 0.05 # attack reduction for exactly 3-day rest

# Form momentum signal (POTENTIAL-02).
# Momentum = recent_3_form - recent_6_form (delta of short vs long form score).
# A team winning their last 3 more than their rolling 6 is on an upswing.
# MOMENTUM_THRESHOLD: delta must exceed this before a boost/penalty fires (avoids noise).
# MOMENTUM_BOOST: attack multiplier on upswing (4% increase — conservative).
# MOMENTUM_PENALTY: attack multiplier on downswing (mirror of boost).
MOMENTUM_THRESHOLD = 0.15
MOMENTUM_BOOST     = 1.04
MOMENTUM_PENALTY   = 0.96

# Streak detection (POTENTIAL-06).
# Clean sheet streaks boost defence; scoring drought streaks reduce attack.
# MIN_STREAK: minimum consecutive matches before the signal fires.
# Boost/penalty per additional match beyond the minimum (additive, capped at MAX_STREAK).
STREAK_MIN_LENGTH      = 3     # minimum consecutive matches to activate
STREAK_MAX_LENGTH      = 6     # cap — beyond 6 the signal is already saturated
STREAK_CS_DEFENCE_BOOST  = 0.05  # per-match defence reduction (lower expected_total)
STREAK_DROUGHT_ATK_PENALTY = 0.04  # per-match attack reduction


def validate_config():
    """Assert critical constants are within sane ranges. Raises ValueError on bad values."""
    errors = []
    for league, rho in DC_RHO.items():
        if not (-0.5 < rho < 0):
            errors.append(f"DC_RHO[{league}]={rho} must be in (-0.5, 0)")
    for league, adv in HOME_ADV.items():
        if not (0.8 < adv < 2.5):
            errors.append(f"HOME_ADV[{league}]={adv} must be in (0.8, 2.5)")
    if not (0.5 < KNOCKOUT_GOALS_FACTOR < 1.2):
        errors.append(f"KNOCKOUT_GOALS_FACTOR={KNOCKOUT_GOALS_FACTOR} must be in (0.5, 1.2)")
    if not (0 < FATIGUE_PENALTY_FULL < 0.5):
        errors.append(f"FATIGUE_PENALTY_FULL={FATIGUE_PENALTY_FULL} must be in (0, 0.5)")
    if not (0 < FATIGUE_PENALTY_MOD < FATIGUE_PENALTY_FULL):
        errors.append(f"FATIGUE_PENALTY_MOD must be < FATIGUE_PENALTY_FULL")
    if not (0 < SEASON_CONF_RAMP <= 38):
        errors.append(f"SEASON_CONF_RAMP={SEASON_CONF_RAMP} must be in (0, 38]")
    if not (1.0 <= MOMENTUM_BOOST <= 1.2):
        errors.append(f"MOMENTUM_BOOST={MOMENTUM_BOOST} must be in [1.0, 1.2]")
    if not (0.8 <= MOMENTUM_PENALTY <= 1.0):
        errors.append(f"MOMENTUM_PENALTY={MOMENTUM_PENALTY} must be in [0.8, 1.0]")
    if not (0 < STREAK_CS_DEFENCE_BOOST < 0.3):
        errors.append(f"STREAK_CS_DEFENCE_BOOST={STREAK_CS_DEFENCE_BOOST} must be in (0, 0.3)")
    if not (0 < STREAK_DROUGHT_ATK_PENALTY < 0.3):
        errors.append(f"STREAK_DROUGHT_ATK_PENALTY={STREAK_DROUGHT_ATK_PENALTY} must be in (0, 0.3)")
    if errors:
        raise ValueError("config.py validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


validate_config()

# Bump this whenever a breaking football-data.org API schema change is confirmed.
# Including it in every cache key forces all cached files to be treated as stale.
CACHE_VERSION = 2

# Per-league SoT → xG conversion factor.
# Global fallback (XG_CONV) stays in analyzer.py for backtest patchability.
# Values from multi-season SoT/goals regression (research-based; run
# scripts/fit_xg_conv.py to update from matches.db once 200+ matches accumulate).
XG_CONV_BY_LEAGUE = {
    "PL":  0.31,   # physical play, fewer clear-cut chances per SoT
    "PD":  0.34,   # La Liga: technical play, better per-shot conversion
    "BL1": 0.29,   # Bundesliga: high SoT volume, lower per-shot conversion
    "SA":  0.34,   # Serie A: clinical finishing
    "FL1": 0.32,   # Ligue 1: slightly below average
    "CL":  0.33,   # Champions League: near average
    "PPL": 0.31,   # Primeira Liga
    "DED": 0.30,   # Eredivisie: open but high SoT volume
    "ELC": 0.30,   # Championship: lower quality finishers
    "BSA": 0.31,   # Brasileirão
}

# Per-league expected total gates for Over/Under markets.
# evaluate_over_under() uses these instead of the global fallback when a league is known.
# Gate: model's expected_total must be >= this for an Over pick.
# Cap:  model's expected_total must be <= this for an Under pick.
# Source: multi-season goals/game averages; update with scripts/fit_xg_conv.py output.
EXPECTED_TOTAL_GATES_BY_LEAGUE = {
    "Over 2.5": {
        "PL":  2.65,   # PL avg ~2.75 → gate 0.10 below mean
        "PD":  2.65,
        "BL1": 3.00,   # BL1 avg ~3.10
        "SA":  2.55,   # SA avg ~2.65 — defensive league
        "FL1": 2.65,
        "CL":  2.80,   # CL avg ~2.90
        "PPL": 2.65,
        "DED": 2.80,   # DED avg ~2.90
        "ELC": 2.60,   # ELC avg ~2.70
        "BSA": 2.70,
    },
    "Over 3.5": {
        "PL":  3.40,
        "PD":  3.40,
        "BL1": 3.60,
        "SA":  3.30,
        "FL1": 3.40,
        "CL":  3.50,
        "PPL": 3.40,
        "DED": 3.50,
        "ELC": 3.35,
        "BSA": 3.45,
    },
}
EXPECTED_TOTAL_CAPS_BY_LEAGUE = {
    "Under 2.5": {
        "PL":  2.85,
        "PD":  2.85,
        "BL1": 3.20,
        "SA":  2.75,
        "FL1": 2.85,
        "CL":  3.00,
        "PPL": 2.85,
        "DED": 3.00,
        "ELC": 2.80,
        "BSA": 2.90,
    },
    "Under 3.5": {
        "PL":  3.85,
        "PD":  3.85,
        "BL1": 4.20,
        "SA":  3.75,
        "FL1": 3.85,
        "CL":  4.00,
        "PPL": 3.85,
        "DED": 4.00,
        "ELC": 3.80,
        "BSA": 3.90,
    },
}
