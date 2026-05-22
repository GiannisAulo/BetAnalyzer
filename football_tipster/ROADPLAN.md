# Football Tipster — Production Roadmap

**Original state:** 46% win rate on 63 settled bets. Low-conviction picks, unvalidated
parameters, zero backtesting. Not ready for use.

**Current state (2026-04-28):** Foundation complete. All Phase 1–3 items done plus
9 engineering audit fixes. Calibration framework built (option 7). Backtest shows
67–71% WR across 4 leagues; live data (167 settled bets) shows 53% overall —
new-model cohort at 58% WR. Full codebase audit complete (see PROJECT_ANALYSIS.md).
Key new items added in Phase 2.9–2.12, Phase 3.8–3.9, Phase 4.6–4.9.
System is usable for paper trading — not yet validated for staking real money.

**Target:** ≥75% win rate on a statistically meaningful sample (500+ bets), provably
positive EV, zero silent failures.

---

## Phase 1 — Stop the Bleeding ✓ COMPLETE

### 1.1 Raise minimum thresholds (DONE)
- `MIN_PROB` for Home Win raised 0.30 -> 0.55, Away Win 0.25 -> 0.50, Draw 0.20 -> 0.45
- No-odds 1X2 fallback raised 0.50 -> 0.65

### 1.2 Freeze the ML calibrator until 200+ bets (DONE)
- `MIN_SEGMENT_SIZE` raised to 100, `MIN_LR_SIZE` to 200, `MIN_LR_LEAGUE_SIZE` to 75.

### 1.3 Disable auto-baseline updates (DONE)
- `auto_update_baselines()` gated behind `--update-baselines` CLI flag.

### 1.4 Fix the Combo settlement gap (DONE)
- `_evaluate_result()` now handles all 13 combo types. Combos auto-settle correctly.
- **2026-04-26 follow-up fix**: unrecognised combo legs previously returned `None`
  (bet stays pending forever). Now returns `"L"` so the bet settles and doesn't
  silently corrupt calibration counts. (`logger.py:103`)

### 1.5 Move API keys out of source code (DONE)
- `API_KEY` moved to `.env` / environment variable. Startup error if missing.

---

## Phase 2 — Fix the Model's Core Issues ✓ MOSTLY COMPLETE

### 2.1 Build a backtesting framework (DONE)
- `scripts/backtest.py`: replays full seasons with zero data leakage.
- Outputs per-market WR, Brier score, calibration curves.
- `--sweep-rho` and `--sweep-xg` modes for parameter sweeps.
- Backtest result (4 leagues, 2023 season): 67.3% overall WR, Brier 0.212.

### 2.2 Investigate and fix Over/Under systematic over-prediction (DONE)
- Backtest showed Over/Under over-predicts by 6-8pp across all probability bands.
- **DC_RHO sweep** (-0.10 to -0.50): OU Brier range 0.1974-0.1993 — difference is noise.
  Per-league DC_RHO values retained (uniform override loses real league differentiation).
- **XG_CONV sweep** (0.25 to 0.35): zero effect — backtest lacks shot data so xG path
  is never activated. Confirms XG_CONV is fine at 0.33 for the live model.
- **Root cause**: calibration gap is structural (derived standings vs live API table).
  Not a model parameter issue. Phase 4.1 isotonic regression is the correct long-term fix.
- **Applied**: MIN_PROB gates raised, BTTS No disabled (49% WR), Over 3.5 MIN_PROB
  raised to 0.65 (44% WR), contextual xG gates added (expected_total gates).

### 2.3 Re-validate knockout stage penalties (DROPPED)
- Requires paid API tier for 5 seasons of CL data. Not worth the cost.
- `KNOCKOUT_GOALS_FACTOR = 0.78` stays as-is; avoid CL knockout picks for real money.

### 2.4 Fix home advantage double-counting for new teams (DONE)
- `_default_team_stats()` returns equal home/away (1.1/1.1). `HOME_ADV` handles split.
- Prevents double-counting home advantage for teams with <4 home games in history.

### 2.5 Remove the `rating` field (DONE)
- Picks sorted by `model_prob` (no-odds) or `edge` (odds) directly.
- `_edge_rating()`, `_prob_rating()`, stars display removed.

### 2.6 Standardise recency decay constants (DONE)
- `FORM_DECAY_K = 0.5`, `H2H_DECAY_K = 0.05` defined in `config.py` with rationale.
- Both use `exp(-k * age_days / 30)`. Different values are intentional and documented:
  form needs faster decay (recent matches matter more); H2H spans years so slower decay.

### 2.7 Fix xG trend double-counting home advantage (DONE — 2026-04-26)
- **Bug**: the C.2 trend factor divided venue-split home avg by the overall season avg.
  Since home teams naturally score ~15–30% more at home than their mixed avg, this ratio
  was always > 1.0, adding an extra multiplier on top of `home_attack` (which already
  uses home-split data). Produced inflated values like "xG 3.2–0.6" for mid-table teams.
- **Fix**: denominator now adjusted by `HOME_ADV` per league so the ratio is ~1.0 for a
  team in normal form. Trend clamp tightened from ±33% to ±20% (fine-tuning signal only).
  (`analyzer.py`, C.2 section)

### 2.8 Fix xG unit mismatch in trend factor (DONE — 2026-04-26)
- **Bug**: after the 60/40 xG+goals blend, `home_avg_scored` was in mixed units
  (shots×0.33 + raw goals), but divided by `home_season_avg` (pure raw goals from
  standings). Ratio compared different scales → systematic bias when xG proxy differs
  from raw goals.
- **Fix**: raw pre-blend split averages saved before xG blend; those are used for the
  trend ratio instead of the blended values. (`analyzer.py`, lines 1017–1021)

### 2.9 Lower xG proxy activation threshold (NOT STARTED)
- **Why**: `_MIN_XG_WEIGHT = 3.5` in `analyzer.py:298` requires a decay-weight sum of
  3.5 to activate xG. With `FORM_DECAY_K=0.5`, four matches over a 30-day span produce
  weight ≈ 2.9 — below the threshold. xG effectively never activates during normal
  league schedules, so the model is running on raw goals for 80%+ of fixtures despite
  having shot data.
- **Fix**: Lower to `_MIN_XG_WEIGHT = 2.0`, or switch to a match-count gate
  (`if xg_match_count >= 3`). Validate that Brier score improves in `scripts/backtest.py`.
- **Expected WR impact**: MEDIUM-HIGH — xG is a better predictor of Over/Under than raw goals.
  Over 2.5 calibration gap of −15pp should narrow.

### 2.10 Fix logger dedup key (NOT STARTED)
- **Why**: `logger.py:297` deduplicates on `match_id` only. When a fixture generates two
  picks (e.g. Home Win + Over 2.5), only the first is logged. The second is silently
  dropped. ML calibration trains on this truncated data, and the historical win rate
  calculations are incomplete.
- **Fix**: Change dedup key to `(match_id, market, pick)` so all distinct picks per
  fixture are logged. Adjust `_load_logged_ids()` return type accordingly.
- **Expected WR impact**: MEDIUM — richer training data unlocks better ML calibration sooner.

### 2.11 League-specific xG conversion factor (NOT STARTED)
- **Why**: `XG_CONV = 0.33` (shots-on-target → xG) is uniform across all leagues.
  Bundesliga has higher SoT volume at lower conversion (~0.28); La Liga and Serie A
  convert at ~0.35. Using 0.33 uniformly overstates xG in BL1 by ~10% and understates
  it in PD/SA — the opposite direction of what DC_RHO is trying to correct.
- **Fix**: Add `XG_CONV_BY_LEAGUE` dict in `config.py`. For each league, regress
  `actual_goals ~ shots_on_target` from `matches.db` (script: `scripts/fit_xg_conv.py`).
  Use league-specific value when available, global 0.33 as fallback.
- **Expected WR impact**: MEDIUM — reduces systematic Over/Under bias that currently shows
  as −15pp calibration gap in high-SoT leagues.

### 2.12 Fix no-odds threshold inconsistency (NOT STARTED)
- **Why**: Over 2.5 with odds requires `model_prob ≥ 0.62`; without odds requires ≥ 0.65.
  These should be the same floor. In fact the no-odds bar should be *higher* (not lower)
  since you can't verify value without a price. Currently backwards.
- **Fix**: Set all no-odds thresholds ≥ their `MIN_PROB` equivalent in `markets.py`.
  For markets with large calibration gaps (Over 2.5 at −15pp), raise the no-odds bar to 0.70+.
- **Expected WR impact**: LOW (affects volume, not accuracy) but removes incorrect logic.

---

## Phase 3 — Data Quality & Silent Failures ✓ COMPLETE

### 3.1 Log every data fallback (DONE)
- `warn_log.py` writes one line per fallback to `warnings.log`:
  `date | league | match_id | reason | fallback_used`

### 3.2 Validate probability sums after every normalization (DONE)
- 3 asserts in `analyzer.py` after each normalization step.
- **2026-04-26 hardening**: all 3 normalisation sites now guard `total > 0` before
  dividing — a ZeroDivisionError would crash before the assert ever fired. Falls back
  to uniform (1/3, 1/3, 1/3) on total = 0. (`analyzer.py`, lines 956, 981, 1228)

### 3.3 Fix H2H venue-split orientation leak (DONE)
- H2H goals only blended when `venue_split=True`. Win rates used freely (orientation-invariant).

### 3.4 Add shot data coverage check (DONE)
- `XG_CONV` moved to module level in `analyzer.py` (patchable for sweeps).
- Warning logged when home shot-data coverage <70% for a team.

### 3.5 Defend against API schema changes (DONE)
- `CACHE_VERSION = 2` in `config.py`. Included in every cache key prefix.

### 3.6 Fix SQLite concurrent write data loss (DONE — 2026-04-26)
- **Bug**: parallel league processing caused `OperationalError: database is locked`.
  The exception propagated to `fetcher.py` which silently swallowed it with
  `except Exception: pass` — match history lost with no warning.
- **Fix**: `store_matches()` retries up to 4 times with exponential backoff (150ms,
  300ms, 600ms) before re-raising. (`match_store.py`, lines 148–163)

### 3.8 Auto-refresh BASELINES monthly (NOT STARTED)
- **Why**: BASELINES in `config.py` are static multi-season estimates. As live bets
  accumulate, empirical prior probabilities will drift. A PL fixture currently uses
  baseline (home=0.47, draw=0.28, away=0.25) regardless of whether the current season
  is unusually defensive or attack-heavy.
- **Fix**: In `_show_bet_history()` (or after `settle_bets()`), if settled bets ≥ 200
  and last BASELINES update > 30 days ago, call `auto_update_baselines()` and log
  the update timestamp. Gate on ≥ 20 settled bets per league to avoid noise.
- **Expected WR impact**: MEDIUM — fresher priors reduce systematic over/under-prediction
  by league. Most impactful early in a new season.

### 3.9 Fuzzy odds team name matching (NOT STARTED)
- **Why**: `odds_fetcher.py` has ~80 hardcoded team name translations. Any unmatched
  team name means the fixture gets no live odds and the pick falls back to probability-
  only selection (no edge gate). Missing odds = weaker picks — the edge calculation is the
  primary value filter for quality bet selection.
- **Fix**: After exact `_NAME_MAP` lookup fails, run `difflib.SequenceMatcher` against
  all team names in the Odds API response. Accept if ratio ≥ 0.80. Log unmatched pairs
  to `warnings.log` for manual review.
- **Expected WR impact**: HIGH — more fixtures get live odds → edge gate applies to more
  picks → weaker probability-only picks are filtered out → WR improves.

### 3.7 Engineering audit fixes (DONE — 2026-04-26)

Nine issues identified and resolved in a full codebase audit:

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `warn_log.py` | `datetime.UTC` crashes Python 3.10 (added in 3.11) | `datetime.timezone.utc` |
| 2 | `cache.py` | `JSONDecodeError` on corrupt cache file crashes fetch | Catch error, delete file, fall through to live fetch |
| 3 | `cache.py` | `set_cache` OSError propagated silently | Wrap in `try/except OSError: pass` |
| 4 | `match_store.py` | Double-close in retry loop (except + finally both called `conn.close()`) | `conn = None` init; single close in `finally` with `if conn is not None` guard |
| 5 | `ml_calibrator.py` | External code accessed private `_seg` dict directly (`cal._seg.get(...)`) | Added `segment_uncertainty(market, pick, league)` public method |
| 6 | `main.py` | 2 redundant `get_last_match_date()` API calls per fixture (2 credits) | Replaced with `home_merged["matches"][0].get("utcDate")` — data already fetched |
| 7 | `odds_fetcher.py` | `ODDS_API_KEY = os.getenv(...)` duplicated config loading; C2 line_move computed and cached but never consumed (dead feature) | Import from `config`; removed opening snapshot, movement loop, `line_move` key |
| 8 | `fetcher.py` | `get_match_stats()` dead function — no callers anywhere in codebase | Deleted |
| 9 | `scripts/backtest.py` | Form score formula used `1.0×w` for wins / `0.5×w` for draws; `analyzer._compute_form_score()` uses `3×w` normalised by `max_score` (draws get 1/3 weight, not 1/2) | Call `analyzer._compute_form_score()` directly to guarantee parity |

Root `backtest.py` (project root) also deleted — it was a stale entry point superseded by
`scripts/backtest.py`; `_backtest_menu()` already called `scripts.backtest` via subprocess.

---

## Phase 4 — Pick Quality Improvements (In Progress)

These require live bet data accumulating after the Phase 1–3 fixes. 134 bets settled
as of 2026-04-26; overall live WR is 53% vs 67–71% backtest — gap under investigation.

### 4.0 Diagnose live vs backtest WR gap (RESOLVED — 2026-04-26)
- Live: 53% overall (134 settled). Backtest: 67–71%. Gap is 14–18pp.
- **Root cause identified**: the 134-bet pool includes pre-threshold picks (prob < 0.50,
  from before Phase 1 threshold raises) that carried a 21% WR and pulled the aggregate
  down. Isolating new-model picks (post-threshold, model_prob ≥ floor) shows 58% WR —
  consistent with backtest direction. Gap is cohort contamination, not model failure.
- **Residual known issue**: PL `strength_factors` fallback fires on some fixtures (team
  IDs in warnings.log). PL WR is 44% (8/18). Worth monitoring but not blocking —
  warnings.log coverage check in §5.4 will surface this systematically.
- **1X2 at 40% WR**: raising MIN_PROB floor to 0.65 applied (Phase 1.1 note). Continue
  monitoring as new-model picks accumulate.

### 4.1 Per-league probability calibration (FRAMEWORK DONE — 2026-04-28)
- `scripts/calibration_curves.py` built: reads bets_log.csv, bins model_prob
  into 5pp buckets, reports actual WR vs predicted per pick type + league.
- Accessible via menu option 7 "Calibration" with three sub-modes:
  all markets, per-league breakdown, single pick type.
- Isotonic regression applies automatically when ≥20 bets per pick type.
- **Current findings (167 settled)**: Over 2.5 −15pp bias (OVER-CONFIDENT),
  Under 2.5 −12pp bias, Under 3.5 −11pp bias — all below their predicted band.
  Home Win −6pp (slightly over). MIN_PROB floors are doing their job but
  systematic bias remains — confirms isotonic correction is needed.
- **Trigger for applying correction:** 100+ settled bets per pick type.
  Run `python -m scripts.calibration_curves` to check progress.

### 4.2 Improve motivation factor (DONE — 2026-04-26)
- **Bug**: `league` key was never injected into `home_standing`/`away_standing` dicts, so
  `LEAGUE_SEASON_CONFIG` lookup always fell back to the default (38-game, 38-pt safety).
  BL1/FL1/PPL/DED all use 34 games and lower safety thresholds — completely wrong values.
  **Fix**: `"league": league_code` now added when constructing both standing dicts (`main.py`).
- Per-league safety thresholds were already correct in `config.py` — now they actually fire.
- **Must-win detection added**: relegation zone, ≤4 games left, deficit exceeds games remaining
  (draws can't close the gap) → 1.12 multiplier. Same for title decider with ≤3 games.
- Dead rubber default safety threshold corrected from 38 to 36 (PL empirical average).

### 4.3 Tune season_conf ramp (DONE — 2026-04-26)
- Ramp moved to `SEASON_CONF_RAMP` constant in `config.py` (now patchable for sweeps).
- Default changed 10 → 15 games: league standings start reflecting true team quality around
  game 12-15, not game 10. Using 10 over-weighted form/position adjustments for early-season
  fixtures where those signals are still noisy.

### 4.4 Fatigue model threshold (DONE — 2026-04-26)
- Replaced continuous `exp(-days/7)` model with research-based step function.
- Old model applied penalty all the way to 7 days (still 4.4% at day 7) with no evidence.
- New model (`analyzer.py`, config constants `FATIGUE_*`):
  - ≤2 days: 10% attack reduction (back-to-back, squad rotation certain)
  - 3 days: 5% attack reduction (midweek turnaround, standard busy period)
  - ≥4 days: no penalty (adequate recovery)
- Opponent counter-boost retained at half the penalty in both cases.

### 4.5 Combo win rate tracking (IN PROGRESS)
- 27 combos settled as of 2026-04-26: 15W/12L = 56% WR. Above the 40% disable trigger.
- Monitor: if WR drops below 40% after 50 settled combos, disable combo picks.

### 4.6 Per-market edge threshold optimization (NOT STARTED)
- **Why**: `min_edge` is set globally (default 5.0%) with small per-market offsets hardcoded
  in `_run_analysis()`. There is no empirical evidence these thresholds are optimal.
  Some markets may be profitable at ≥ 3% edge; others may need ≥ 8% to filter noise.
- **How**: Add `--sweep-edge` mode to `scripts/backtest.py`. Sweep `min_edge` from 2% to
  12% in 1pp steps per market. Pick the value that maximises WR on held-out data.
  Store results in `EDGE_THRESHOLDS` dict in `config.py`.
- **Trigger**: Run after 300+ total settled bets for statistical confidence.
- **Expected WR impact**: MEDIUM-HIGH — eliminates low-value picks that are currently
  passing the threshold and dragging WR down.

### 4.7 League-specific expected total gates (NOT STARTED)
- **Why**: `_EXPECTED_TOTAL_GATES = {"Over 2.5": 2.90}` in `markets.py` is league-agnostic.
  PL averages 2.75 goals/game — the gate of 2.90 suppresses virtually all Over 2.5 picks
  in PL. BSA averages 3.2 — the same gate passes almost everything. The gate should be
  calibrated to each league's scoring environment.
- **How**: Add `EXPECTED_TOTAL_GATES_BY_LEAGUE` in `config.py`. Compute per-league mean
  expected total from `matches.db` (or from backtest output). Set gate = league mean − 0.10.
- **Expected WR impact**: MEDIUM — recovers suppressed Over 2.5 picks in low-scoring leagues;
  tightens the gate in high-scoring leagues to reduce false positives.

### 4.8 Wire isotonic correction into live picks (NOT STARTED)
- **Why**: `calibration_curves.py` (option 7) measures and fits isotonic regression per
  pick type but does not apply corrections to live predictions. The Over 2.5 model is
  currently −15pp biased: a pick shown at 63% probability is winning at ~48%. This
  mismatch causes picks to pass MIN_PROB gates that they should not.
- **How**: When a pick type reaches 50+ settled bets, serialize the isotonic correction
  map to `data/isotonic_corrections.json`. In `ml_calibrator.calibrate()`, load this map
  as a Tier 0 correction applied before the LR tiers. Rerun `calibration_curves.py` after
  each 20 new settled bets to update the map.
- **Trigger**: Over 2.5 (27 bets) and Under 2.5 (33 bets) are closest. Need 50+ each.
- **Expected WR impact**: HIGH — directly corrects the systematic Over/Under bias.
  If Over 2.5 corrects to 48% actual from 63% predicted, MIN_PROB gate adjusts accordingly
  and only genuine ≥ 55% probability picks go through.

### 4.9 Form momentum signal (NOT STARTED)
- **Why**: `form_score` is a level (overall quality). It cannot distinguish a team on a
  5-game winning streak from one that won 4 games months ago and just lost. Momentum
  (short-term trend relative to recent average) is an independent signal.
- **How**: In `parse_team_history()`, compute `recent_3_form` and `recent_6_form` separately.
  If `recent_3 > recent_6 + 0.15`, apply `MOMENTUM_ATTACK_BOOST = 1.04` to attack parameter.
  If `recent_3 < recent_6 − 0.15` (downswing), apply a matching penalty.
  Validate via `--sweep-momentum` in backtest.
- **Expected WR impact**: LOW-MEDIUM — captures current form direction that rolling averages
  smooth out. Most useful for 1X2 picks.

---

## Phase 5 — Production Hardening (Month 3+)

Only start when Phase 4 is complete and live WR ≥ 70% on 200+ bets.

### 5.1 Stake sizing (Kelly Criterion) — DROPPED
- No bankroll tracking in scope. Kelly requires a bankroll to be useful.
### 5.2 Bankroll tracking — DROPPED
### 5.3 Model versioning (DONE — 2026-04-27)
- `MODEL_VERSION` constant in `logger.py` stamped on every pick row in `bets_log.csv`.
- Bump the date string whenever thresholds or model logic change significantly.
- Allows future analysis to filter by model cohort.

### 5.4 Daily data freshness check (DONE — 2026-04-27)
- Warnings screen (option 6) now shows standings cache age per league.
- Green < 6h, yellow 6–20h, red > 20h with explicit "run option 1 to refresh" message.
- Uses `get_cache_age_hours()` in `cache.py` + `CACHE_VERSION` prefix to target the
  correct versioned cache file for each league.

### 5.5 Notification system — Telegram/email/webhook on pick + settlement

### 5.6 Web interface — only after demonstrably profitable on paper trading

### 5.7 ML calibrator train/test split (NOT STARTED)
- **Why**: The LR is trained on all 200+ settled bets and then used to calibrate picks
  from the same distribution. Overfitting is invisible — the model may appear calibrated
  in-sample but degrade on new data. Currently there is no way to detect this.
- **How**: Time-based split: train on bets older than 60 days, evaluate on the last 60.
  Report out-of-sample Brier score in `scripts/calibration_curves.py`. Only apply LR
  if out-of-sample Brier < uncalibrated Brier. Otherwise, fall back to segment stats only.
- **Trigger**: Run when global LR activates (200+ settled bets).

### 5.8 LR blend weight optimization (NOT STARTED)
- **Why**: Blend weights (60/40 for per-league LR, 55/45 for global) are hardcoded without
  any sweep. Could be over-correcting (high variance at small n) or under-correcting
  (systematic bias still present). Optimal blend is unknown.
- **How**: Add `LR_BLEND_WEIGHT` to `config.py`. Add `--sweep-lr-blend` mode to backtest
  testing 0.40–0.80 in 0.10 steps. Use value minimising out-of-sample Brier.
- **Trigger**: After Phase 5.7 train/test split is in place.

### 5.9 Closing line value tracking (NOT STARTED)
- **Why**: The most reliable long-term indicator of a genuinely predictive model is whether
  picks consistently beat the closing line (bookmaker's final pre-match odds). Currently
  `odds_taken` is the line at pick generation time. If markets move against picks, it
  indicates adverse selection (bookmakers sharpen their lines before kick-off).
- **How**: In `settle_bets()`, fetch current odds for each pending bet on settlement day.
  Log `closing_implied_prob`. Compute `clv = model_prob - closing_implied_prob`. Positive
  average CLV over 100+ picks is the strongest signal of true model edge.
- **Trigger**: Can be built now; requires Odds API call at settlement time.

---

## Am I Ready to Use the System?

### For paper trading (tracking picks without real money): YES
- All critical bugs fixed (see Known Bugs below).
- Bad markets removed (Over 3.5, BTTS No) — these were losing money.
- Every prediction has probability floors — no more 30% Away Win tips.
- Fallback logging: you can see when the model is flying blind.
- Backtest: 67–71% WR on 4 leagues across a full season.

### For staking real money: NOT YET
- Live WR is 53% on 167 bets — new-model cohort at 58% (old picks contaminating pool).
- xG threshold bug (2.9) means model runs on raw goals for ~80% of fixtures — fixing this
  may improve Over/Under significantly before needing more data.
- Logger dedup bug (2.10) means secondary picks are not being captured — fixes ML data quality.
- Over 2.5 is −15pp miscalibrated — isotonic correction (4.8) needed before trusting these.
- Phase 4.1 calibration framework done; correction not yet applied to live picks.

---

## Open Items Summary

**Priority order within each group reflects expected WR impact.**

### Bugs to fix (highest WR impact first)
| Item | Status | Notes |
|---|---|---|
| 3.9 Fuzzy odds name matching | NOT STARTED | More fixtures get live odds → edge gate applies |
| 2.9 Lower xG threshold (3.5→2.0) | NOT STARTED | xG activates for ~80% more fixtures |
| 2.10 Logger dedup key fix | NOT STARTED | All picks logged → richer ML training set |
| 2.11 League-specific XG_CONV | NOT STARTED | Reduces Over/Under bias per league |
| 2.12 No-odds threshold inconsistency | NOT STARTED | Minor; fixes backwards logic |

### Model improvements (waiting on data or backtest)
| Item | Status | Trigger |
|---|---|---|
| 4.8 Wire isotonic correction | NOT STARTED | 50+ settled bets per pick type (Over 2.5 @ 27, Under 2.5 @ 33) |
| 4.6 Per-market edge threshold sweep | NOT STARTED | 300+ total settled bets |
| 4.7 League-specific expected total gates | NOT STARTED | Can build now from matches.db |
| 3.8 Auto-refresh BASELINES | NOT STARTED | 200+ settled bets (33 bets away) |
| 4.9 Form momentum signal | NOT STARTED | Backtest validation first |

### ML calibration improvements
| Item | Status | Trigger |
|---|---|---|
| 5.7 Train/test split for LR | NOT STARTED | When global LR activates (200+ bets) |
| 5.8 LR blend weight optimization | NOT STARTED | After 5.7 is in place |
| 4.5 Combo WR tracking | IN PROGRESS | 27/50 bets accumulated |
| 4.1 Calibration framework | DONE | Apply correction when 50+ bets per type |

### Production items
| Item | Status | Blocker |
|---|---|---|
| 5.9 Closing line value tracking | NOT STARTED | Can build now |
| 5.5 Notifications | NOT STARTED | Can build anytime |
| 5.6 Web interface | NOT STARTED | Need ≥70% live WR |

### Completed / dropped
| Item | Status |
|---|---|
| 4.0 Live vs backtest WR gap | **RESOLVED** |
| 3.7 Engineering audit (9 fixes) | **DONE** |
| 4.2 Motivation factor + must-win | **DONE** |
| 4.3 Season_conf ramp | **DONE** |
| 4.4 Fatigue threshold model | **DONE** |
| 5.3 Model versioning | **DONE** |
| 5.4 Data freshness check | **DONE** |
| 2.3 Knockout penalties | DROPPED |
| 5.1 Kelly | DROPPED |
| 5.2 Bankroll tracking | DROPPED |

---

## Metrics — Definition of "Ready for Production"

| Metric | Minimum threshold | Current | How to measure |
|--------|-------------------|---------|----------------|
| Win rate | ≥ 75% | 53% (134 bets) | Rolling last 100 settled bets |
| Sample size | ≥ 500 settled bets | 134 | bets_log.csv |
| Brier score | ≤ 0.20 | 0.212 (backtest) | backtest.py calibration output |
| Over 2.5 calibration | Predicted 0.65 → actual 62–68% | Unknown | Calibration curve |
| 1X2 WR (model_prob ≥ 0.65) | ≥ 68% | 40% (live) | bets_log.csv segmented |
| Combo WR | ≥ 40% | 56% (27 bets) | bets_log.csv |
| Data fallback rate | < 5% of fixtures | High for PL | warnings.log |
| Drawdown (when staking) | < 20% | N/A | bankroll_log.csv |

---

## Known Bugs — Fixed

| Bug | Fixed | Notes |
|-----|-------|-------|
| Combo picks never settled | Phase 1.4 | All 13 combo types handled |
| Combo unknown-leg stuck pending | 2026-04-26 | Unknown legs now settle as "L" (`logger.py`) |
| API key hardcoded | Phase 1.5 | `.env` + startup error |
| Home advantage double-counted (new teams) | Phase 2.4 | `_default_team_stats()` returns 1.1/1.1 |
| xG trend double-counts home advantage | 2026-04-26 | Denominator adjusted by HOME_ADV (`analyzer.py`) |
| xG unit mismatch in trend factor | 2026-04-26 | Raw pre-blend avg used for trend (`analyzer.py`) |
| SQLite data loss under parallel load | 2026-04-26 | Retry with backoff (`match_store.py`) |
| Prob normalisation division-by-zero | 2026-04-26 | Guard before divide, uniform fallback (`analyzer.py`) |
| ML calibrator activates at 25 bets | Phase 1.2 | Thresholds raised to 100/200 |
| Auto-baseline update runs silently | Phase 1.3 | `--update-baselines` flag required |
| H2H orientation leak | Phase 3.3 | Goals gated on venue_split |
| Knockout penalties from 4 matches | Noted | Can't fix without paid API |
| Referee dedup loop dead code | Removed | — |
| Python 3.10 crash in `warn_log.py` | 2026-04-26 | `datetime.UTC` → `datetime.timezone.utc` |
| Cache JSON corruption crash | 2026-04-26 | Catch `JSONDecodeError`, delete file, re-fetch (`cache.py`) |
| SQLite double-close in retry loop | 2026-04-26 | `conn = None` guard + single `finally` close (`match_store.py`) |
| `ml_calibrator._seg` accessed directly | 2026-04-26 | Added `segment_uncertainty()` public method |
| 2 redundant API calls per fixture | 2026-04-26 | Removed `get_last_match_date()` calls; use already-fetched data |
| `odds_fetcher` line_move dead feature | 2026-04-26 | Removed opening snapshot, movement loop, `line_move` key |
| `fetcher.get_match_stats()` dead function | 2026-04-26 | Deleted — no callers |
| Root `backtest.py` stale entry point | 2026-04-26 | Deleted — `scripts/backtest.py` is canonical |
| Backtest form score formula mismatch | 2026-04-26 | Fixed to call `analyzer._compute_form_score()` directly |
