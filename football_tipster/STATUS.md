# Football Tipster — Status & Open Work

> Single source of truth. Supersedes the old `ROADPLAN.md`, `PROJECT_ANALYSIS.md`,
> and root-level `ROADMAP.md` (all deleted 2026-05-23).

---

## Current State

- **Live WR:** 53% overall on 167 settled bets · 58% on new-model cohort (post-threshold)
- **Backtest WR:** 67–71% across 4 leagues (full 2023 season)
- **Status:** Usable for paper trading. **Not validated for staking real money.**
- **Target:** ≥75% WR on 500+ settled bets with provably positive EV.

---

## Open Items — Can Improve Predictions Now

Nothing left that's both unblocked AND high-impact. The model is feature-complete for
the current data tier. The next round of WR gains comes from data accumulation
(see *Blocked* below).

Remaining buildable items are tracking / quality-of-life:

| Item | Impact | Notes |
|---|---|---|
| **5.9 Closing line value tracking** | Long-term EV validation | Schema change to `bets_log.csv`; fetch closing odds in `settle_bets()`. Doesn't change picks. |
| **5.5 Notifications** | UX | Telegram / email / webhook on pick + settlement. |
| **DATA-05 LR imputation fix** | Medium (when LR activates) | Impute missing context fields as `None` + indicator var instead of `0.0`. Only matters once global LR is live (200+ bets). |

---

## Blocked on Data

Each item triggers automatically once its bet threshold is met. **No code work required now.**

| Item | Impact | Trigger |
|---|---|---|
| **3.8 Auto-refresh BASELINES** | MEDIUM-HIGH — fresher priors per league | 200+ settled bets total (33 away) |
| **4.6 Per-market edge threshold sweep** | MEDIUM-HIGH — eliminate low-value picks | 300+ settled bets |
| **5.7 LR train/test split** | MEDIUM — detect overfitting | Global LR active (200+ bets) |
| **5.8 LR blend weight optimisation** | MEDIUM | After 5.7 |
| **4.5 Combo WR monitoring** | Risk gate | Disable combos if WR < 40% on 50+ bets (currently 56% on 27) |

---

## Deferred — Decided Not to Do

| Item | Reason |
|---|---|
| **5.1 Kelly staking** | No bankroll tracking in scope |
| **5.2 Bankroll tracking** | Out of scope |
| **2.3 Knockout penalty revalidation** | Requires paid API tier |
| **5.6 Web interface** | Premature — need ≥70% live WR first |
| **MODEL-01 Draw-specific model** | Dixon-Coles diagonal sum is already good enough |
| **MODEL-02 Form window sweep** | Low expected delta |
| **MODEL-03 Opponent difficulty weight** | Complex, unvalidated upside |
| **MODEL-04 CL group stage motivation** | Edge case |
| **MODEL-05 Red card rate in referee factor** | Marginal |
| **POTENTIAL-03 Asian handicap cross-check** | Free Odds API doesn't expose AH cleanly |
| **ARCH-01 Backtest replays ML calibration** | Big rewrite, low immediate value |
| **ARCH-04 Refactor `compute_match_probabilities`** | Working code; cosmetic |
| **DATA-01 Disable xG for low-coverage leagues** | Already gated by `_MIN_XG_WEIGHT` |

---

## Metrics — Ready for Production

| Metric | Threshold | Current |
|---|---|---|
| Win rate | ≥ 75% | 53% (167 bets) |
| Sample size | ≥ 500 settled | 167 |
| Brier score | ≤ 0.20 | 0.212 (backtest) |
| 1X2 WR @ model_prob ≥ 0.65 | ≥ 68% | 40% (live) |
| Combo WR | ≥ 40% | 56% (27 bets) |

---

## Changelog — What's Been Built

Lead numbers (e.g. 2.9, BUG-02) are kept so old git history references still make sense.

### Model / accuracy
- **2.9** xG threshold lowered 3.5 → 2.0 (`_MIN_XG_WEIGHT`)
- **2.10** Logger dedup key now `(match_id, market, pick)` — all picks per fixture logged
- **2.11** Per-league `XG_CONV_BY_LEAGUE` dict
- **2.12** No-odds thresholds aligned (always ≥ corresponding `MIN_PROB`)
- **3.9** Fuzzy team-name matching (`SequenceMatcher`, threshold 0.75) in `lookup_odds`
- **4.2** Motivation factor + must-win detection (relegation deficit > games remaining)
- **4.3** `SEASON_CONF_RAMP` = 15 games
- **4.4** Fatigue step function (≤2d full, 3d moderate, ≥4d none)
- **4.7** `EXPECTED_TOTAL_GATES_BY_LEAGUE` / `_CAPS_BY_LEAGUE` per league
- **4.9** Form momentum signal (`MOMENTUM_BOOST/PENALTY`)
- **B3** Per-league Dixon-Coles ρ (`DC_RHO` dict)
- **B5** Combo joint prob from Dixon-Coles matrix (no more multiplication discount table)
- **B6** Draw probability uses league baseline, not residual
- **POTENTIAL-04** Winter under-scoring proxy — `WINTER_UNDER_FACTOR=0.95` Nov–Feb for PL/BL1/FL1/ELC/DED *(2026-05-23)*
- **POSTCAL-EDGE-FILTER** Drop picks whose edge vanishes post-calibration *(2026-05-25)* — `best_value_pick` selects based on pre-calibration edge, but ML calibration can shift `model_prob` enough that the recomputed edge goes ≤ 0. Previously these picks still appeared in VALUE PICKS with `—` for both Edge and Stake (the pick wasn't really value anymore, but it stayed in the section). Fix: edge refresh now runs BEFORE the post-cal filter, and verified-edge picks are dropped if their calibrated edge is no longer positive. The coupon now only shows picks where the calibrated probability and bookmaker odds still produce genuine positive edge.
- **VALUE-ONLY-COUPON** Coupon restricted to genuine value bets *(2026-05-25)* —
   - **Removed BEST AVAILABLE section** entirely. Solo no-odds picks (1.20–1.45 estimated favourites) aren't value bets and have been creating noise. Their useful form is as accumulator legs (Tier 2 already surfaces them).
   - **No-odds picks no longer logged to `bets_log.csv`** — keeps the cohort stats clean and ROI tied exclusively to bets with calculated edge.
   - **`compute_stake_units` no longer clamps up to 1u.** When Kelly recommends less than 0.5 of a unit (banker's rounding to 0), the function returns `None` and the coupon shows `—`. Previously this case force-rounded to 1u, over-betting marginal-edge picks.
   - The two valid value-bet categories are now explicit: (1) singles with real bookmaker odds and positive edge; (2) cross-fixture accumulators (VERIFIED or MODEL).
- **AUDIT-2026-05-25** Five-phase quality audit *(2026-05-25)* — **test suite now fully green: 565/565 passing** (was 545/565 with 20 stale failures). Changes:
   - **Phase A:** 20 stale test assertions updated to current threshold values (Home Win 0.62, Away Win 0.65, Over/Under 2.5 floors 0.65). One DC test rewritten to verify the MIN_FAIR_ODDS gate against real bookmaker odds (it never applied to the no-odds path).
   - **Phase B:** 3 broad `except Exception` blocks tightened. `fetcher.store_matches` and `get_last_match_date` now log via `warn_log` instead of silent `pass`. `match_store.init_db` migration catches `sqlite3.OperationalError` specifically — real schema errors now propagate instead of being silently swallowed.
   - **Phase C:** Input validation guards added at three boundaries — `_safe_prob_pct()` helper in `main.py` protects bet-history view from malformed CSV values; `_best_h2h` and `_best_totals` in `odds_fetcher.py` skip malformed bookmaker entries instead of crashing the whole odds map.
   - **Phase D:** Consistency audit — edge formula and probability-handling are uniform across all 7 sites that touch them. No real inconsistencies found (the stale-edge-after-calibration bug was already fixed in the previous session).
   - **Phase E:** Dead code removed — `cache.clear_cache` (no callers; `evict_stale_cache` covers the use case), two unused `os` imports.
- **STAKING-FIXES** Three staking improvements *(2026-05-25)* —
   - **Bug:** stake column was `—` for some positive-edge picks because pre-calibration `edge` was passed to Kelly while `model_prob` had been calibrated. Edge is now recomputed from the calibrated probability before stake sizing, so display and Kelly math always agree.
   - **Dynamic bankroll:** new menu option 8 sets bankroll at runtime; persisted to `data/user_settings.json`. Default stays €10. `staking.set_bankroll(eur)` for programmatic use. All stake amounts auto-scale.
   - **Table fit:** coupon columns retuned (Match 50→40, Pick 24→22, padding tightened) so Stake column never truncates on the 140-char console.
- **DRIFT-DETECTOR + ISOTONIC-CORRECTION** Per-(market, pick) calibration drift detection + opt-in isotonic correction *(2026-05-25)* — new `drift.py` computes actual-WR vs predicted gaps for the current cohort (min 30 settled bets per bucket); rows with `|gap| ≥ 10pp` are flagged yellow, `≥ 15pp` red. After the coupon renders, drifted buckets appear in a footer table; in interactive mode the user is prompted (y/N) to fit an isotonic correction. Corrections persist in `data/isotonic_corrections.json` and are loaded by `ml_calibrator.calibrate()` as Tier 0 before the LR tiers (LR features still see raw `model_prob` so we don't double-correct). The ±15pp safety clamp still bounds the final output. Re-prompts only after 20+ new bets since the last fit to avoid prompt fatigue. Implements `MISSING-01` from the previous roadmap.
- **STAKING-KELLY** Quarter-Kelly stake sizing *(2026-05-24)* — new `staking.py` module computes per-pick stake in units. Bankroll = €10, 1 unit = 1% (€0.10), quarter-Kelly fraction, capped at 10 units (10% of bankroll). Picks with real bookmaker odds get a recommended stake; no-odds picks show `—`. Verified accas get joint-Kelly stakes; MODEL accas don't (prices uncertain). New `stake_units` column in `bets_log.csv` (backwards compatible — existing rows blank). Coupon shows inline `Stake` column with euro amount. Tunable constants live at the top of `staking.py`.
- **VERSIONING-SEQUENTIAL** Switched model version scheme from date-based (`v2026-04-27`) to sequential (`v1`) *(2026-05-24)* — 229 existing rows renamed; comment block in `logger.py` documents what's in each version. Bump on threshold/calibration/scoring changes; don't bump on additive features or bug fixes.
- **ROI-VERSIONED** ROI summary now filterable by `model_version` *(2026-05-24)* — `compute_roi_summary(model_version=…)` isolates the current cohort from retired pre-threshold history. UI shows both "Current model" and "Lifetime" lines in the bet-history panel. `MODEL_VERSION` bumped to `v2026-05-24` for the acca release.
- **ACCA-CROSS** Safe cross-fixture accumulators *(2026-05-24)* — pools short-odds high-confidence picks across different leagues into 2–3 leg parlays. Two tiers:
   - **VERIFIED**: every leg has real bookmaker odds, joint edge ≥ 8%. Logged with `ACC:` match_id prefix and `AccaCross` market type; `settle_bets` parses the synthetic id, fetches each leg, and W's only if all surviving legs win; postponed legs are dropped and ROI is recomputed on the survivors.
   - **MODEL** (informational): at least one leg uses model fair odds (when the Odds API has no price). Shown in the coupon with a `~` marker; not logged for ROI since the real bookmaker price will differ.
   - Filters: per-leg prob ≥ 0.70, odds in 1.15–1.45, combined odds 1.60–2.50, one leg per fixture, one leg per league. Includes 1X2, Double Chance, and Over/Under markets.
- **POTENTIAL-06** Streak detection (`cs_streak`, `drought_streak`)
- **C1** xG proxy from shots-on-target
- **C3** Richer ML calibrator features (9 features global, 8 per-league)
- **C4** `scripts/calibrate_baselines.py` for empirical league baselines

### Quality / silent-failures
- **3.1** Fallback logger (`warn_log.py`)
- **3.2** Probability sum asserts + zero-div guards
- **3.5** `CACHE_VERSION` cache-busts on API schema changes
- **3.6** SQLite write retries with backoff
- **5.3** `MODEL_VERSION` stamped on every pick row
- **5.4** Standings cache age display in warnings screen
- **DATA-02** `matches.db` pruning (2 seasons)
- **DATA-03** Cache file housekeeping
- **DATA-04** API call budget tracker
- **ARCH-03** `validate_config()` on startup
- **POTENTIAL-05** Composite bet quality score for pick ranking

### Bug fixes (chronological)
- Combo settlement gap — all 13 combo types now handled
- Unknown combo legs settle as `ERR` (was: phantom losses)
- API key moved to `.env`
- Home advantage double-count for new teams
- xG trend double-counted home advantage
- xG unit mismatch in trend factor (mixed-units division)
- SQLite parallel-write data loss
- Probability normalisation zero-div
- ML calibrator activated too early (raised to 100/200 thresholds)
- Auto-baseline silent updates (now `--update-baselines` only)
- H2H orientation leak
- Python 3.10 `datetime.UTC` crash
- Cache JSON corruption crash
- SQLite double-close in retry loop
- `ml_calibrator._seg` accessed directly externally
- 2 redundant API calls per fixture
- `odds_fetcher` dead `line_move` feature
- `fetcher.get_match_stats()` dead function
- Stale root-level `backtest.py`
- Backtest form-score formula mismatch
