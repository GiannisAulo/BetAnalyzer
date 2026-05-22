# BetAnalyzer — Project Roadmap & Accuracy Improvement Plan

> Full codebase analysis completed 2026-04-18.
> Goal: identify bugs, weak calculations, and missing features to increase prediction accuracy.

---

## Current State

### What works well
- Dixon-Coles score matrix with ρ = -0.13 correction for low-score cell bias
- Recency-weighted team history (exponential decay, k=0.5 per 30 days)
- Home/away split strength indices via `_compute_strength_factors`
- 3-tier ML calibration: segment stats → global LR → per-league LR
- Fatigue penalty (continuous exponential curve by rest days)
- Motivation factor (title race, European spots, relegation battle, dead rubbers)
- Referee tendency multiplier (goals-per-game from match history)
- Knockout stage suppression (CL/cup defensive setups)
- Rolling Over 2.5 / BTTS blend with clean-sheet model

---

## TIER A — Bugs / Misimplementations

> These are wrong today. Fix before anything else.

---

### ~~A1. `over_3.5` not blended with rolling rate~~ ✅ DONE
**File:** `analyzer.py` ~line 1078
**Fix applied:** `p_over_35 = p_over_35 * 0.7 + rolling_over35 * 0.3`

---

### A2. H2H goals blend is inconsistent with venue-split filtering ← next
**File:** `analyzer.py` ~line 1027
**Problem:** `parse_h2h` applies venue-split filtering (only meetings where `home_team_id` was the home side) when the sample is ≥ 3. The win-rate signals come from this filtered subset. But the `total_goals` list blended into `exp_total` is also from this subset — which is correct in isolation. However when the fallback to all meetings is used, `total_goals` includes both home and away legs, producing a goals average that doesn't match the 1X2 baseline orientation (home team always on left). The blend can therefore nudge goals in the wrong direction.
**Fix:** Store venue-split goals separately in `parse_h2h` result and use that list in the blend. When the venue-split fallback fires, skip the H2H goals blend rather than using a mixed list.

---

### ~~A3. Motivation factor double-counts opponent motivation on xG~~ ✅ DONE
**Fix applied:** `exp_home *= home_motiv` / `exp_away *= away_motiv`

---

### ~~A4. Fatigue threshold fires for 13 days rest (essentially noise)~~ ✅ DONE
**Fix applied:** `_FATIGUE_REST_FULL = 7.0`

---

### ~~A5. Dead code — `kelly_stake` and `market_variance_penalty`~~ ✅ DONE
**Fix applied:** Both functions deleted from `markets.py`.

---

### ~~A6. Orphaned `DEFAULT_BANKROLL` in config~~ ✅ DONE
**Fix applied:** Constant deleted from `config.py`.

---

## TIER B — Weak Calculations

> The code runs and produces numbers, but the calculations are inaccurate or internally inconsistent. These have meaningful impact on prediction quality.

---

### ~~B1. Form score uses overall form string — no home/away split~~ ✅ DONE
**Fix applied:** `home_form_score` / `away_form_score` computed in `parse_team_history` from last 6 venue-specific results; used in `compute_match_probabilities` with fallback to standings string.

---

### ~~B2. Season confidence scaling (`season_conf`) does not guard H2H or trend modifier~~ ✅ DONE
**Already implemented:** Both H2H nudge and C.1 trend modifier already multiplied by `season_conf` — confirmed on inspection.

---

### B3. Dixon-Coles `_RHO = -0.13` is a fixed constant across all leagues
**File:** `analyzer.py` line 533
**Problem:** ρ = -0.13 comes from the original 1997 Dixon-Coles paper fitted on English football in the 1990s. High-scoring leagues (BL1 ~3.1 gpg) have proportionally fewer 0-0 and 1-0 scorelines than defensive leagues (SA ~2.5 gpg). A single ρ distorts the score matrix for non-English leagues.
**Fix:** Add a `RHO` dict to `config.py` per league (e.g. `{"PL": -0.13, "BL1": -0.10, "SA": -0.15, ...}`) and look up the correct value in `_score_matrix`. Empirically tune from historical score distributions.

---

### ~~B4. `_compute_strength_factors` computes league split averages from only 2 teams~~ ✅ DONE
**Fix applied:** Full league `_league_avg_scored` / `_league_avg_conceded` used as denominator for all split indices.

---

### ~~B5. Combo joint probability ignores the score matrix~~ ✅ DONE
**Fix applied:** `_joint_prob_from_matrix()` added to `markets.py`; `evaluate_combos(all_picks, probs=probs)` now derives exact P(leg1 AND leg2) from the Dixon-Coles grid. Discount table removed.

---

### B6. Draw probability is a residual — no draw-specific model
**File:** `analyzer.py` ~line 851
**Problem:** The form/position baseline assigns draw probability as `1 - home_prob - away_prob`. The Dixon-Coles matrix provides a proper draw probability from all `matrix[i][i]` cells, and the 80% DC blend helps significantly. However the 20% baseline component still uses residual draw — teams with extreme form differentials get a draw probability that is driven entirely by what's left over, not by any draw-specific signal.
**Fix:** In the baseline, compute draw independently: use the league-specific draw baseline from `BASELINES` and add a small form-based adjustment. Only adjust draw downward when form differential is very large (one team dominant), rather than letting it be a residual of two independent adjustments.

---

## TIER C — Missing Features

> Not currently implemented. Each has meaningful accuracy upside.

---

### C1. No xG data — historical goal counts are a noisy proxy
**Problem:** The Poisson lambda estimates rely on goals scored/conceded from team history. Goals are high-variance (a single lucky deflection counts the same as a clinical finish). Expected goals (shots × shot quality) is a far better predictor of future scoring. The football-data.org API returns match statistics (shots, shots on target, corners) for some leagues — these are not currently fetched.
**Plan:** Fetch match statistics from `/v4/matches/{id}` for stored historical matches. Use `shotsOnTarget` as an xG proxy: `xG_proxy = shots_on_target × 0.35`. Replace raw `goals_scored_list` with `xg_proxy_list` in `parse_team_history`. The variance nudge in A.5 becomes more meaningful with xG data.

---

### C2. No odds line movement tracking
**Problem:** The odds fetcher fetches current prices but doesn't record opening prices or track movement. Sharp line movement (odds shortening significantly between open and match time) is one of the strongest publicly available signals for true edge — it indicates professional money agreeing with the model. Currently all edges are treated identically regardless of market confidence.
**Plan:**
1. On first daily fetch, store prices with a `fetched_at` timestamp in the cache.
2. On subsequent fetches (e.g. an hour before kickoff), compare current vs stored opening price.
3. Add `line_movement` field to the odds dict: `(current - opening) / opening`.
4. Expose in `lookup_odds` result and display in coupon. Boost rating by +1 when line moves in the same direction as the model pick.

---

### ~~C3. ML calibrator features are too sparse~~ ✅ DONE
**Fix applied:**
- `bets_log.csv` schema extended with 4 new columns: `home_position`, `away_position`, `form_adv`, `expected_total`
- `logger.py`: `log_bets` now writes all 4 fields; `FIELDS` updated
- `ml_calibrator.py`: `_load_history` reads all new fields; global LR uses 9 features `(market, league, model_prob, edge, home_pos, away_pos, form_adv, exp_total, odds_taken)`; per-league LR uses 8 features (no league index)
- `main.py`: context features computed before calibration loop and attached to every pick dict; `extra=` dict passed to `cal.calibrate()`
- Old rows with blank values imputed with sensible defaults (position=0.5, form_adv=0.0, exp_total=2.5)

---

### ~~C4. No cross-league calibration of league-specific biases~~ ✅ DONE
**Fix applied:** `scripts/calibrate_baselines.py` created.
- Reads all settled 1X2 rows from `bets_log.csv`
- Groups by league, computes empirical home/draw/away win rates
- Compares against `BASELINES` in `config.py` and prints delta table with colour coding
- Recommends config updates when a league has ≥ 200 bets AND delta ≥ 2 pp
- Run with: `python scripts/calibrate_baselines.py` from the `football_tipster/` directory

---

## TIER D — Code Quality

| Item | File | Description |
|------|------|-------------|
| D1 | `main.py` ~line 141 | `_compute_strength_factors` called twice per fixture — once league-wide (correct) and once per-fixture with 2-team histories (incorrect, see B4). Refactor to single call. |
| D2 | `coupon.py` | Not reviewed. Check for any remaining stake/bankroll references in render functions. |
| D3 | `config.py` | `RATE_LIMIT_SLEEP = 6` is defined but never used. Remove or wire up. |
| ~~D4~~ ✅ | `ml_calibrator.py` | `reset_calibrator()` called in `main.py` after `settle_bets()` when new settlements are written. |

---

## Priority Order

| # | Item | Accuracy Impact | Status |
|---|------|-----------------|--------|
| 1 | **A3** — Fix motivation factor (divide bug) | High | ✅ Done |
| 2 | **A4** — Fix fatigue threshold (14 → 7 days) | Medium | ✅ Done |
| 3 | **B4** — Fix league split average (2-team sample) | High | ✅ Done |
| 4 | **A1** — Blend over_3.5 with rolling rate | Medium | ✅ Done |
| 5 | **B2** — Apply season_conf to H2H and trend | Medium | ✅ Done |
| 6 | **B5** — Use score matrix for combo probabilities | High | ✅ Done |
| 7 | **D4** — Reset calibrator after settle_bets | Medium | ✅ Done |
| 8 | **B1** — Home/away split form score | Medium | ✅ Done |
| 9 | **A5/A6/D3** — Remove all dead bankroll code | Low | ✅ Done |
| 10 | **B3** — Per-league Dixon-Coles ρ | Medium | ✅ Done |
| 11 | **A2** — H2H goals blend venue-split fix | Medium | ✅ Done |
| 12 | **B6** — Draw probability not a residual | Medium | ✅ Done |
| 13 | **D1/D2/D3** — Code quality cleanup | Low | ✅ Done |
| 14 | **C2** — Odds line movement tracking | High | ✅ Done |
| 15 | **C3** — Richer ML calibrator features | High | ✅ Done |
| 16 | **C1** — xG proxy from shot statistics | Very High | ⬜ |
| 17 | **C4** — Baseline calibration script | Medium | ✅ Done |

---

## Notes

- Items 1–7 are safe to implement back-to-back without schema changes.
- Items 11–13 require a `bets_log.csv` column migration (add columns, existing rows get blank defaults — fully backwards compatible).
- Item 13 (xG) depends on football-data.org returning match stats for your subscribed leagues — verify coverage before implementing.
