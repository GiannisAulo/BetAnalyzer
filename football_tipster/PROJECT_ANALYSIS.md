# Football Tipster — Project Analysis & Audit

**Date:** 2026-04-28  
**Settled bets:** 167  
**Live WR:** 53% overall (58% new-model cohort)  
**Backtest WR:** 67–71%

This document is an honest audit of the full codebase. Every finding has been verified against
the source. False positives from automated analysis have been filtered out.

---

## 1. Bugs — Issues That Cause Wrong Predictions or Silent Failures

---

### BUG-01 · Logger dedup key only on match_id
**Impact: HIGH**  
**File:** `logger.py:297`

```python
if mid and mid in already_logged:
    continue
...
already_logged.add(mid)  # added after FIRST pick for this match_id
```

`_load_logged_ids()` returns a set of match_id strings. When a fixture generates two picks (e.g.
Home Win + Over 2.5), the second pick is silently skipped because `mid` is already in
`already_logged` after the first write. This means:

- The bets log has at most one pick per fixture, no matter how many were generated.
- ML training data is artificially sparse for multi-pick fixtures.
- The user never sees that the second pick was suppressed.

**Fix:** Change dedup key to `(match_id, market, pick)` tuple so multiple distinct picks for the
same fixture can all be logged.

```python
already_logged = {(r["match_id"], r["market"], r["pick"]) for r in _load_rows()}
...
key = (mid, pick.get("market",""), pick.get("pick",""))
if mid and key in already_logged:
    continue
already_logged.add(key)
```

---

### BUG-02 · xG proxy threshold almost never activates
**Impact: HIGH**  
**File:** `analyzer.py:298`

```python
_MIN_XG_WEIGHT = 3.5
xg_home = wavg(home_xg_w, home_xg_total) if home_xg_total >= _MIN_XG_WEIGHT else None
```

`home_xg_total` accumulates the sum of recency-decay weights for matches with shot data.
With `FORM_DECAY_K=0.5`, a match 30 days ago has weight ≈ 0.607. Four matches over 30 days
produce total weight ≈ 0.607 + 0.68 + 0.76 + 0.84 = 2.89 — **below the 3.5 threshold**.
Only teams with four matches all within roughly the last 10 days reliably pass the gate.

Since most domestic leagues play every 7–10 days, a team's last 4 matches easily span 21–28
days, giving weight sums of 2.6–3.2. xG effectively never activates during normal periods.
The model silently falls back to raw goals for almost every fixture.

**Fix:** Lower to `_MIN_XG_WEIGHT = 2.0`, or switch to a minimum match count (`if xg_match_count
>= 3`) independent of decay weight. The weight sum penalises teams for having recent matches
further apart — that is not the intent.

---

### BUG-03 · XG_CONV is uniform across all leagues
**Impact: MEDIUM**  
**File:** `analyzer.py:15`

```python
XG_CONV = 0.33  # shots-on-target × XG_CONV = xG estimate
```

SoT-to-goal conversion rates vary meaningfully by league:
- Bundesliga: ~0.28–0.30 (high SoT volume, lower conversion)
- La Liga / Serie A: ~0.33–0.35
- Brasileirão (BSA): ~0.30–0.32
- Championship (ELC): ~0.29–0.31

Using 0.33 uniformly overstates xG for Bundesliga by ~10% and understates it for La Liga.
This creates systematic Over/Under bias that differs by league — the opposite of what the
per-league DC_RHO correction is trying to fix.

**Fix:** Add `XG_CONV_LEAGUE` dict in `config.py` keyed by league code; default to 0.33 for
unconfigured leagues. Run backtest per league to find optimal values.

---

### BUG-04 · Strength factors computed twice per fixture
**Impact: LOW**  
**File:** `main.py:122, 177`

```python
strength_factors = analyzer._compute_strength_factors(standings)          # line 122
...
strength_factors_fx = analyzer._compute_strength_factors(standings, team_histories)  # line 177
```

The first call (without team histories) computes league-average attack/defence for use in
`parse_team_history()`. The second call (with histories) computes refined per-team strength
for `compute_match_probabilities()`. The first result is only used for `league_avg_gpg` (line
199), which could be extracted more cheaply. This isn't a correctness bug but adds latency
per fixture (O(n_teams) computation twice).

**Fix:** Extract `league_avg_gpg` from the second call instead; remove the first call.

---

### BUG-05 · settle_bets unrecognized combo leg silently logs "L"
**Impact: MEDIUM**  
**File:** `logger.py:100–109`

When a combo leg type is not matched by `_evaluate_leg()`, the function returns `None`, and
the outer code treats that as a loss. The bet settles as "L" with no warning to the user
and no entry in `warnings.log`. If a combo pick name was ever stored with a typo or the
market categories change, it silently corrupts the win rate with phantom losses.

**Fix:** Log unrecognized legs to `warn_log.py` with the exact leg string. Mark result as
`"ERR"` (not `"L"`) and add `"ERR"` to the `_LOSS_VALUES` set so it's excluded from calibration.

---

### BUG-06 · No-odds threshold inconsistency for Over 2.5
**Impact: LOW**  
**File:** `markets.py:27, 136`

```python
MIN_PROB = {"Over 2.5": 0.62, ...}
...
elif name == "Over 2.5" and model_prob >= 0.65:   # no-odds path
    picks.append(...)
```

When odds are present, the floor is 0.62. When odds are absent, the floor is 0.65. These
should be the same value — the probability floor represents model confidence, not whether
you have a price. The discrepancy means picks at 0.62–0.65 probability appear when odds
are available but disappear when they are not, which is backwards (you need *more* confidence
when you can't verify value against a market price).

**Fix:** Align no-odds thresholds with `MIN_PROB` values. For Over 2.5 specifically, the
no-odds threshold should be ≥ 0.68 (higher than the odds floor, since you cannot verify EV).

---

## 2. Missing Implementations

---

### MISSING-01 · Isotonic correction not wired into live picks
**Impact: HIGH**  
**File:** `scripts/calibration_curves.py` (analysis only), no integration in pipeline

`calibration_curves.py` reports the bias per market but does not apply any correction to
`model_prob` during live predictions. The Over 2.5 model is currently −15pp biased. Every
pick using `model_prob` for thresholding is operating on a wrong number.

**Fix:** Once a pick type reaches 50+ settled bets, fit the isotonic regression and serialize
the correction map to a JSON file (e.g., `data/isotonic_corrections.json`). In
`ml_calibrator.calibrate()`, load this map at startup as a Tier 0 correction applied before
the LR tiers. Trigger: run `scripts/calibration_curves.py --apply` to regenerate the map.

---

### MISSING-02 · BASELINES never auto-refreshed
**Impact: MEDIUM-HIGH**  
**File:** `scripts/calibrate_baselines.py`, `config.py`

`calibrate_baselines.py` exists and works, but it only runs when explicitly invoked with
`--update-baselines`. The BASELINES in `config.py` are static values from multi-season
analysis. As the model accumulates 200+ settled bets, empirical win rates will diverge
from these priors, but nothing updates them automatically.

**Fix:** In the `_show_bet_history()` or a scheduled hook: if settled bets ≥ 200 and last
baseline update > 30 days ago, run `auto_update_baselines()` and log the timestamp. Gate
this on a minimum sample (≥ 20 per league) to avoid updating on noise.

---

### MISSING-03 · Odds team name matching has no fuzzy fallback
**Impact: HIGH**  
**File:** `odds_fetcher.py`

The `_NAME_MAP` dict has ~80 hardcoded team name translations. When a team name from the
football-data.org API doesn't match any entry, the fixture gets no live odds and the pick
is generated without edge calculation. This affects:

- Any new promoted team not in the map
- Teams with alternate name spellings (e.g. "Nottingham Forest" vs "Nott'm Forest")
- International fixtures where API returns local-language names

Without odds, the model can only use probability floors (0.58+) with no EV gate, meaning
the selection criterion is weaker.

**Fix:** After exact map lookup fails, run `difflib.SequenceMatcher` against all API team
names in the response. Accept the match if ratio ≥ 0.80. Log unmatched pairs to
`warnings.log` for manual map updates.

---

### MISSING-04 · ML calibrator has no train/test split
**Impact: MEDIUM**  
**File:** `ml_calibrator.py:159`

The logistic regression is trained on all 200+ settled bets, then immediately used to
calibrate picks from the same distribution. Reported calibration accuracy (segment hit
rates, Brier scores in backtest) is in-sample. If the model is overfitting, this is
invisible.

**Fix:** Implement time-based train/test split: train on bets older than 60 days, evaluate
on last 60 days. Report out-of-sample Brier score in `scripts/calibration_curves.py`. Only
apply LR calibration if out-of-sample Brier < uncalibrated Brier.

---

### MISSING-05 · LR blend weights have no empirical basis
**Impact: MEDIUM**  
**File:** `ml_calibrator.py:353, 378`

```python
calibrated = 0.60 * lr_prob + 0.40 * raw_prob   # per-league LR
calibrated = 0.55 * lr_prob + 0.45 * raw_prob   # global LR
```

These weights were chosen without a sweep. A 60/40 blend could be over-correcting (high
variance in small samples) or under-correcting (systematic bias not fully addressed).

**Fix:** Add `LR_BLEND_WEIGHT` to `config.py`. Run `--sweep-lr-blend` mode in `backtest.py`
testing 0.40–0.80 in 0.10 steps. Use the value that minimises out-of-sample Brier score.

---

### MISSING-06 · Expected total gates are not league-specific
**Impact: MEDIUM**  
**File:** `markets.py:35–42`

```python
_EXPECTED_TOTAL_GATES = {"Over 2.5": 2.90, "Over 3.5": 3.50}
_EXPECTED_TOTAL_CAPS  = {"Under 2.5": 2.50, "Under 3.5": 4.00}
```

League averages vary significantly:
- BSA, SA: ~2.9–3.0 goals/game → Over 2.5 gate of 2.90 suppresses all picks
- BL1: ~3.1 → gate passes comfortably
- PL: ~2.75 → gate of 2.90 is too strict, suppresses valid picks

A uniform gate of 2.90 penalises lower-scoring leagues. Fixtures in PL or PPL with genuine
Over 2.5 signal get filtered out even when model_prob is 65%+.

**Fix:** Add `EXPECTED_TOTAL_GATES_LEAGUE` dict in `config.py` per league. Fallback to current
global values for unconfigured leagues.

---

### MISSING-07 · No closing line value tracking
**Impact: HIGH (for long-term EV validation)**  
**File:** not implemented

The most reliable long-term indicator of a profitable model is whether picks consistently
beat the closing line (the bookmaker's final pre-match odds). Currently, `odds_taken` is
logged from whatever odds the API returned when picks were generated — often the opening
line. There is no fetch of closing odds and no comparison.

**Fix:** In `settle_bets()` (or a separate script), fetch the current odds for each pending
bet on settlement day. If odds have moved against the pick (e.g., model picked Home Win at
1.80, closing line is 1.50), log `closing_implied_prob` and compute closing line value (CLV).
Positive average CLV over 100+ picks is the strongest signal of a genuinely predictive model.

---

## 3. Model & Algorithm Gaps

---

### MODEL-01 · Draw probability is a residual, not modelled
**Impact: MEDIUM**

The Dixon-Coles score matrix gives P(home goals=i, away goals=j) for all (i,j). P(Draw) is
computed as `sum(matrix[i][i])` — the diagonal sum. The rho (DC_RHO) correction applies
specifically to the 0-0 and 1-1 cells to adjust the model's systematic underestimation of
draws. However:

- Draw WR is not tracked separately in `calibration_curves.py`.
- DC_RHO values are set from multi-season analysis but not re-validated against live bets.
- The model makes no Draw picks currently (MIN_PROB = 0.45, which is rarely reached given
  typical 25–35% draw probabilities in Poisson models).

**Fix:** Track Draw market outcomes separately. Add per-league DC_RHO sweep to `backtest.py`.
Consider lowering Draw MIN_PROB to 0.40 if backtest shows acceptable WR.

---

### MODEL-02 · Form score uses last 6 results with arbitrary window
**Impact: MEDIUM**  
**File:** `analyzer.py:_compute_form_score()`

Form is computed from the last 6 matches. Six is a reasonable window but was not validated
against alternatives (4, 8, or dynamic by games-played). Early in a season (first 6 games),
the window spans the entire season history, making form and position perfectly correlated.
Later in the season (game 30+), a team's recent 6 may include cup games or anomalies.

**Fix:** Run backtest sweeping form window (4, 5, 6, 8 matches) and compare Brier scores per
league. Also investigate separating league form from cup form.

---

### MODEL-03 · Opponent difficulty weighting has no backtest validation
**Impact: MEDIUM**  
**File:** `analyzer.py:parse_team_history()`

When computing rolling averages, the code weights goals/shots by recency (decay) but does
not adjust for opponent quality. A team scoring 3 goals against a bottom-half side gets the
same weight as scoring 3 against a top-half side. This makes form averages noisy for teams
with uneven fixture schedules.

**Fix:** Compute `opponent_difficulty = 1 / (1 + strength_factors[opponent]["attack"])` as a
multiplier on the match weight. Teams facing weak opposition get lower-weighted goal tallies.
Validate via backtest Brier score improvement.

---

### MODEL-04 · No model for draw/win probability in early-season knockouts (CL group stage)
**Impact: LOW-MEDIUM**  
**File:** `config.py:CL_KNOCKOUT_STAGES`

The knockout factor (`KNOCKOUT_GOALS_FACTOR = 0.93`) is applied to CL knockout stages. But
the group stage also differs from domestic football: teams rest key players, results can be
meaningless if qualification is already decided, and European experience matters more than
domestic form. None of this is captured.

**Fix:** Add a `CL_GROUP_STAGE_MOTIVATION_FACTOR` that adjusts expected goals based on whether
the team is already through or eliminated. Detect this from standings data.

---

### MODEL-05 · Referee factor ignores red card history
**Impact: LOW**  
**File:** `analyzer.py:compute_referee_factor()`

`compute_referee_factor()` uses goals-per-game from referee history. A referee who shows many
red cards significantly changes match dynamics (10-man game, defensive play, more pens).
Red card rate is available in the API `statistics` endpoint but is not fetched or used.

**Fix:** Fetch red card count per referee match. If referee reds/match > league average + 1
std, apply a `DEFENSIVE_MODE_FACTOR` (e.g. 0.88) to expected goals.

---

## 4. Untapped Potential

---

### POTENTIAL-01 · Per-league XG_CONV calibration
**Impact: HIGH (when xG activates)**

As noted in BUG-03, `XG_CONV = 0.33` is uniform. Even if it were fixed as a per-league
constant, a more powerful approach is to compute it empirically: for each league, regress
`actual_goals ~ shots_on_target` from the match history database. `matches.db` already
stores shot data. This would give league-specific, data-driven conversion rates rather than
constants set by external research.

**Implementation:** `scripts/fit_xg_conv.py` — query `matches.db`, group by league, fit OLS
`goals = beta * sot` (no intercept), store `beta` per league in `config.py`.

---

### POTENTIAL-02 · Form momentum (trend direction) vs form level
**Impact: MEDIUM-HIGH**

Currently `form_score` is a level (how good is the team overall). It doesn't capture
momentum: a team on a 5-game winning streak has the same form_score as a team that won 4
games months ago and lost their last one. Momentum (first derivative of form) is a separate
signal with genuine predictive value.

**Implementation:** In `parse_team_history()`, compute `recent_3_form` and `recent_6_form`
separately. If `recent_3_form > recent_6_form + 0.15`, the team is on an upswing.
Apply a `MOMENTUM_BOOST = 1.05` to attack and `MOMENTUM_DEF_BOOST = 0.97` to conceded.

---

### POTENTIAL-03 · Asian handicap probability as a cross-check
**Impact: MEDIUM**

The Odds API returns Asian handicap lines alongside 1X2. AH lines from sharp bookmakers
(Pinnacle) are the closest thing to a consensus probability. If the model's home win
probability is 0.65 but the AH-implied probability is 0.55, that gap is a red flag. Using
AH as a sanity check against the model would filter out picks where the model is clearly
miscalibrated.

**Implementation:** In `odds_fetcher.py`, extract `asian_handicap` market. Compute
AH-implied home probability. In `markets.py`, suppress picks where
`abs(model_prob - ah_implied) > 0.15` (model is more than 15pp away from sharp consensus).

---

### POTENTIAL-04 · Weather and pitch condition proxy
**Impact: LOW-MEDIUM**

Heavy rain increases under-scoring probability by 10–15% in research studies. The model
has no weather input. While fetching live weather data adds complexity, a seasonal proxy
exists: winter months in Northern Europe (Nov–Feb) correlate with more defensive, under-
scoring matches. Adding a `WINTER_UNDER_BOOST_LEAGUES` dict (BL1, FL1, PL, ELC) with a
small multiplier on expected total would be zero API cost.

**Implementation:** In `compute_match_probabilities()`, detect month from fixture date. Apply
`expected_total *= 0.95` for winter months in Northern European leagues.

---

### POTENTIAL-05 · Bet quality score for pick prioritisation
**Impact: MEDIUM**

Currently picks are sorted by `model_prob` (no-odds path) or `edge` (odds path). Neither is
a composite quality score. A pick at 0.70 prob with no odds is shown above a 0.65 prob pick
with +12% edge — but the latter is likely better value. A quality score combining
probability, edge, and model confidence (from `segment_uncertainty`) would rank picks more
usefully.

**Implementation:**
```python
quality = model_prob * 0.4 + (edge/20 if edge else 0) * 0.4 + (1 - uncertainty) * 0.2
```
Use as sort key in `_pick_sort_key()`.

---

### POTENTIAL-06 · Streaks and run detection (won't score, won't concede)
**Impact: LOW-MEDIUM**

A team on a 5-match clean sheet streak is a much stronger Under signal than its rolling
conceded average suggests. Rolling averages smooth out these signals. A binary `clean_sheet_streak`
and `scoring_drought` count (matches without scoring) would capture current defensive/offensive
patterns more sharply.

**Implementation:** In `parse_team_history()`, count consecutive tail matches in the sorted
list where clean_sheet/no_goal condition holds. Attach to history dict. In
`compute_match_probabilities()`, apply a `CLEAN_SHEET_STREAK_BOOST` to defence parameter.

---

## 5. Code Quality & Architecture

---

### ARCH-01 · Backtest does not replay the full ML calibration pipeline
**Impact: HIGH**  
**File:** `scripts/backtest.py`

The backtest replays match probability computation and pick selection, but it does NOT
replay ML calibration (the LR models) or simulate how the calibrator would have been
trained on historical data at each point in time. This means:

- Backtest WR (67–71%) is measured on raw model_prob, not calibrated prob.
- When ML calibration activates live (200+ bets), the live picks will differ from backtest.
- The reported Brier score (0.212) is for the raw model; calibrated model may be better or worse.

**Fix:** Implement temporal simulation in backtest: when replaying match N, train a calibrator
on the first N-1 matches' outcomes, then apply it to match N's probabilities. This gives
a realistic forward-test rather than an in-sample replay.

---

### ARCH-02 · `_load_rows()` in logger re-reads the CSV on every call
**Impact: LOW**  
**File:** `logger.py:28–36`

`_load_logged_ids()` calls `_load_rows()` which opens and reads the entire CSV. It is called
in `log_bets()` and `settle_bets()`, but also indirectly from `_show_bet_history()` (via
`compute_roi_summary()`). On a CSV with 1000+ rows, this becomes noticeable. No caching.

**Fix:** Add a module-level `_ROW_CACHE` with a dirty-flag. Invalidate on write.

---

### ARCH-03 · Config has no validation on startup
**Impact: LOW**  
**File:** `config.py`

`config.py` defines many critical constants (HOME_ADV, DC_RHO, FATIGUE_*) but no startup
validation ensures they are within sane ranges. A misconfigured `DC_RHO = 0.5` (instead of
-0.10) or `FATIGUE_SEVERE_PENALTY = 2.0` (instead of 0.10) would produce wildly wrong
probabilities with no error.

**Fix:** Add a `validate_config()` function with assertions (e.g. `-0.5 < DC_RHO < 0`,
`0 < HOME_ADV < 2.0`). Call at import time.

---

### ARCH-04 · `compute_match_probabilities()` is 300+ lines with no sub-functions
**Impact: LOW (maintainability)**  
**File:** `analyzer.py:850–1299`

The main probability computation function is a monolith. It handles: fatigue adjustment,
motivation adjustment, position adjustment, form adjustment, xG blending, trend calculation,
H2H blending, Poisson matrix generation, market probability extraction, and normalisation —
all inline. Adding a new factor requires reading the entire function to find the right
insertion point.

**Fix:** Extract each adjustment into a named function: `_apply_fatigue()`, `_apply_motivation()`,
`_blend_h2h()`, etc. The main function becomes a readable pipeline of calls.

---

### ARCH-05 · Market evaluators lack parameter documentation
**Impact: LOW (maintainability)**  
**File:** `markets.py`

`evaluate_1x2()`, `evaluate_over_under()` etc. accept `probs` as a dict but the expected
keys are not documented in the signature. New contributors (or future-you) have no way to
know which keys are required without reading `compute_match_probabilities()` output.

**Fix:** Add TypedDict `MatchProbs` with all required keys; type-hint `probs: MatchProbs`.

---

## 6. Data Pipeline Concerns

---

### DATA-01 · Shot data coverage is very low for most leagues
**Impact: HIGH (for xG reliability)**

`warn_log.py` logs when a team's shot-data coverage is < 70%. Given BUG-02 (xG threshold
rarely passes), and the fact that the API's free tier returns shots only for some competitions,
xG is effectively unused for: PPL, DED, BSA (no SoT in free API response), and infrequently
for SA, PD. The model claims to use xG but degrades silently to raw goals for 60–70% of
fixtures.

**Fix:** Track `xg_coverage_rate` per league in `_show_bet_history()`. Print a warning if
< 30% of a league's matches have xG activated. Consider disabling the xG blend entirely for
leagues where it never activates (BSA, PPL, DED) to avoid false confidence in the model path.

---

### DATA-02 · `matches.db` has no pruning strategy
**Impact: LOW-MEDIUM**  
**File:** `match_store.py`

Historical matches accumulate indefinitely in `matches.db`. Currently 389KB but will grow
without bound. Older matches are already heavily discounted by recency decay, so matches from
3+ seasons ago contribute essentially zero to predictions but still slow down queries.

**Fix:** Add a retention policy: delete matches older than `MAX_HISTORY_DAYS = 730` (2 seasons)
during `_merge_history()`. Run as part of the daily standings refresh.

---

### DATA-03 · Cache has no housekeeping; stale files accumulate
**Impact: LOW**  
**File:** `cache.py`

Every league × date combination generates a new `.json` file in `.cache/`. TTLs are checked
at read time but expired files are never deleted. A full season of daily runs will create
~3000+ cache files. There is no cleanup on startup.

**Fix:** On startup (or weekly), scan `.cache/` and delete files older than 2× the longest
TTL (48h for most resources). Add to `main.py` startup sequence.

---

### DATA-04 · API rate limiting has no global budget tracker
**Impact: MEDIUM**  
**File:** `fetcher.py`

The football-data.org free tier allows 10 requests/minute. `fetcher.py` adds a 1-second
sleep between calls but does not track total daily usage or approaching limits. During full
10-league analysis, the pipeline makes ~50–80 API calls. No warning if quota is close to
the daily limit.

**Fix:** Add a `_api_call_count` module-level counter. Log total calls at end of
`_run_analysis()`. Warn if count > 80 (approaching free-tier daily limit).

---

### DATA-05 · bets_log.csv context fields blank for older picks
**Impact: MEDIUM (for ML quality)**

Rows from before the context-field feature (`home_position`, `away_position`, `form_adv`,
`expected_total`) was added to the pipeline have blank values for these columns. When the LR
calibrator trains on these rows, it imputes them as 0.0 (the default), which is a misleading
value (0.0 home_position means bottom of table, not unknown).

**Fix:** Impute missing context fields with `None` (keep as missing). Add an `is_missing_*`
binary indicator column for each nullable feature. The LR handles `None` via indicator
variables, distinguishing unknown from zero.

---

## Summary Table

Legend: ✅ Done · ⏳ Blocked on data · ⬜ Deferred

| ID | Area | Title | Impact | Status |
|----|------|-------|--------|--------|
| BUG-01 | Logger | Dedup key on match_id only — secondary picks silently dropped | HIGH | ✅ 2026-04-29 |
| BUG-02 | Analyzer | xG threshold 3.5 almost never activates | HIGH | ✅ 2026-04-29 |
| BUG-03 | Analyzer | XG_CONV uniform across all leagues | MEDIUM | ✅ 2026-04-29 |
| BUG-04 | Main | Strength factors computed twice per fixture | LOW | ✅ Intentional — skipped |
| BUG-05 | Logger | Unrecognized combo leg silently logs "L" | MEDIUM | ✅ 2026-04-29 |
| BUG-06 | Markets | No-odds threshold inconsistency for Over 2.5 | LOW | ✅ 2026-04-29 |
| MISSING-01 | Pipeline | Isotonic correction not wired into live picks | HIGH | ⏳ Need 50+ bets/pick-type |
| MISSING-02 | Config | BASELINES never auto-refreshed | MEDIUM-HIGH | ⏳ Need 200+ bets |
| MISSING-03 | Odds | No fuzzy team name matching → fixtures miss live odds | HIGH | ✅ 2026-04-29 |
| MISSING-04 | ML | No train/test split for LR calibration | MEDIUM | ⏳ Need 200+ bets |
| MISSING-05 | ML | LR blend weights have no empirical basis | MEDIUM | ⏳ After MISSING-04 |
| MISSING-06 | Markets | Expected total gates not league-specific | MEDIUM | ✅ 2026-04-29 |
| MISSING-07 | Logger | No closing line value tracking | HIGH | ⬜ Needs pre-kickoff snapshot job |
| MODEL-01 | Model | Draw probability is a residual, not modelled | MEDIUM | ⬜ Deferred |
| MODEL-02 | Model | Form window (6 matches) not validated | MEDIUM | ⬜ Deferred |
| MODEL-03 | Model | No opponent difficulty weighting in form | MEDIUM | ⬜ Deferred |
| MODEL-04 | Model | CL group stage motivation not modelled | LOW-MEDIUM | ⬜ Deferred |
| MODEL-05 | Model | Referee factor ignores red card rate | LOW | ⬜ Deferred |
| POTENTIAL-01 | xG | Per-league XG_CONV from match history regression | HIGH | ✅ 2026-04-29 (script + defaults) |
| POTENTIAL-02 | Model | Form momentum (trend direction) signal | MEDIUM-HIGH | ✅ 2026-04-29 |
| POTENTIAL-03 | Odds | Asian handicap cross-check vs model | MEDIUM | ⬜ Deferred |
| POTENTIAL-04 | Model | Winter/weather under-scoring proxy | LOW-MEDIUM | ⬜ Deferred |
| POTENTIAL-05 | UX | Composite bet quality score for pick ranking | MEDIUM | ✅ 2026-04-29 |
| POTENTIAL-06 | Model | Streak detection (clean sheets, scoring droughts) | LOW-MEDIUM | ⬜ Deferred |
| ARCH-01 | Backtest | Backtest does not replay ML calibration | HIGH | ⬜ Deferred |
| ARCH-02 | Logger | CSV re-read on every call | LOW | ⬜ Deferred |
| ARCH-03 | Config | No startup validation of config values | LOW | ✅ 2026-04-29 |
| ARCH-04 | Analyzer | `compute_match_probabilities` is a 300-line monolith | LOW | ⬜ Deferred |
| ARCH-05 | Markets | No type-hint for probs dict keys | LOW | ⬜ Deferred |
| DATA-01 | xG | Shot data coverage very low; xG unused for 60%+ of fixtures | HIGH | ⬜ Deferred |
| DATA-02 | DB | `matches.db` has no pruning strategy | LOW-MEDIUM | ✅ 2026-04-29 |
| DATA-03 | Cache | Stale cache files accumulate indefinitely | LOW | ✅ 2026-04-29 |
| DATA-04 | API | No global API call budget tracker | MEDIUM | ✅ 2026-04-29 |
| DATA-05 | ML | Missing context fields imputed as 0.0 in LR training | MEDIUM | ⬜ Deferred |
