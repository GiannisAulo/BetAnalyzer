# ⚽ Football Betting Tipster — Project Blueprint

> **Stack**: Python 3.10+ · football-data.org v4 API (free tier) · Rich terminal UI  
> **Goal**: A terminal script that fetches live data across all major European leagues, performs multi-metric statistical analysis, and outputs a formatted betting coupon with value-detected picks across 1X2, Over/Under, BTTS, Double Chance, and combo markets.

---

## 1. API Constraints & Design Decisions

### football-data.org Free Tier

| Constraint | Value | Impact on Design |
|---|---|---|
| Rate limit | 10 requests / minute | Must throttle all calls with `time.sleep(6)` |
| Leagues (free) | PL, PD, BL1, SA, FL1, CL, PPL, DED, BSA | All major targets are included |
| xG data | ❌ Not available at any tier | Must derive goal expectancy from match history |
| Head-to-head | ✅ `/matches/{id}/head2head` | Available — use last 10 meetings |
| Team match history | ✅ `/teams/{id}/matches` | Key for rolling averages |
| Standings + form | ✅ `/competitions/{code}/standings` | Includes `form`, `goalsFor`, `goalsAgainst` |
| Scorers / assists | ❌ Paid only | Not available — skip for free tier |

### API Key
```
X-Auth-Token: 4fffa16f33814693addd003ad9e2829b
Base URL: https://api.football-data.org/v4
```

### Competition Codes
```
PL  — Premier League (England)
PD  — La Liga (Spain)
BL1 — Bundesliga (Germany)
SA  — Serie A (Italy)
FL1 — Ligue 1 (France)
CL  — UEFA Champions League
PPL — Primeira Liga (Portugal)
DED — Eredivisie (Netherlands)
```

---

## 2. Project Structure

```
football_tipster/
│
├── main.py              # CLI entry point — orchestrates the full pipeline
├── config.py            # API key, league codes, league baselines, constants
├── fetcher.py           # All API calls to football-data.org (with rate limiting)
├── analyzer.py          # Core stats engine: form, averages, probability model
├── markets.py           # Market evaluators: 1X2, O/U, BTTS, DC, combo markets
├── coupon.py            # Rich-formatted terminal coupon output
├── logger.py            # Saves bet history to CSV for tracking
├── cache.py             # Local JSON cache to reduce API calls during reruns
│
├── .env                 # API key (never commit)
├── requirements.txt     # requests, rich, python-dotenv
└── bets_log.csv         # Auto-generated — tracks historical picks & outcomes
```

---

## 3. Data Collection Pipeline

For each upcoming matchday, the script executes these API calls **in order**:

### Step 1 — Get upcoming fixtures per league
```
GET /v4/competitions/{code}/matches?status=SCHEDULED&matchday={N}
```
Returns all scheduled matches for the next matchday. Extract: `matchId`, `homeTeam.id`, `awayTeam.id`, `utcDate`.

### Step 2 — Get standings (once per league)
```
GET /v4/competitions/{code}/standings
```
Extracts for each team: `form` (last 5: W/D/L string), `goalsFor`, `goalsAgainst`, `playedGames`, `position`, `points`.

Derived stats from standings:
- **Avg goals scored** = `goalsFor / playedGames`
- **Avg goals conceded** = `goalsAgainst / playedGames`
- **Form score** = weighted W/D/L over last 5 (W=3, D=1, L=0)

### Step 3 — Get team match history (per team, last 10)
```
GET /v4/teams/{id}/matches?status=FINISHED&limit=10
```
For each match: extract score, home/away context, compute:
- Goals scored at home vs away (split averages)
- Goals conceded at home vs away (split averages)
- Clean sheet rate (home and away separately)
- BTTS rate (did both teams score in this match?)
- Over 2.5 rate (total goals > 2.5?)
- Over/Under 3.5 rate

> ⚡ **Rate limit note**: For a 10-fixture matchday you need ~1 (standings) + 20 (team histories) + 10 (H2H) = **~31 calls per league**. At 10/min that's ~4 minutes. Cache standings and team history per session.

### Step 4 — Head-to-head per fixture
```
GET /v4/matches/{matchId}/head2head?limit=10
```
Extract: historical win/draw/loss counts, avg goals in H2H, BTTS rate in H2H.

> ⚠️ Only use H2H as a modifier if ≥5 meetings exist. Fewer than that is noise.

---

## 4. Statistics Computed per Fixture

| Stat | Home Team | Away Team | Source |
|---|---|---|---|
| Avg goals scored | home split | away split | team match history |
| Avg goals conceded | home split | away split | team match history |
| Clean sheet rate | home % | away % | team match history |
| BTTS rate | home % | away % | team match history |
| Over 2.5 rate | home % | away % | team match history |
| Form score (weighted) | last 5 | last 5 | standings |
| League position | ✅ | ✅ | standings |
| Season avg goals scored | ✅ | ✅ | standings |
| H2H win rate | ✅ | ✅ | head2head |
| H2H avg goals | ✅ | — | head2head |
| H2H BTTS rate | ✅ | — | head2head |

---

## 5. Probability Model

### 5.1 — League Baselines (Home / Draw / Away)

```python
BASELINES = {
    "PL":  {"home": 0.46, "draw": 0.25, "away": 0.29},
    "PD":  {"home": 0.46, "draw": 0.26, "away": 0.28},
    "BL1": {"home": 0.44, "draw": 0.24, "away": 0.32},  # higher away rate
    "SA":  {"home": 0.44, "draw": 0.30, "away": 0.26},  # more draws
    "FL1": {"home": 0.45, "draw": 0.28, "away": 0.27},  # more variance
    "CL":  {"home": 0.43, "draw": 0.27, "away": 0.30},  # neutral venue effect
}
```

### 5.2 — Form Adjustment

```python
# Weights: [most recent → oldest]
FORM_WEIGHTS = [0.30, 0.22, 0.18, 0.13, 0.10, 0.07]
# Max score = 9.0 (all wins). Normalize to 0–1.

form_adv = home_form_score - away_form_score  # range: -1.0 to +1.0

# Adjustments:
if form_adv > 0.30:   home_prob += 0.05; away_prob -= 0.03; draw_prob -= 0.02
if form_adv < -0.30:  away_prob += 0.05; home_prob -= 0.03; draw_prob -= 0.02
```

### 5.3 — Goals Model (for Over/Under & BTTS)

Expected total goals for the match:
```python
expected_home_goals = (home_avg_scored_at_home + away_avg_conceded_away) / 2
expected_away_goals = (away_avg_scored_away + home_avg_conceded_home) / 2
expected_total = expected_home_goals + expected_away_goals
```

Use Poisson distribution to compute:
- `P(Over 2.5)` = 1 - P(0 goals) - P(1 goal) - P(2 goals)
- `P(BTTS Yes)` = P(home scores ≥1) × P(away scores ≥1)
- `P(Under 3.5)` = P(total ≤ 3)

```python
import math

def poisson_prob(lam, k):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def prob_over(lam_total, threshold=2.5):
    return 1 - sum(poisson_prob(lam_total, k) for k in range(int(threshold)+1))
```

### 5.4 — H2H Modifier (conditional)

If ≥ 5 H2H meetings:
- H2H home win rate significantly above/below baseline → apply ±3–5% nudge
- H2H avg goals consistently above/below 2.5 → reinforce Over/Under call

---

## 6. Market Evaluators

### 6.1 — 1X2 (Match Result)
- Input: adjusted home/draw/away probabilities
- Bet when edge ≥ 5% vs bookmaker implied probability
- Rating: ⭐⭐⭐ if edge ≥ 8%, ⭐⭐ if ≥ 5%, ⭐ if ≥ 3%

### 6.2 — Double Chance (1X, X2, 12)
- Combine two outcomes: `P(1X) = P(home) + P(draw)`
- Lower odds but higher certainty — value requires ≥ 3% edge
- Best for moderate confidence picks where draw risk is real

### 6.3 — Over/Under 2.5 Goals
- Input: Poisson model output from expected goals
- Reinforce with: both teams' BTTS rate, H2H avg goals, clean sheet rates
- Bet when model probability ≥ bookmaker implied + 4%

### 6.4 — BTTS (Both Teams To Score)
- `P(BTTS Yes)` from Poisson (see §5.3)
- Reinforce with: rolling BTTS rate from last 10 matches for each team
- BTTS No: stronger when either team has clean sheet rate > 40%

### 6.5 — Combo Markets (Auto-generated)
- Evaluate cross-market combos: `1X + Over 2.5`, `X2 + Under 3.5`, `Home + BTTS`, etc.
- Only recommend when BOTH legs have independent edge (not correlated false confidence)
- Use combined probability: `P(combo) = P(leg1) × P(leg2)` ← valid only if legs are weakly correlated

---

## 7. Output — Betting Coupon (Terminal)

```
╔══════════════════════════════════════════════════════════════╗
║   ⚽  BETTING COUPON  —  Sunday 13 April 2026 — 6 Leagues   ║
╠══════════════════════════════════════════════════════════════╣
║ Match:   Arsenal vs Chelsea         [Premier League]         ║
║ Bet:     Home Win                                            ║
║ Odds:    1.90   Model: 58%   Implied: 52%   Edge: +6%        ║
║ Rating:  ⭐⭐                                                 ║
║ Stake:   €18.50  (Quarter-Kelly on €500 bankroll)            ║
║ Reason:  Arsenal avg 2.1 goals/home game, Chelsea conceding  ║
║          1.6/away. Form: Arsenal WWDWW vs Chelsea LDWDL.     ║
╠══════════════════════════════════════════════════════════════╣
║ ... (repeat per pick, max 8 singles) ...                     ║
╠══════════════════════════════════════════════════════════════╣
║ ACCUMULATOR (top 3 picks ≥ ⭐⭐)                             ║
║ Legs: Arsenal HW + Bayern Over 2.5 + Real Madrid DC          ║
║ Combined odds: 7.42   Stake: €5.00 (1% of bankroll)         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 8. Stake Sizing (Quarter-Kelly)

```python
def kelly_stake(prob, odds, bankroll, fraction=0.25):
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0
    return round(min(kelly * fraction * bankroll, bankroll * 0.05), 2)
```

- Max single bet: **5% of bankroll** regardless of edge
- Accumulator pool: **separate 1–2% of bankroll**
- Track every pick in `bets_log.csv`

---

## 9. Caching Strategy

To avoid burning rate limit quota on reruns:

```python
# cache.py — saves API responses as JSON files
CACHE_DIR = ".cache/"
CACHE_TTL = 3600  # 1 hour (standings + fixtures change rarely intra-day)

def get_cached(key):
    path = CACHE_DIR + key + ".json"
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_TTL:
            return json.load(open(path))
    return None

def set_cache(key, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(data, open(CACHE_DIR + key + ".json", "w"))
```

---

## 10. Roadmap — Future Improvements

### Phase 1 — MVP (current scope)
- [x] Fetch fixtures, standings, team history, H2H via football-data.org
- [x] Compute form, rolling goal averages, clean sheet rate, BTTS rate, Over/Under rate
- [x] Poisson probability model for goals markets
- [x] 1X2 + Double Chance + Over/Under + BTTS + combo markets
- [x] Quarter-Kelly stake sizing
- [x] Rich terminal coupon output
- [x] JSON/CSV cache for rate limit protection
- [x] Bet history logger

### Phase 2 — Accuracy Improvements
- [ ] **Integrate Understat for xG data** (free, scraping-based)
  - Real xG per match replaces simple goals averages — the single biggest accuracy upgrade
  - Use `understat` Python package: `pip install understat`
  - Map team names between football-data and Understat (different naming conventions)
- [ ] **Injury & suspension feed**
  - football-data.org does not provide this — use BBC Sport scraping or a secondary API
  - Flag matches where a key striker/GK is absent — adjust attacking/defensive probability by ~10%
- [ ] **Referee tendencies**
  - Some referees have significantly higher or lower card/foul rates — affects Over/Under slightly
- [ ] **Venue / distance factor for CL**
  - For Champions League away legs, factor in travel distance and rest days

### Phase 3 — Odds Integration
- [ ] **Connect to The Odds API** (free tier: 500 req/month)
  - Fetch live bookmaker odds instead of relying on user-supplied odds
  - Enables fully automated value detection without manual input
  - `https://the-odds-api.com`
- [ ] **Line movement tracking**
  - Store odds at T-48h, T-24h, T-2h — sharp money moves the line; follow smart money signals
- [ ] **Closing line value (CLV) tracking**
  - Best long-term ROI predictor: did your price beat the closing line?

### Phase 4 — ML Model (advanced)
- [ ] **Logistic regression baseline**
  - Features: home/away avg goals, form score, H2H win rate, xG if available
  - Train on 2–3 seasons of historical match data (downloadable from football-data.org paid, or from open datasets like football-data.co.uk CSVs)
- [ ] **Poisson Dixon-Coles correction**
  - Adds correlation adjustment for low-scoring matches (0-0, 1-0, 0-1 are correlated)
  - Standard improvement over naive independent Poisson model
- [ ] **ELO rating system**
  - Persistent rating per team updated after every result — stronger than form string alone
  - Open-source implementations available (ClubElo data is freely downloadable)

### Phase 5 — UX & Automation
- [ ] **Scheduled daily runs** via cron / Task Scheduler
  - Auto-run at 08:00 on matchdays, save coupon to file
- [ ] **Telegram bot integration**
  - Push coupon to a private Telegram channel automatically
  - `python-telegram-bot` library
- [ ] **Web dashboard** (optional)
  - Streamlit app wrapping the same logic with interactive league/market filtering

---

## 11. Dependencies

```
# requirements.txt
requests>=2.31.0
rich>=13.0.0
python-dotenv>=1.0.0
```

Optional (future phases):
```
understat>=1.5.0        # xG data
python-telegram-bot     # Telegram push
streamlit               # Web dashboard
scipy                   # Poisson CDF (faster than manual math.factorial)
```

Install:
```bash
pip install -r requirements.txt
```

---

## 12. Running the Script

```bash
# All leagues, next matchday, default bankroll €1000
python main.py

# Specific leagues only
python main.py --leagues PL BL1 SA

# Custom bankroll and lower edge threshold
python main.py --bankroll 500 --min-edge 3

# Force refresh (bypass cache)
python main.py --no-cache

# Save coupon to file
python main.py --output coupon_2026-04-13.txt
```

---

*⚠️ Disclaimer: This tool is for research and entertainment purposes only. No statistical model guarantees profit. Always bet responsibly and never wager more than you can afford to lose.*
