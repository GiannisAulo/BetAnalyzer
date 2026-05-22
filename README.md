# BetAnalyzer

A football match prediction and betting tipster tool powered by a statistical model built on the Dixon-Coles score matrix, ML calibration, and live odds comparison.

## What it does

- Fetches fixtures and team data from **football-data.org**
- Fetches live bookmaker odds from **the-odds-api.com**
- Computes match probabilities using a Dixon-Coles Poisson model with:
  - Recency-weighted team form (home/away split)
  - H2H history with venue filtering
  - Fatigue, motivation, and referee tendency factors
  - Knockout stage suppression for CL/cup matches
  - 3-tier ML calibration (segment stats → global LR → per-league LR)
- Compares model probabilities against market odds to find value bets
- Generates a daily coupon with picks, edge %, and confidence ratings
- Logs settled bets to `bets_log.csv` for ongoing calibration

## Supported Leagues

| Code | League |
|------|--------|
| PL | English Premier League |
| PD | Spanish La Liga |
| BL1 | German Bundesliga |
| SA | Italian Serie A |
| FL1 | French Ligue 1 |
| CL | UEFA Champions League |
| PPL | Portuguese Primeira Liga |
| DED | Dutch Eredivisie |
| ELC | English Championship |
| BSA | Brazilian Série A |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/GiannisAulo/BetAnalyzer.git
cd BetAnalyzer/football_tipster
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file inside `football_tipster/`:

```env
API_KEY=your_football_data_org_key
ODDS_API_KEY=your_the_odds_api_key
```

- Get `API_KEY` from [football-data.org](https://www.football-data.org)
- Get `ODDS_API_KEY` from [the-odds-api.com](https://the-odds-api.com)

`ODDS_API_KEY` is optional — the tool runs without it but won't show live odds.

## Usage

### Interactive mode (recommended)

```bash
python main.py
```

### Scripted / automated

```bash
python main.py --leagues PL BL1 SA      # specific leagues
python main.py --min-edge 3             # lower confidence threshold
python main.py --no-cache               # force fresh API calls
python main.py --output coupon.txt      # save coupon to file
python main.py --over25                 # Over 2.5 table only
```

## Running tests

```bash
pytest tests/
```

## Project structure

```
football_tipster/
├── main.py              # Entry point
├── analyzer.py          # Dixon-Coles model & probability engine
├── markets.py           # Market evaluation & value bet detection
├── fetcher.py           # football-data.org API client
├── odds_fetcher.py      # the-odds-api.com client
├── match_store.py       # SQLite match history store
├── ml_calibrator.py     # 3-tier ML probability calibration
├── coupon.py            # Coupon rendering
├── logger.py            # Bet logging to bets_log.csv
├── cache.py             # API response caching
├── config.py            # League baselines & model constants
├── warn_log.py          # Warning logger
├── scripts/
│   ├── calibrate_baselines.py     # Empirical baseline calibration
│   ├── calibration_curves.py      # Calibration curve plots
│   ├── fit_xg_conv.py             # xG conversion fitting
│   └── validate_knockout_factor.py
└── tests/               # pytest test suite
```

## Contributing

All changes to `master` require a Pull Request. Claude automatically reviews every PR — see `.github/workflows/claude-review.yml`.

## License

Private repository — all rights reserved.
