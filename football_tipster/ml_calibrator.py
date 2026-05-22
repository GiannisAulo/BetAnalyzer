"""
ml_calibrator.py — Historical performance-based pick calibrator.

Reads bets_log.csv to measure per-market success rates and calibrate
model probability estimates. Three tiers:

  Tier 1 — Segment stats: per-(market, pick, league) empirical win rate,
            applied with Bayesian smoothing when >= MIN_SEGMENT_SIZE samples.

  Tier 2 — Global logistic regression: trained over ALL labelled history when
            sklearn is installed and >= MIN_LR_SIZE completed bets exist.
            Recalibrates raw model_prob using (market, league, model_prob)
            as features.

  Tier 3 — Per-league logistic regression: one LR model per league, trained
            only on that league's history when >= MIN_LR_LEAGUE_SIZE bets
            exist for that league. Features: (market, model_prob).
            Activates before the global LR — a league-specific model captures
            structural differences (e.g. BL1 is higher-scoring than SA) that
            the global model dilutes by pooling all leagues together.

Fallback chain per pick:
  per-league LR  →  global LR  →  segment stats  →  identity

Integration contract
--------------------
Call `get_calibrator()` once to get the singleton CalibrationModel.
Then per pick:

    cal = get_calibrator()
    pick["model_prob"] = cal.calibrate(pick["model_prob"], market, league)
"""

import csv
import os
from collections import defaultdict

LOG_FILE          = "bets_log.csv"
MIN_SEGMENT_SIZE  = 100  # Frozen: Wilson CI half-width ~10pp at 100 samples — signal, not noise
MIN_LR_SIZE       = 200  # Frozen: global LR needs 200+ bets to generalise
MIN_LR_LEAGUE_SIZE = 75  # Frozen: per-league LR needs 75+ bets per league

# ── Result normalisation ────────────────────────────────────────────────────
_WIN_VALUES  = {"W", "WIN", "1", "YES"}
_LOSS_VALUES = {"L", "LOSS", "0", "NO"}


def _parse_result(raw):
    """Return 1 (win), 0 (loss), or None (incomplete/unknown)."""
    v = (raw or "").strip().upper()
    if v in _WIN_VALUES:
        return 1
    if v in _LOSS_VALUES:
        return 0
    return None


def _load_history():
    """
    Parse bets_log.csv into a list of completed-bet dicts.
    Skips rows where result is blank or unrecognised.
    """
    records = []
    if not os.path.exists(LOG_FILE):
        return records

    with open(LOG_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcome = _parse_result(row.get("result", ""))
            if outcome is None:
                continue
            try:
                model_prob = float(row.get("model_prob") or 0)
            except ValueError:
                continue

            def _opt_float(key):
                raw = (row.get(key) or "").strip()
                try:
                    return float(raw) if raw else None
                except ValueError:
                    return None

            records.append({
                "market":         row.get("market", "").strip(),
                "pick":           row.get("pick", "").strip(),
                "league":         row.get("league", "").strip(),
                "model_prob":     model_prob,
                "won":            outcome,
                "edge":           _opt_float("edge"),
                "home_position":  _opt_float("home_position"),
                "away_position":  _opt_float("away_position"),
                "form_adv":       _opt_float("form_adv"),
                "expected_total": _opt_float("expected_total"),
                "odds_taken":     _opt_float("odds_taken"),
            })

    return records


# ── Core model ──────────────────────────────────────────────────────────────

class CalibrationModel:
    """
    Two-tier calibration layer trained from bet history in bets_log.csv.
    Safe to construct even when the log is absent or empty — falls back
    to identity (returns inputs unchanged).
    """

    def __init__(self):
        # Segment stats keyed by (market, pick, league)
        self._seg:    dict[tuple, tuple] = {}   # (market, pick, league) -> (win_rate, n)
        self._global: dict[str,   tuple] = {}   # market                -> (win_rate, n)

        # Optional sklearn logistic regression — global (Tier 2)
        self._lr         = None
        self._scaler     = None
        self._m_enc: dict = {}
        self._l_enc: dict = {}

        # Per-league logistic regression — Tier 3
        # {league: {"lr": LR, "scaler": scaler, "m_enc": dict, "n": int}}
        self._league_lr: dict = {}

        self.sample_count = 0
        self._train()

    # ── Training ────────────────────────────────────────────────────────

    def _train(self):
        records = _load_history()
        self.sample_count = len(records)
        if not records:
            return

        seg   = defaultdict(lambda: [0, 0])   # [wins, total]
        glob  = defaultdict(lambda: [0, 0])

        for r in records:
            key = (r["market"], r["pick"], r["league"])
            seg[key][0]          += r["won"]
            seg[key][1]          += 1
            glob[r["market"]][0] += r["won"]
            glob[r["market"]][1] += 1

        self._seg = {
            k: (v[0] / v[1], v[1])
            for k, v in seg.items()
            if v[1] >= MIN_SEGMENT_SIZE
        }
        self._global = {
            m: (v[0] / v[1], v[1])
            for m, v in glob.items()
            if v[1] >= 5
        }

        # Tier 2 — global logistic regression
        if self.sample_count >= MIN_LR_SIZE:
            self._train_lr(records)

        # Tier 3 — per-league logistic regression
        self._train_league_lr(records)

    @staticmethod
    def _build_global_row(r, m_enc, l_enc):
        """Build one feature row for the global LR.
        Features: market_idx, league_idx, model_prob,
                  edge, home_position, away_position, form_adv, expected_total, odds_taken,
                  + 5 missing-indicator columns (one per nullable feature).
        Indicators distinguish "unknown" from the neutral imputation value, preventing
        old rows without context fields from being treated as "no edge" or "no form advantage".
        """
        e   = r.get("edge")
        hp  = r.get("home_position")
        ap  = r.get("away_position")
        fa  = r.get("form_adv")
        et  = r.get("expected_total")
        return [
            m_enc.get(r["market"], -1),
            l_enc.get(r["league"], -1),
            r["model_prob"],
            e  if e  is not None else 0.0,
            hp if hp is not None else 0.5,
            ap if ap is not None else 0.5,
            fa if fa is not None else 0.0,
            et if et is not None else 2.5,
            r["odds_taken"] if r.get("odds_taken") is not None else 0.0,
            # Missing indicators (1 = field was absent in the logged row)
            0 if e  is not None else 1,
            0 if hp is not None else 1,
            0 if ap is not None else 1,
            0 if fa is not None else 1,
            0 if et is not None else 1,
        ]

    def _train_lr(self, records):
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            markets = sorted({r["market"] for r in records})
            leagues = sorted({r["league"] for r in records})
            m_enc   = {m: i for i, m in enumerate(markets)}
            l_enc   = {l: i for i, l in enumerate(leagues)}

            X = np.array(
                [self._build_global_row(r, m_enc, l_enc) for r in records],
                dtype=float,
            )
            y = np.array([r["won"] for r in records])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
            lr.fit(X_scaled, y)

            self._lr      = lr
            self._scaler  = scaler
            self._m_enc   = m_enc
            self._l_enc   = l_enc
        except ImportError:
            pass   # sklearn not installed — tier 1 only

    def _train_league_lr(self, records):
        """
        Train one logistic regression per league using only that league's bets.
        Features: (market_index, model_prob) — no league feature needed since
        each model already isolates a single league.
        Activates per league when >= MIN_LR_LEAGUE_SIZE completed bets exist.
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return

        # Group records by league
        by_league: dict = {}
        for r in records:
            by_league.setdefault(r["league"], []).append(r)

        for league, recs in by_league.items():
            if len(recs) < MIN_LR_LEAGUE_SIZE:
                continue

            markets = sorted({r["market"] for r in recs})
            m_enc   = {m: i for i, m in enumerate(markets)}

            def _league_row(r):
                e  = r.get("edge")
                hp = r.get("home_position")
                ap = r.get("away_position")
                fa = r.get("form_adv")
                et = r.get("expected_total")
                return [
                    m_enc.get(r["market"], -1),
                    r["model_prob"],
                    e  if e  is not None else 0.0,
                    hp if hp is not None else 0.5,
                    ap if ap is not None else 0.5,
                    fa if fa is not None else 0.0,
                    et if et is not None else 2.5,
                    r["odds_taken"] if r.get("odds_taken") is not None else 0.0,
                    0 if e  is not None else 1,
                    0 if hp is not None else 1,
                    0 if ap is not None else 1,
                    0 if fa is not None else 1,
                    0 if et is not None else 1,
                ]
            X = np.array([_league_row(r) for r in recs], dtype=float)
            y = np.array([r["won"] for r in recs])

            # Need both classes present to fit
            if len(set(y)) < 2:
                continue

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
            lr.fit(X_scaled, y)

            self._league_lr[league] = {
                "lr":     lr,
                "scaler": scaler,
                "m_enc":  m_enc,
                "n":      len(recs),
            }

    # ── Public API ───────────────────────────────────────────────────────

    def calibrate(self, model_prob: float, market: str, league: str,
                  pick: str = "", extra: dict | None = None) -> float:
        """
        Return a calibrated probability.
        Blends model_prob with historical signal — never moves it
        more than ±15 pp to avoid extreme over-correction.
        """
        adjusted = self._league_lr_calibrate(model_prob, market, league, extra)
        if adjusted is None:
            adjusted = self._lr_calibrate(model_prob, market, league, extra)
        if adjusted is None:
            adjusted = self._segment_calibrate(model_prob, market, league, pick)

        # Hard clamp: never drift too far from raw model
        delta = adjusted - model_prob
        delta = max(-0.15, min(0.15, delta))
        return max(0.01, min(0.99, model_prob + delta))

    def summary(self) -> list[tuple]:
        """
        Return list of (market, win_rate, n) for display, sorted by n desc.
        Only markets with >= 5 samples included.
        """
        rows = [
            (market, rate, n)
            for market, (rate, n) in self._global.items()
        ]
        return sorted(rows, key=lambda x: x[2], reverse=True)

    def segment_uncertainty(self, market: str, pick: str, league: str):
        """
        Return (win_rate, n) for the (market, pick, league) segment, or None if
        the segment has fewer than MIN_SEGMENT_SIZE samples.
        """
        return self._seg.get((market, pick, league))

    @property
    def is_active(self) -> bool:
        """True if the model has any labelled history to work with."""
        return self.sample_count > 0

    @property
    def uses_lr(self) -> bool:
        return self._lr is not None

    @property
    def league_lr_leagues(self) -> list[str]:
        """Leagues that have a trained per-league LR model."""
        return sorted(self._league_lr.keys())

    def league_lr_sample_count(self, league: str) -> int:
        """Return number of training samples used for a league's LR model."""
        entry = self._league_lr.get(league)
        return entry["n"] if entry else 0

    # ── Private helpers ──────────────────────────────────────────────────

    def _league_lr_calibrate(self, model_prob, market, league, extra=None):
        """Return per-league LR-predicted probability, or None if unavailable."""
        entry = self._league_lr.get(league)
        if entry is None:
            return None
        ex = extra or {}
        try:
            import numpy as np
            e  = ex.get("edge")
            hp = ex.get("home_position")
            ap = ex.get("away_position")
            fa = ex.get("form_adv")
            et = ex.get("expected_total")
            xi = np.array([[
                entry["m_enc"].get(market, -1),
                model_prob,
                e  if e  is not None else 0.0,
                hp if hp is not None else 0.5,
                ap if ap is not None else 0.5,
                fa if fa is not None else 0.0,
                et if et is not None else 2.5,
                ex.get("odds_taken") if ex.get("odds_taken") is not None else 0.0,
                0 if e  is not None else 1,
                0 if hp is not None else 1,
                0 if ap is not None else 1,
                0 if fa is not None else 1,
                0 if et is not None else 1,
            ]], dtype=float)
            xi_scaled = entry["scaler"].transform(xi)
            lr_prob = float(entry["lr"].predict_proba(xi_scaled)[0][1])
            # Blend per-league LR with raw model (60/40) — league model is more
            # trusted than global LR since it's trained on matching data.
            return 0.60 * lr_prob + 0.40 * model_prob
        except Exception:
            return None

    def _lr_calibrate(self, model_prob, market, league, extra=None):
        """Return LR-predicted probability, or None if LR unavailable."""
        if self._lr is None:
            return None
        ex = extra or {}
        try:
            import numpy as np
            e  = ex.get("edge")
            hp = ex.get("home_position")
            ap = ex.get("away_position")
            fa = ex.get("form_adv")
            et = ex.get("expected_total")
            xi = np.array([[
                self._m_enc.get(market, -1),
                self._l_enc.get(league, -1),
                model_prob,
                e  if e  is not None else 0.0,
                hp if hp is not None else 0.5,
                ap if ap is not None else 0.5,
                fa if fa is not None else 0.0,
                et if et is not None else 2.5,
                ex.get("odds_taken") if ex.get("odds_taken") is not None else 0.0,
                0 if e  is not None else 1,
                0 if hp is not None else 1,
                0 if ap is not None else 1,
                0 if fa is not None else 1,
                0 if et is not None else 1,
            ]], dtype=float)
            xi_scaled = self._scaler.transform(xi)
            lr_prob = float(self._lr.predict_proba(xi_scaled)[0][1])
            # Blend LR with raw model (55/45) — preserve some model signal
            return 0.55 * lr_prob + 0.45 * model_prob
        except Exception:
            return None

    def _segment_calibrate(self, model_prob, market, league, pick=""):
        """Bayesian blend using segment or global win rate."""
        key = (market, pick, league)

        if key in self._seg:
            hist_rate, n = self._seg[key]
            alpha = min(0.35, n / 100.0)   # max 35% weight to history
            return (1 - alpha) * model_prob + alpha * hist_rate

        if market in self._global:
            hist_rate, n = self._global[market]
            alpha = min(0.20, n / 100.0)
            return (1 - alpha) * model_prob + alpha * hist_rate

        return model_prob   # no history — identity


def calibration_uncertainty(p: float, n: int) -> float:
    """
    Return the half-width (in probability points) of the 95% Wilson confidence
    interval for an observed proportion p over n trials.
    Returns 0.0 when n == 0.
    """
    if n == 0:
        return 0.0
    import math
    z = 1.96   # 95% confidence
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half   = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return half   # half-width in probability units


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: CalibrationModel | None = None


def get_calibrator() -> CalibrationModel:
    """Return the process-level singleton CalibrationModel (lazy init)."""
    global _instance
    if _instance is None:
        _instance = CalibrationModel()
    return _instance


def reset_calibrator():
    """Force re-initialisation (used in tests / after log updates)."""
    global _instance
    _instance = None
