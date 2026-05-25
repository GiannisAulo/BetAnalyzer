"""
drift.py — Per-market WR vs predicted-probability drift detector.

For each (market, pick) bucket in the current model cohort, compute:
  - actual win rate over the last N settled bets
  - average model_prob over the same bets
  - gap = predicted - actual (positive = model overconfident)

Severity bands:
  green:  |gap| < 10pp           — within normal noise
  yellow: 10pp <= |gap| < 15pp   — possible drift, watch
  red:    |gap| >= 15pp          — meaningful drift, recommend calibration

Markets with fewer than _MIN_SAMPLE settled bets are not evaluated (sample
too small to distinguish drift from noise).
"""

import csv
import os
from collections import defaultdict

import logger as _logger   # for LOG_FILE constant and MODEL_VERSION

_MIN_SAMPLE = 30                 # never alert below this — noise dominates
_YELLOW_GAP = 10.0               # percentage points (model − actual)
_RED_GAP    = 15.0


def _settled_v_rows(model_version):
    """Return v-cohort settled rows with model_prob parsable."""
    path = _logger.LOG_FILE
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("model_version") or "").strip() != model_version:
                continue
            result = (r.get("result") or "").strip().upper()
            if result not in ("W", "L"):
                continue
            try:
                mp = float(r.get("model_prob") or 0)
            except ValueError:
                continue
            out.append({
                "market":     (r.get("market") or "").strip(),
                "pick":       (r.get("pick") or "").strip(),
                "model_prob": mp,
                "won":        1 if result == "W" else 0,
            })
    return out


def severity_for(gap_pp):
    """Return 'green' | 'yellow' | 'red' for a signed gap in percentage points."""
    g = abs(gap_pp)
    if g >= _RED_GAP:
        return "red"
    if g >= _YELLOW_GAP:
        return "yellow"
    return "green"


def compute_drift(model_version=None, min_sample=None):
    """Compute drift per (market, pick) for the active cohort.

    Returns a list of dicts sorted by severity (red first, then yellow) and
    absolute gap descending. Empty list when nothing qualifies.

    Each entry:
        {
            "market":   str,
            "pick":     str,
            "n":        int,
            "actual_wr_pct": float,  # 0..100
            "predicted_pct": float,  # 0..100 — mean model_prob
            "gap_pp":   float,       # predicted − actual (positive = overconfident)
            "severity": "green" | "yellow" | "red",
        }
    """
    if model_version is None:
        model_version = _logger.MODEL_VERSION
    if min_sample is None:
        min_sample = _MIN_SAMPLE

    rows = _settled_v_rows(model_version)
    if not rows:
        return []

    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["market"], r["pick"])].append(r)

    out = []
    for (market, pick), bucket in buckets.items():
        n = len(bucket)
        if n < min_sample:
            continue
        actual   = sum(b["won"] for b in bucket) / n
        predicted = sum(b["model_prob"] for b in bucket) / n
        gap_pp   = (predicted - actual) * 100
        out.append({
            "market":         market,
            "pick":           pick,
            "n":              n,
            "actual_wr_pct":  actual * 100,
            "predicted_pct":  predicted * 100,
            "gap_pp":         gap_pp,
            "severity":       severity_for(gap_pp),
        })

    # Sort: red > yellow > green; within tier, larger absolute gap first
    _SEV_ORDER = {"red": 0, "yellow": 1, "green": 2}
    out.sort(key=lambda d: (_SEV_ORDER[d["severity"]], -abs(d["gap_pp"])))
    return out


def has_actionable(drift_rows):
    """True when at least one row is yellow or red."""
    return any(d["severity"] in ("yellow", "red") for d in drift_rows)
