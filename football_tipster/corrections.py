"""
corrections.py — Isotonic probability corrections per (market, pick).

When the drift detector finds that the model is systematically over- or
under-confident for a given (market, pick), this module fits a monotone
correction from raw model_prob → observed win rate and persists it to
data/isotonic_corrections.json. The ml_calibrator loads the corrections at
startup and applies them as Tier 0 before any logistic-regression calibration.

The correction is stored as two parallel arrays:
    raw_probs   — sorted unique raw model_prob breakpoints
    calibrated  — isotonic-fitted win-rate at each breakpoint

At runtime, apply_correction() does a linear interpolation between adjacent
breakpoints (clamped at the endpoints). This is small, fast, and dependency-
free at apply time (sklearn only needed for fitting).

Backwards compatibility: when the JSON file is missing or contains no entry
for a (market, pick), apply_correction() returns the input unchanged.
"""

import csv
import json
import os
from datetime import datetime, timezone

import logger as _logger

# Where corrections live. Created on first save.
_DATA_DIR  = "data"
_STORE_FILE = os.path.join(_DATA_DIR, "isotonic_corrections.json")

# Minimum settled bets required before we'll fit. Below this the regression
# overfits noise. Matches the drift detector floor.
_MIN_FIT_SAMPLE = 30


# ── In-process cache ────────────────────────────────────────────────────────
# Loaded once per process via _load_store(); _STORE_CACHE is invalidated when
# save_correction() writes a new entry.

_STORE_CACHE = None


def _load_store():
    """Return the corrections dict from disk; cached after first load."""
    global _STORE_CACHE
    if _STORE_CACHE is not None:
        return _STORE_CACHE
    if not os.path.exists(_STORE_FILE):
        _STORE_CACHE = {}
        return _STORE_CACHE
    try:
        with open(_STORE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            _STORE_CACHE = {}
        else:
            _STORE_CACHE = data
    except (json.JSONDecodeError, OSError):
        _STORE_CACHE = {}
    return _STORE_CACHE


def invalidate_cache():
    """Drop the in-process cache. Useful in tests; also called from save."""
    global _STORE_CACHE
    _STORE_CACHE = None


def _key(market, pick):
    """Composite key under which a correction is stored."""
    return f"{market}||{pick}"


# ── Fit / save ──────────────────────────────────────────────────────────────

def _load_bucket(market, pick, model_version):
    """Return (model_probs, won_flags) for one (market, pick) in a v-cohort."""
    path = _logger.LOG_FILE
    if not os.path.exists(path):
        return [], []
    probs, wons = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("model_version") or "").strip() != model_version:
                continue
            if (r.get("market") or "").strip() != market:
                continue
            if (r.get("pick") or "").strip() != pick:
                continue
            result = (r.get("result") or "").strip().upper()
            if result not in ("W", "L"):
                continue
            try:
                p = float(r.get("model_prob") or 0)
            except ValueError:
                continue
            probs.append(p)
            wons.append(1 if result == "W" else 0)
    return probs, wons


def fit_correction(market, pick, model_version=None):
    """Fit an isotonic correction from the v-cohort settled bets for (market, pick).

    Returns the correction entry dict on success, or None if:
      - fewer than _MIN_FIT_SAMPLE settled bets exist
      - sklearn is unavailable
      - the fit degenerates (e.g. all wins or all losses)

    The returned entry is NOT saved automatically — call save_correction() to persist.
    """
    if model_version is None:
        model_version = _logger.MODEL_VERSION

    probs, wons = _load_bucket(market, pick, model_version)
    if len(probs) < _MIN_FIT_SAMPLE:
        return None

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    try:
        iso.fit(probs, wons)
    except Exception:
        return None

    # Take a sweep of breakpoints across the observed range for compact storage
    p_min, p_max = min(probs), max(probs)
    if p_max - p_min < 1e-6:
        return None
    n_points = 11
    xs = [p_min + (p_max - p_min) * i / (n_points - 1) for i in range(n_points)]
    ys = [float(iso.predict([x])[0]) for x in xs]

    return {
        "market":         market,
        "pick":           pick,
        "model_version":  model_version,
        "fitted_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples":      len(probs),
        "raw_probs":      xs,
        "calibrated":     ys,
    }


def save_correction(entry):
    """Persist a correction entry produced by fit_correction()."""
    if not entry:
        return
    store = _load_store()
    # Mutate a copy in case _load_store returned the cached dict
    store = dict(store)
    key = _key(entry["market"], entry["pick"])
    store[key] = entry

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
    invalidate_cache()


def delete_correction(market, pick):
    """Remove a stored correction if present."""
    store = _load_store()
    key = _key(market, pick)
    if key not in store:
        return
    store = dict(store)
    del store[key]
    with open(_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
    invalidate_cache()


# ── Apply ───────────────────────────────────────────────────────────────────

def apply_correction(prob, market, pick):
    """Apply the stored correction to a raw model probability.

    Returns the input unchanged when no correction exists for (market, pick),
    when the input is out of range, or when the stored entry is malformed.
    """
    if prob is None or prob < 0 or prob > 1:
        return prob
    store = _load_store()
    entry = store.get(_key(market, pick))
    if not entry:
        return prob
    xs = entry.get("raw_probs")
    ys = entry.get("calibrated")
    if not xs or not ys or len(xs) != len(ys):
        return prob

    # Clamp at endpoints
    if prob <= xs[0]:
        return float(ys[0])
    if prob >= xs[-1]:
        return float(ys[-1])

    # Linear interpolation
    for i in range(len(xs) - 1):
        if xs[i] <= prob <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span <= 0:
                return float(ys[i])
            t = (prob - xs[i]) / span
            return float(ys[i] + t * (ys[i + 1] - ys[i]))
    return prob   # unreachable, defensive


def list_corrections():
    """Return summary info for all stored corrections."""
    store = _load_store()
    return [
        {
            "market":    e.get("market"),
            "pick":      e.get("pick"),
            "n_samples": e.get("n_samples"),
            "fitted_at": e.get("fitted_at"),
        }
        for e in store.values()
        if isinstance(e, dict)
    ]
