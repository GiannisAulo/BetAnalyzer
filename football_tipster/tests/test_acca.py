"""
test_acca.py — Tests for cross-fixture accumulator construction and settlement.
"""
import csv
import pytest

import markets
import logger
from logger import FIELDS, _load_rows, settle_bets


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finished(home_goals, away_goals):
    if home_goals > away_goals:
        winner = "HOME_TEAM"
    elif away_goals > home_goals:
        winner = "AWAY_TEAM"
    else:
        winner = "DRAW"
    return {
        "status": "FINISHED",
        "score": {"winner": winner, "fullTime": {"home": home_goals, "away": away_goals}},
    }


def _postponed():
    return {"status": "POSTPONED", "score": {"fullTime": {"home": None, "away": None}}}


def _scheduled():
    return {"status": "SCHEDULED", "score": {"fullTime": {"home": None, "away": None}}}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({field: r.get(field, "") for field in FIELDS})


def _acca_row(match_id, pick, odds, prob=0.7, result="", attempts="0"):
    return {
        "match_id": match_id, "home": "Multi", "away": "(acca)",
        "league": "MULTI", "market": "AccaCross", "pick": pick,
        "model_prob": f"{prob:.3f}", "odds_taken": f"{odds:.2f}",
        "edge": "10.0", "result": result, "settle_attempts": attempts,
    }


# ── build_cross_fixture_accas ─────────────────────────────────────────────────

class TestBuildAccas:
    def _candidate(self, pick="Home Win", market="1X2", odds=1.25, prob=0.85):
        return {
            "market": market, "pick": pick, "model_prob": prob,
            "implied_prob": 1 / odds, "odds": odds, "edge": (prob - 1 / odds) * 100,
        }

    def _fixture(self, mid, league, candidates):
        return {
            "match_id": mid, "home_name": f"H{mid}", "away_name": f"A{mid}",
            "league": league, "acca_candidates": candidates,
        }

    def test_no_candidates_returns_empty(self):
        fxs = [self._fixture("1", "PL", []), self._fixture("2", "BL1", [])]
        assert markets.build_cross_fixture_accas(fxs) == []

    def test_single_candidate_returns_empty(self):
        fxs = [self._fixture("1", "PL", [self._candidate()])]
        assert markets.build_cross_fixture_accas(fxs) == []

    def test_two_legs_qualifying_builds_2leg_acca(self):
        # 1.25 * 1.40 = 1.75 ∈ [1.60, 2.50] ✓
        # 0.85 * 0.80 = 0.68 ≥ 0.55 ✓ (verified threshold)
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.25, prob=0.85), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.40, prob=0.80), "inferred_odds": False}]),
        ]
        accas = markets.build_cross_fixture_accas(fxs)
        assert len(accas) == 1
        a = accas[0]
        assert a["size"] == 2
        assert abs(a["joint_odds"] - 1.75) < 1e-9
        assert abs(a["joint_prob"] - 0.68) < 1e-9
        assert a["edge"] > 8.0
        assert a["verified_edge"] is True

    def test_three_legs_same_league_rejected(self):
        # All three legs in PL — independence safeguard must reject
        fxs = [
            self._fixture("1", "PL", [{**self._candidate(odds=1.20, prob=0.85), "inferred_odds": False}]),
            self._fixture("2", "PL", [{**self._candidate(odds=1.25, prob=0.85), "inferred_odds": False}]),
            self._fixture("3", "PL", [{**self._candidate(odds=1.30, prob=0.85), "inferred_odds": False}]),
        ]
        assert markets.build_cross_fixture_accas(fxs) == []

    def test_combined_odds_below_min_rejected(self):
        # 1.18 * 1.18 = 1.39 < 1.60 floor
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.18, prob=0.88), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.18, prob=0.88), "inferred_odds": False}]),
        ]
        assert markets.build_cross_fixture_accas(fxs) == []

    def test_inferred_odds_acca_built(self):
        # Two legs without real bookmaker odds — should still build an acca
        # but marked as not verified (informational only).
        # Fair odds 1.30 × 1.30 = 1.69 ∈ [1.60, 2.50], joint prob 0.77² = 0.593 ≥ 0.45
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.30, prob=0.77), "inferred_odds": True}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.30, prob=0.77), "inferred_odds": True}]),
        ]
        accas = markets.build_cross_fixture_accas(fxs)
        assert len(accas) == 1
        assert accas[0]["verified_edge"] is False
        assert accas[0]["legs"][0]["inferred_odds"] is True

    def test_mixed_real_and_inferred_marked_unverified(self):
        # One real, one inferred → acca exists but verified_edge=False
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.30, prob=0.85), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.30, prob=0.77), "inferred_odds": True}]),
        ]
        accas = markets.build_cross_fixture_accas(fxs)
        assert len(accas) == 1
        assert accas[0]["verified_edge"] is False

    def test_combined_odds_above_max_rejected(self):
        # All pair/triple combinations exceed the 2.50 combined-odds cap:
        # 1.80 * 1.80 = 3.24 (pair too high) and 1.80^3 = 5.83 (triple too high)
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.80, prob=0.95), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.80, prob=0.95), "inferred_odds": False}]),
            self._fixture("3", "SA",  [{**self._candidate(odds=1.80, prob=0.95), "inferred_odds": False}]),
        ]
        assert markets.build_cross_fixture_accas(fxs) == []

    def test_low_joint_prob_rejected_for_verified(self):
        # 0.72 * 0.72 = 0.518 < 0.55 verified floor; should NOT appear as verified.
        # (Could still appear as inferred if the legs were inferred — but here they're real.)
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.30, prob=0.72), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.40, prob=0.72), "inferred_odds": False}]),
        ]
        accas = markets.build_cross_fixture_accas(fxs)
        # No verified accas (low joint prob); no inferred either (legs aren't inferred).
        assert accas == []

    def test_overlap_dedup(self):
        # Two accas with same first leg — only the highest-edge one kept
        fxs = [
            self._fixture("1", "PL",  [{**self._candidate(odds=1.25, prob=0.88), "inferred_odds": False}]),
            self._fixture("2", "BL1", [{**self._candidate(odds=1.40, prob=0.80), "inferred_odds": False}]),
            self._fixture("3", "SA",  [{**self._candidate(odds=1.35, prob=0.82), "inferred_odds": False}]),
        ]
        accas = markets.build_cross_fixture_accas(fxs)
        # All three are mutually compatible — should get 1 acca (3-leg uses all matches,
        # blocking any 2-leg from being added after). Either way, no leg can appear twice.
        seen = set()
        for a in accas:
            for leg in a["legs"]:
                assert leg["match_id"] not in seen
                seen.add(leg["match_id"])


# ── collect_acca_candidates ───────────────────────────────────────────────────

class TestCollectCandidates:
    def _probs(self, home=0.85, draw=0.10, away=0.05, o15=0.92, u15=0.08,
              o25=0.78, u25=0.22, o35=0.55, u35=0.45):
        # Minimal probs dict that the evaluators consume
        return {
            "home": home, "draw": draw, "away": away,
            "over_1_5": o15, "under_1_5": u15,
            "over_2_5": o25, "under_2_5": u25,
            "over_3_5": o35, "under_3_5": u35,
            "btts_yes": 0.5, "btts_no": 0.5,
            "expected_total": 2.7,
        }

    def test_picks_short_odds_home_favourite(self):
        # 1.20 odds, 85% prob → qualifies (edge = 0.85 - 0.833 ≈ +1.7pp)
        probs = self._probs(home=0.85)
        fx_odds = {"home_odds": 1.20, "draw_odds": 7.0, "away_odds": 15.0,
                   "over_1.5": 1.15, "under_1.5": 6.0, "over_2.5": 1.50, "under_2.5": 2.60,
                   "over_3.5": 2.20, "under_3.5": 1.65}
        cands = markets.collect_acca_candidates(probs, fx_odds, league="PL")
        names = [c["pick"] for c in cands]
        assert "Home Win" in names

    def test_skips_when_no_positive_edge(self):
        # 85% prob but 1.10 odds → implied 0.909 > 0.85 → no edge
        probs = self._probs(home=0.85)
        fx_odds = {"home_odds": 1.10, "draw_odds": 8.0, "away_odds": 18.0,
                   "over_1.5": 1.50, "under_1.5": 2.50, "over_2.5": 1.50, "under_2.5": 2.60,
                   "over_3.5": 2.20, "under_3.5": 1.65}
        cands = markets.collect_acca_candidates(probs, fx_odds, league="PL")
        assert all(c["pick"] != "Home Win" for c in cands)

    def test_skips_when_prob_too_low(self):
        probs = self._probs(home=0.70)   # below _ACCA_LEG_PROB_MIN (0.75)
        fx_odds = {"home_odds": 1.30, "draw_odds": 5.0, "away_odds": 8.0,
                   "over_1.5": 1.50, "under_1.5": 2.50, "over_2.5": 1.50, "under_2.5": 2.60,
                   "over_3.5": 2.20, "under_3.5": 1.65}
        cands = markets.collect_acca_candidates(probs, fx_odds, league="PL")
        assert all(c["pick"] != "Home Win" for c in cands)


# ── Acca settlement ──────────────────────────────────────────────────────────

class TestSettleAcca:
    @pytest.fixture(autouse=True)
    def cd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_all_legs_win(self, tmp_path, monkeypatch):
        row = _acca_row(
            match_id="ACC:100+200",
            pick="Bayern: Home Win @1.25 + Real Madrid: Home Win @1.40",
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        data = {"100": _finished(2, 0), "200": _finished(3, 0)}
        monkeypatch.setattr(fetcher, "get_match", lambda mid: data[mid])
        settled, failed = settle_bets()
        assert settled == 1 and failed == 0
        rows = _load_rows()
        assert rows[0]["result"] == "W"
        assert float(rows[0]["roi"]) == pytest.approx(0.75, abs=0.001)

    def test_one_leg_loses_is_loss(self, tmp_path, monkeypatch):
        row = _acca_row(
            match_id="ACC:100+200",
            pick="Bayern: Home Win @1.25 + Real Madrid: Home Win @1.40",
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        data = {"100": _finished(2, 0), "200": _finished(0, 2)}   # leg 2 lost
        monkeypatch.setattr(fetcher, "get_match", lambda mid: data[mid])
        settle_bets()
        rows = _load_rows()
        assert rows[0]["result"] == "L"
        assert rows[0]["roi"] == "-1.000"

    def test_postponed_leg_skipped(self, tmp_path, monkeypatch):
        # 3-leg acca; middle leg postponed → settle on 1.25 * 1.30 = 1.625 payout
        row = _acca_row(
            match_id="ACC:100+200+300",
            pick="A: Home Win @1.25 + B: Home Win @1.40 + C: Home Win @1.30",
            odds=2.275,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        data = {"100": _finished(2, 0), "200": _postponed(), "300": _finished(1, 0)}
        monkeypatch.setattr(fetcher, "get_match", lambda mid: data[mid])
        settle_bets()
        rows = _load_rows()
        assert rows[0]["result"] == "W"
        # ROI = (1.25 * 1.30) - 1 = 0.625
        assert float(rows[0]["roi"]) == pytest.approx(0.625, abs=0.001)

    def test_all_legs_postponed_is_void(self, tmp_path, monkeypatch):
        row = _acca_row(
            match_id="ACC:100+200",
            pick="A: Home Win @1.25 + B: Home Win @1.40",
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: _postponed())
        settle_bets()
        rows = _load_rows()
        assert rows[0]["result"] == "VOID"
        assert rows[0]["roi"] == ""

    def test_pending_leg_keeps_acca_pending(self, tmp_path, monkeypatch):
        row = _acca_row(
            match_id="ACC:100+200",
            pick="A: Home Win @1.25 + B: Home Win @1.40",
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        data = {"100": _finished(2, 0), "200": _scheduled()}
        monkeypatch.setattr(fetcher, "get_match", lambda mid: data[mid])
        settled, failed = settle_bets()
        assert settled == 0
        rows = _load_rows()
        assert rows[0]["result"] == ""
        assert int(rows[0]["settle_attempts"]) == 1

    def test_api_error_keeps_pending(self, tmp_path, monkeypatch):
        row = _acca_row(
            match_id="ACC:100+200",
            pick="A: Home Win @1.25 + B: Home Win @1.40",
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        def boom(mid):
            raise Exception("network down")
        monkeypatch.setattr(fetcher, "get_match", boom)
        settled, failed = settle_bets()
        assert settled == 0 and failed == 1
        rows = _load_rows()
        assert rows[0]["result"] == ""
        assert int(rows[0]["settle_attempts"]) == 1

    def test_malformed_pick_keeps_pending(self, tmp_path, monkeypatch):
        # Pick string has wrong number of legs vs match_id
        row = _acca_row(
            match_id="ACC:100+200+300",
            pick="A: Home Win @1.25 + B: Home Win @1.40",   # only 2 legs in pick
            odds=1.75,
        )
        _write_csv(tmp_path / "bets_log.csv", [row])
        import fetcher
        monkeypatch.setattr(fetcher, "get_match", lambda mid: _finished(2, 0))
        settled, failed = settle_bets()
        assert settled == 0 and failed == 1

    def test_existing_single_pick_path_still_works(self, tmp_path, monkeypatch):
        # A non-acca row alongside an acca row — single path must still settle.
        single = {
            "match_id": "555", "home": "X", "away": "Y", "league": "PL",
            "market": "1X2", "pick": "Home Win", "model_prob": "0.60",
            "odds_taken": "2.00", "edge": "5.0", "result": "", "settle_attempts": "0",
        }
        acca = _acca_row("ACC:100+200",
                         "A: Home Win @1.25 + B: Home Win @1.40", odds=1.75)
        _write_csv(tmp_path / "bets_log.csv", [single, acca])
        import fetcher
        data = {"555": _finished(2, 0), "100": _finished(1, 0), "200": _finished(2, 1)}
        monkeypatch.setattr(fetcher, "get_match", lambda mid: data[mid])
        settled, failed = settle_bets()
        assert settled == 2
        rows = _load_rows()
        results = {r["match_id"]: r["result"] for r in rows}
        assert results["555"] == "W"
        assert results["ACC:100+200"] == "W"
