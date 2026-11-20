"""Experimental track: BTTS and Over/Under 1.5 / 3.5, scored apart from the main record."""

import json

import numpy as np
import pandas as pd
import pytest

from jobs.common import CFG, CFG_EXP, cfg_for, log_pick
from predictor.backtest import settle as outcome_won
from predictor.db import DBError, Store, Supabase
from predictor.odds_feed import extract_prices
from predictor.paper import closing_clv
from predictor.targets import all_targets, match_targets
from predictor.telegram_bot import format_targets, format_weekly_summary
from predictor.value import MARKET_OF, is_experimental

from tests.test_automation import FakeStore


def test_tracks_and_market_lookup():
    assert cfg_for("BTTS") is CFG_EXP and cfg_for("OU35") is CFG_EXP and cfg_for("1X2") is CFG
    assert CFG.markets == ["1X2", "OU25"]                       # the locked rule is untouched
    assert CFG_EXP.min_edge == CFG.min_edge and CFG_EXP.min_edge_pin == CFG.min_edge_pin
    assert MARKET_OF["BTTS_N"] == "BTTS" and MARKET_OF["U15"] == "OU15" and MARKET_OF["H"] == "1X2"
    assert is_experimental("OU15") and not is_experimental("OU25")


@pytest.mark.parametrize("outcome,hg,ag,won", [
    ("BTTS_Y", 1, 1, True), ("BTTS_Y", 2, 0, False), ("BTTS_N", 0, 0, True), ("BTTS_N", 3, 1, False),
    ("O15", 1, 1, True), ("O15", 1, 0, False), ("U15", 0, 1, True), ("U15", 2, 0, False),
    ("O35", 2, 2, True), ("O35", 3, 0, False), ("U35", 2, 1, True), ("U35", 4, 0, False),
])
def test_new_outcomes_settle(outcome, hg, ag, won):
    assert outcome_won(outcome, hg, ag) is won


def test_feed_reads_new_markets():
    p = lambda x: {"players": {"0": {"price": x, "active": True}}}
    markets = {"104": {"outcomes": {"104": p(1.80), "105": p(2.05)}},
               "108": {"outcomes": {"108": p(1.28), "109": p(3.70)}},
               "1012": {"outcomes": {"1012": p(2.60), "1013": p(1.50)}}}
    got = extract_prices(markets)
    assert (got["BTTS_Y"], got["BTTS_N"], got["O15"], got["U15"], got["O35"], got["U35"]) == \
        (1.80, 2.05, 1.28, 3.70, 2.60, 1.50)
    assert np.isnan(got["H"])


def test_experimental_targets_only_from_experimental_config():
    row = {"pin_H": 1.769, "pin_D": 4.13, "pin_A": 4.31, "pin_O25": 1.526, "pin_U25": 2.59,
           "pin_BTTS_Y": 1.80, "pin_BTTS_N": 2.05}
    model = {"H": 0.536, "D": 0.230, "A": 0.234, "O25": 0.629, "U25": 0.371,
             "BTTS_Y": 0.50, "BTTS_N": 0.50}
    main = match_targets(row, model, CFG)
    exp = match_targets(row, model, CFG_EXP)
    assert all(t["market"] in ("1X2", "OU25") for t in main)
    assert [t["outcome"] for t in exp] == ["BTTS_N"]             # model 50% vs Pinnacle fair ~46.7%
    assert exp[0]["target_odds"] <= 2.05 * 1.10
    df = all_targets(pd.DataFrame([row]), {0: model}, CFG_EXP, per_match=1)
    assert list(df["market"]) == ["BTTS"]


def test_message_marks_experimental_lines():
    t = pd.DataFrame([
        {"id": 5, "league": "E0", "kickoff": "2026-11-21T15:00:00Z", "home": "Chelsea", "away": "Leeds",
         "market": "OU25", "outcome": "U25", "target_odds": 2.13, "pin_odds": 2.00, "over_pinnacle": .06},
        {"id": 6, "league": "E0", "kickoff": "2026-11-21T15:00:00Z", "home": "Chelsea", "away": "Leeds",
         "market": "BTTS", "outcome": "BTTS_N", "target_odds": 2.10, "pin_odds": 2.02, "over_pinnacle": .04},
    ])
    msg = format_targets(t, now="2026-11-21T06:00:00Z")
    assert "1 targets + 1 🧪 · 1 matches" in msg
    assert "<code>  #6    BTTS No   ≥  2.10   Pin 2.02</code> 🧪" in msg
    assert msg.index("#5 ") < msg.index("#6 ")                      # main line first
    assert "🧪 <i>= experimental" in msg
    assert "🧪" not in format_targets(t.iloc[:1], now="2026-11-21T06:00:00Z")


def _store_with_two_targets():
    store = FakeStore()
    ids = store.team_ids({"E0": ["Chelsea", "Leeds"]})
    mid = store.upsert_matches([{"league": "E0", "season": "2627", "kickoff": "2026-11-21T15:00:00Z",
                                 "home_id": ids["Chelsea"], "away_id": ids["Leeds"], "oddspapi_id": "f1"}])["f1"]
    store.upsert_targets([
        {"match_id": mid, "prediction_id": None, "market": "OU25", "outcome": "U25", "target_odds": 2.13,
         "pin_odds": 2.0, "p_model": 0.50, "p_pinnacle": 0.48},
        {"match_id": mid, "prediction_id": None, "market": "BTTS", "outcome": "BTTS_N", "target_odds": 2.10,
         "pin_odds": 2.02, "p_model": 0.50, "p_pinnacle": 0.467},
    ])
    return store


def test_experimental_pick_is_its_own_track():
    store = _store_with_two_targets()
    when = "2026-11-21T10:00:00Z"
    pick, msg = log_pick(store, 2, 2.25, "SportyBet", "sol", when)   # target 1.02/0.467 = 2.18
    assert pick["strategy"] == "experimental" and "🧪 experimental" in msg
    pick, msg = log_pick(store, 1, 2.20, "Betway", "sol", when)        # main pick on the same match
    assert pick["strategy"] == "both" and "🧪" not in msg
    again, msg = log_pick(store, 2, 2.30, "Betway", "tk", when)
    assert again is None and "experimental 🧪 pick" in msg


def test_closing_clv_for_btts():
    row = {"market": "BTTS", "outcome": "BTTS_N", "odds": 2.15, "close_BTTS_Y": 1.85, "close_BTTS_N": 2.00}
    fair_n = (1 / 2.00) / (1 / 1.85 + 1 / 2.00)
    assert closing_clv(row) == pytest.approx(2.15 * fair_n - 1)


def test_add_pick_falls_back_before_migration():
    calls = []

    def fake(method, url, headers, body):
        calls.append(url)
        if "on_conflict=match_id%2Cstrategy" in url or "on_conflict=match_id,strategy" in url:
            return 400, {"code": "42P10", "message": "there is no unique or exclusion constraint matching"}
        return 201, [{"id": 9}]

    store = Store(Supabase(url="https://x", key="k", fetch=fake))
    assert store.add_pick({"match_id": 1, "strategy": "both"}) == {"id": 9}
    assert len(calls) == 2 and calls[-1].endswith("on_conflict=match_id")

    other = Store(Supabase(url="https://x", key="k", fetch=lambda *a: (500, {"message": "down"})))
    with pytest.raises(DBError):
        other.add_pick({"match_id": 1, "strategy": "both"})


def test_weekly_summary_shows_experimental_apart():
    main = pd.DataFrame([{"status": "won", "profit_units": 1.1, "clv": 0.02, "kickoff": "2026-11-21"}])
    exp = pd.DataFrame([{"status": "lost", "profit_units": -1.0, "clv": -0.01, "kickoff": "2026-11-21"}])
    msg = format_weekly_summary(main, main, "gate text", exp)
    assert "🧪 Experimental, all time" in msg
    assert msg.index("All time") < msg.index("🧪 Experimental")
    assert "🧪" not in format_weekly_summary(main, main, "gate text")
