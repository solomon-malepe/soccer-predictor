import numpy as np
import pandas as pd
import pytest

from predictor.paper import (GATE_PICKS, add_picks, empty_log, gate_status, read_log, report,
                             settle_log)


def _picks():
    return pd.DataFrame([
        dict(league="E0", date="2026-11-22", home="Arsenal", away="Chelsea", market="OU25",
             outcome="U25", odds=2.60, book="Betway", p_model=0.50, p_pin=0.41,
             edge_model=0.30, edge_pin=0.07, p_used=0.5, edge=0.3, disagreement=0.09, qualifies=True),
        dict(league="D1", date="2026-11-22", home="Bayern Munich", away="Dortmund", market="1X2",
             outcome="D", odds=4.50, book="SportyBet", p_model=0.25, p_pin=0.23,
             edge_model=0.125, edge_pin=0.035, p_used=0.25, edge=0.125, disagreement=0.02, qualifies=True),
    ])


def _results():
    return pd.DataFrame([
        dict(date=pd.Timestamp("2026-11-22"), home="Arsenal", away="Chelsea", hg=1, ag=0),
        dict(date=pd.Timestamp("2026-11-22"), home="Bayern Munich", away="Dortmund", hg=3, ag=1),
    ])


def test_add_skips_matches_already_logged():
    log, added, skipped = add_picks(empty_log(), _picks())
    assert (added, skipped) == (2, 0)
    assert (log["status"] == "open").all() and (log["stake"] == 1).all()
    changed = _picks().assign(outcome="O25", odds=1.80)  # re-run gives a different pick
    log2, added, skipped = add_picks(log, changed)
    assert (added, skipped) == (0, 2)
    assert list(log2["outcome"]) == ["U25", "D"]          # first logged pick is kept


def test_settle_results_profit_and_clv(tmp_path):
    log, _, _ = add_picks(empty_log(), _picks())
    log.loc[0, ["close_O25", "close_U25"]] = [1.95, 1.95]   # fair 50/50 at close
    log, n = settle_log(log, _results())
    assert n == 2
    a, b = log.iloc[0], log.iloc[1]
    assert (a["status"], a["profit"]) == ("won", pytest.approx(1.60))
    assert (b["status"], b["profit"]) == ("lost", -1.0)
    assert a["clv"] == pytest.approx(2.60 * 0.5 - 1, abs=1e-4)      # +30%
    assert np.isnan(b["clv"])                              # closing odds not entered yet

    # round-trip through CSV, add closing odds later, settle again: CLV appears, result unchanged
    path = tmp_path / "log.csv"
    log.to_csv(path, index=False)
    log = read_log(path)
    log.loc[1, ["close_H", "close_D", "close_A"]] = [1.50, 4.00, 6.00]
    log, n = settle_log(log, _results())
    assert n == 0
    inv = np.array([1 / 1.5, 1 / 4.0, 1 / 6.0])
    assert log.loc[1, "clv"] == pytest.approx(4.50 * (inv[1] / inv.sum()) - 1, abs=1e-4)
    assert log.loc[1, "status"] == "lost"


def test_void_and_unplayed_matches():
    log, _, _ = add_picks(empty_log(), _picks())
    log.loc[1, "status"] = "void"
    log, n = settle_log(log, _results().iloc[:0])            # no results yet
    assert n == 0 and log.loc[0, "status"] == "open"
    assert log.loc[1, "profit"] == 0.0
    assert report(log)["overall"]["picks"] == 1               # void picks don't count


def test_gate_status():
    row = lambda n, clv: pd.Series({"with_clv": n, "avg_clv": clv})
    assert gate_status(row(10, 0.02)).startswith(f"Gate: 10/{GATE_PICKS}")
    assert "PASSED" in gate_status(row(GATE_PICKS, 0.01))
    assert "FAILED" in gate_status(row(GATE_PICKS + 5, -0.01))
