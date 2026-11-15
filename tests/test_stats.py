import numpy as np
import pandas as pd
import pytest

from predictor.stats import breakdown, clv_series, kpis, profit_curve


def _picks():
    return pd.DataFrame([
        dict(status="won", odds=2.75, profit_units=1.75, stake_units=1, clv=0.05, league="E0", outcome="U25",
             book="Betway", kickoff="2026-11-21T15:00:00Z", home="A", away="B"),
        dict(status="lost", odds=4.0, profit_units=-1.0, stake_units=1, clv=-0.02, league="SP1", outcome="D",
             book="SportyBet", kickoff="2026-11-20T20:00:00Z", home="C", away="D"),
        dict(status="open", odds=3.1, profit_units=None, stake_units=1, clv=0.03, league="E0", outcome="H",
             book="Betway", kickoff="2026-11-22T15:00:00Z", home="E", away="F"),
        dict(status="void", odds=2.0, profit_units=0.0, stake_units=1, clv=None, league="E0", outcome="A",
             book="Betway", kickoff="2026-11-19T15:00:00Z", home="G", away="H"),
    ])


def test_kpis_ignore_voids_and_count_clv_on_open_picks():
    k = kpis(_picks())
    assert (k["picks"], k["open"], k["settled"], k["won"]) == (3, 1, 2, 1)
    assert k["units"] == pytest.approx(0.75) and k["roi"] == pytest.approx(0.375)
    assert k["avg_clv"] == pytest.approx(0.02) and k["with_clv"] == 3
    assert k["gate_progress"] == pytest.approx(3 / 150)


def test_empty_record():
    empty = _picks().iloc[:0]
    k = kpis(empty)
    assert k["picks"] == 0 and np.isnan(k["avg_clv"]) and k["gate_progress"] == 0
    assert profit_curve(empty).empty and clv_series(empty).empty and breakdown(empty, "league").empty


def test_curves_are_in_kickoff_order():
    curve = profit_curve(_picks())
    assert list(curve["match"]) == ["C v D", "A v B"]               # 20 Nov before 21 Nov
    assert list(curve["cum_units"]) == pytest.approx([-1.0, 0.75])
    clv = clv_series(_picks())
    assert list(clv["clv"]) == pytest.approx([-0.02, 0.05, 0.03])
    assert clv["running_avg"].iloc[-1] == pytest.approx(0.02)
    assert list(clv["sign"]) == ["worse than close", "beat the close", "beat the close"]


def test_breakdown_by_league():
    bd = breakdown(_picks(), "league").set_index("league")
    assert bd.loc["E0", "picks"] == 2 and bd.loc["E0", "units"] == pytest.approx(1.75)
    assert bd.loc["SP1", "avg_clv"] == pytest.approx(-0.02)
