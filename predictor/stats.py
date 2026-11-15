"""Track-record numbers for the dashboard (pure functions on a picks DataFrame).

Expects the columns produced by predictor.db.picks_frame: status, odds, profit_units,
stake_units, clv, kickoff, league, market, outcome, home, away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paper import GATE_PICKS


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def kpis(picks: pd.DataFrame) -> dict:
    live = picks[picks["status"] != "void"] if len(picks) else picks
    done = live[live["status"].isin(["won", "lost"])] if len(live) else live
    clv = _num(live["clv"]).dropna() if len(live) else pd.Series(dtype=float)
    units = float(_num(done["profit_units"]).sum()) if len(done) else 0.0
    staked = float(_num(done["stake_units"]).fillna(1).sum()) if len(done) else 0.0
    return {
        "picks": int(len(live)),
        "open": int((live["status"] == "open").sum()) if len(live) else 0,
        "settled": int(len(done)),
        "won": int((done["status"] == "won").sum()) if len(done) else 0,
        "hit_rate": float((done["status"] == "won").mean()) if len(done) else np.nan,
        "units": units,
        "roi": units / staked if staked else np.nan,
        "avg_clv": float(clv.mean()) if len(clv) else np.nan,
        "with_clv": int(len(clv)),
        "clv_t": float(clv.mean() / (clv.std() / np.sqrt(len(clv)))) if len(clv) > 2 and clv.std() > 0 else np.nan,
        "gate_progress": min(len(clv) / GATE_PICKS, 1.0),
    }


def profit_curve(picks: pd.DataFrame) -> pd.DataFrame:
    """Settled picks in kickoff order with running units."""
    done = picks[picks["status"].isin(["won", "lost"])].copy() if len(picks) else picks
    if done.empty:
        return pd.DataFrame(columns=["n", "kickoff", "match", "result", "profit", "cum_units"])
    done["kickoff"] = pd.to_datetime(done["kickoff"], utc=True)
    done = done.sort_values("kickoff")
    done["profit"] = _num(done["profit_units"])
    return pd.DataFrame({
        "n": range(1, len(done) + 1), "kickoff": done["kickoff"].array,
        "match": (done["home"] + " v " + done["away"]).values, "result": done["status"].values,
        "profit": done["profit"].values, "cum_units": done["profit"].cumsum().values,
    })


def clv_series(picks: pd.DataFrame) -> pd.DataFrame:
    """Picks with closing odds in kickoff order, CLV per pick and running average."""
    df = picks[_num(picks["clv"]).notna()].copy() if len(picks) else picks
    if df.empty:
        return pd.DataFrame(columns=["n", "kickoff", "match", "clv", "running_avg", "sign"])
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df = df.sort_values("kickoff")
    clv = _num(df["clv"])
    return pd.DataFrame({
        "n": range(1, len(df) + 1), "kickoff": df["kickoff"].array,
        "match": (df["home"] + " v " + df["away"]).values, "clv": clv.values,
        "running_avg": clv.expanding().mean().values,
        "sign": np.where(clv.values >= 0, "beat the close", "worse than close"),
    })


def breakdown(picks: pd.DataFrame, by: str) -> pd.DataFrame:
    """Picks, units, ROI and average CLV per league / market / outcome / book."""
    live = picks[picks["status"] != "void"] if len(picks) else picks
    if live.empty:
        return pd.DataFrame(columns=[by, "picks", "settled", "units", "roi", "avg_clv"])
    rows = []
    for key, g in live.groupby(by):
        k = kpis(g)
        rows.append({by: key, "picks": k["picks"], "settled": k["settled"], "units": k["units"],
                     "roi": k["roi"], "avg_clv": k["avg_clv"]})
    return pd.DataFrame(rows).sort_values("picks", ascending=False).reset_index(drop=True)
