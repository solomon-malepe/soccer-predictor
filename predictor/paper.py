"""Paper-trading log: record picks, settle them from results, measure CLV.

The log is a plain CSV (paper_log.csv) you can open in VS Code / Excel. The only
columns you fill in by hand are the Pinnacle closing odds for the pick's market:

    1X2 picks:      close_H, close_D, close_A
    Over/Under 2.5: close_O25, close_U25

All prices in the market are needed because the margin is removed before CLV is
worked out (same method as the backtest). You can also correct `odds` if the price
you actually got differs from the one predict.py showed, or set `status` to `void`
for a postponed match.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .backtest import settle as outcome_won
from .value import MARKET_GROUPS, devig

CLOSE_COLS = {o: f"close_{o}" for o in ["H", "D", "A", "O25", "U25"]}

LOG_COLUMNS = [
    "logged_at", "league", "date", "home", "away", "market", "outcome", "book", "odds",
    "p_model", "p_pin", "edge_model", "edge_pin", "stake",
    *CLOSE_COLS.values(),
    "hg", "ag", "status", "profit", "clv",
]

GATE_PICKS = 150   # Phase 2 gate: 150+ paper picks with average CLV > 0


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def read_log(path) -> pd.DataFrame:
    try:
        log = pd.read_csv(path)
    except FileNotFoundError:
        return empty_log()
    for c in LOG_COLUMNS:
        if c not in log:
            log[c] = np.nan
    log["status"] = log["status"].fillna("open").astype(str).str.strip().str.lower()
    return log[LOG_COLUMNS + [c for c in log.columns if c not in LOG_COLUMNS]]


def _key(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df["league"].astype(str) + "|" + d + "|" + df["home"].astype(str) + "|" + df["away"].astype(str)


def add_picks(log: pd.DataFrame, picks: pd.DataFrame,
              now: dt.datetime | None = None) -> tuple[pd.DataFrame, int, int]:
    """Append new picks. A match already in the log is skipped (one pick per match;
    the first one logged is the one that was posted). Returns (log, added, skipped)."""
    if picks.empty:
        return log, 0, 0
    now = now or dt.datetime.now(dt.timezone.utc)
    new = picks.copy()
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    seen = set(_key(log)) if len(log) else set()
    keys = _key(new)
    fresh = new[~keys.isin(seen)].drop_duplicates(subset=["league", "date", "home", "away"])
    rows = pd.DataFrame({c: fresh[c] if c in fresh else np.nan for c in LOG_COLUMNS})
    rows["logged_at"] = now.strftime("%Y-%m-%d %H:%M UTC")
    rows["stake"] = 1.0
    rows["status"] = "open"
    out = pd.concat([log, rows], ignore_index=True) if len(log) else rows.reset_index(drop=True)
    return out, len(fresh), len(new) - len(fresh)


def closing_clv(row) -> float:
    """CLV against Pinnacle's de-vigged closing price; NaN until all closing odds are in."""
    group = MARKET_GROUPS.get(row["market"])
    if group is None or row["outcome"] not in group:
        return np.nan
    close = [pd.to_numeric(row.get(CLOSE_COLS[o]), errors="coerce") for o in group]
    fair = dict(zip(group, devig(close)))
    p = fair[row["outcome"]]
    odds = pd.to_numeric(row["odds"], errors="coerce")
    return float(odds * p - 1) if np.isfinite(p) and np.isfinite(odds) else np.nan


def settle_log(log: pd.DataFrame, results: pd.DataFrame, max_days: int = 4) -> tuple[pd.DataFrame, int]:
    """Fill scores, status and profit for open picks from played matches, and
    (re)compute CLV for every pick with closing odds. Returns (log, newly_settled)."""
    log = log.copy()
    played = results.dropna(subset=["hg", "ag"]).copy()
    played["date"] = pd.to_datetime(played["date"])
    settled = 0
    for i, r in log.iterrows():
        if r["status"] == "open" and len(played):
            m = played[(played["home"] == r["home"]) & (played["away"] == r["away"]) &
                       ((played["date"] - pd.Timestamp(r["date"])).abs().dt.days <= max_days)]
            if len(m):
                hg, ag = int(m.iloc[0]["hg"]), int(m.iloc[0]["ag"])
                won = outcome_won(r["outcome"], hg, ag)
                stake = float(r["stake"]) if pd.notna(r["stake"]) else 1.0
                log.at[i, "hg"], log.at[i, "ag"] = hg, ag
                log.at[i, "status"] = "won" if won else "lost"
                log.at[i, "profit"] = stake * (float(r["odds"]) - 1) if won else -stake
                settled += 1
        if log.at[i, "status"] == "void":
            log.at[i, "profit"] = 0.0
        log.at[i, "clv"] = closing_clv(log.loc[i])
    log["profit"] = pd.to_numeric(log["profit"], errors="coerce").round(3)
    log["clv"] = pd.to_numeric(log["clv"], errors="coerce").round(4)
    for c in ("hg", "ag"):
        log[c] = pd.to_numeric(log[c], errors="coerce").astype("Int64")
    return log, settled


def _stats(g: pd.DataFrame) -> pd.Series:
    done = g[g["status"].isin(["won", "lost"])]
    clv = pd.to_numeric(g["clv"], errors="coerce").dropna()
    staked = pd.to_numeric(done["stake"], errors="coerce").fillna(1).sum()
    return pd.Series({
        "picks": len(g),
        "settled": len(done),
        "hit_rate": (done["status"] == "won").mean() if len(done) else np.nan,
        "units": pd.to_numeric(done["profit"], errors="coerce").sum(),
        "roi": pd.to_numeric(done["profit"], errors="coerce").sum() / staked if staked else np.nan,
        "with_clv": len(clv),
        "avg_clv": clv.mean() if len(clv) else np.nan,
        "clv_t": clv.mean() / (clv.std() / np.sqrt(len(clv))) if len(clv) > 2 and clv.std() > 0 else np.nan,
    })


def report(log: pd.DataFrame) -> dict:
    live = log[log["status"] != "void"]
    out = {"overall": _stats(live)}
    if len(live):
        out["by_league"] = live.groupby("league").apply(_stats, include_groups=False)
        out["by_outcome"] = live.groupby("outcome").apply(_stats, include_groups=False)
    return out


def gate_status(overall: pd.Series) -> str:
    n, clv = int(overall["with_clv"]), overall["avg_clv"]
    if n < GATE_PICKS:
        trend = "" if not np.isfinite(clv) else f" Average CLV so far {clv:+.2%}."
        return f"Gate: {n}/{GATE_PICKS} picks with closing odds. Keep paper trading.{trend}"
    if clv > 0:
        return f"Gate PASSED: {n} picks, average CLV {clv:+.2%}. Real money can start at small stakes."
    return f"Gate FAILED: {n} picks, average CLV {clv:+.2%}. Back to Phase 1 before any real money."
