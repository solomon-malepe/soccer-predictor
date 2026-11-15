"""Night job (23:30 SAST): read /took replies, fetch Pinnacle closing odds for picks
that have kicked off, settle finished matches from football-data, post results.

    python -m jobs.settle
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
import pandas as pd

from predictor.backtest import settle as outcome_won
from predictor.db import picks_frame
from predictor.odds_feed import OddsPapi, OddsPapiError, closing_from_history
from predictor.paper import closing_clv
from predictor.telegram_bot import format_settled

from .common import load_history, process_took

MARKET_OF = {"H": "1X2", "D": "1X2", "A": "1X2", "O25": "OU25", "U25": "OU25"}


def fetch_closing(store, client, picks: pd.DataFrame, now: pd.Timestamp, log=print) -> int:
    """Closing odds + CLV for picks that have kicked off and don't have them yet."""
    n = 0
    for _, p in picks.iterrows():
        if p["status"] == "void" or pd.notna(p.get("closing_odds")) or not p.get("fixture_id"):
            continue
        ko = pd.Timestamp(p["kickoff"])
        if ko > now:
            continue
        try:
            close = closing_from_history(client.history(p["fixture_id"]), ko)
        except OddsPapiError as e:
            log(f"closing odds: {e}")
            break
        if not np.isfinite(close.get(p["outcome"], np.nan)):
            log(f"closing odds: none found for {p['home']} v {p['away']}")
            continue
        row = {"market": p["market"], "outcome": p["outcome"], "odds": p["odds"],
               **{f"close_{o}": v for o, v in close.items()}}
        clv = closing_clv(row)
        store.add_snapshots([{"match_id": p["match_id"], "book": "pinnacle", "market": MARKET_OF[o],
                              "outcome": o, "odds": v, "taken_at": ko.isoformat(), "is_closing": True}
                             for o, v in close.items() if np.isfinite(v)])
        store.update_pick(int(p["id"]), {"closing_odds": close[p["outcome"]],
                                         "clv": round(clv, 4) if np.isfinite(clv) else None})
        n += 1
    return n


def settle_results(store, picks: pd.DataFrame, results: pd.DataFrame, max_days: int = 4) -> pd.DataFrame:
    """Mark open picks won/lost from football-data scores. Returns the newly settled rows."""
    played = results.dropna(subset=["hg", "ag"]).copy()
    played["date"] = pd.to_datetime(played["date"])
    done = []
    for _, p in picks[picks["status"] == "open"].iterrows():
        day = pd.Timestamp(p["kickoff"]).tz_convert(None).normalize()
        m = played[(played["home"] == p["home"]) & (played["away"] == p["away"]) &
                   ((played["date"] - day).abs().dt.days <= max_days)]
        if m.empty:
            continue
        hg, ag = int(m.iloc[0]["hg"]), int(m.iloc[0]["ag"])
        won = outcome_won(p["outcome"], hg, ag)
        stake = float(p.get("stake_units") or 1)
        profit = round(stake * (float(p["odds"]) - 1), 3) if won else -stake
        store.set_result(int(p["match_id"]), hg, ag)
        store.update_pick(int(p["id"]), {"status": "won" if won else "lost", "profit_units": profit})
        done.append({**p.to_dict(), "status": "won" if won else "lost", "profit_units": profit,
                     "home_goals": hg, "away_goals": ag})
    return pd.DataFrame(done)


def run(args, store, bot=None, client=None, history=None, now=None, log=print) -> pd.DataFrame:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if bot is not None:
        process_took(store, bot, log=log)
    picks = picks_frame(store.picks(status="neq.void"))
    if picks.empty:
        log("No picks to settle.")
        return picks
    need_close = picks[picks["closing_odds"].isna()] if "closing_odds" in picks else picks
    if len(need_close):
        client = client or OddsPapi()
        n = fetch_closing(store, client, need_close, now, log=log)
        log(f"Closing odds for {n} pick(s). OddsPapi requests: {client.calls} (month: {client.month_usage()})")
        picks = picks_frame(store.picks(status="neq.void"))
    open_picks = picks[picks["status"] == "open"]
    if open_picks.empty:
        return pd.DataFrame()
    history = history or load_history(sorted(set(open_picks["league"])), history_seasons=1,
                                      refresh_current=not args.offline)
    results = pd.concat([cur for _, cur in history.values()], ignore_index=True)
    settled = settle_results(store, open_picks, results)
    if len(settled):
        fresh = picks_frame(store.picks(id=f"in.({','.join(str(int(i)) for i in settled['id'])})"))
        settled = settled.drop(columns=["clv"], errors="ignore").merge(
            fresh[["id", "clv"]], on="id", how="left")
        log(f"Settled {len(settled)} pick(s)")
        if bot is not None:
            bot.send(format_settled(settled))
    return settled


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="use cached football-data files")
    args = ap.parse_args(argv)
    from predictor.db import Store, Supabase
    from predictor.telegram_bot import Bot
    bot = Bot()
    store = Store(Supabase())
    try:
        run(args, store, bot=bot)
    except Exception:
        bot.alert("⚠️ settle failed:\n" + traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
