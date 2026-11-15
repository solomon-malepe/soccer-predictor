"""Morning job (08:00 SAST): read /took replies, fit the model, fetch Pinnacle odds,
work out target prices for the next 24 hours of matches, save, post to Telegram.
On Mondays it also posts the weekly summary.

    python -m jobs.daily_picks             # the real thing (needs .env / secrets)
    python -m jobs.daily_picks --dry-run   # print the message, touch no database or chat
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from predictor.model import DixonColes, market_probs
from predictor.odds_feed import MARKET_IDS, OddsPapi, live_fixtures
from predictor.paper import gate_status, report
from predictor.targets import all_targets
from predictor.telegram_bot import format_targets, format_weekly_summary

from .common import (ALL_LEAGUES, CFG, MODEL_VERSION, load_history, process_took, team_names)

OUTCOMES = list(MARKET_IDS)
MARKET_OF = {"H": "1X2", "D": "1X2", "A": "1X2", "O25": "OU25", "U25": "OU25"}


def model_probs(history: dict, fixtures: pd.DataFrame, min_matches: int = 5, log=print):
    """Fit one model per league; return ({fixture index: probs}, {fixture index: (xg_h, xg_a)})."""
    probs, xg = {}, {}
    for lg, (hist, cur) in history.items():
        fx = fixtures[fixtures["league"] == lg] if len(fixtures) else fixtures
        if fx.empty:
            continue
        model = DixonColes().fit(pd.concat([hist, cur], ignore_index=True))
        for idx, r in fx.iterrows():
            if not (model.knows(r["home"], min_matches) and model.knows(r["away"], min_matches)):
                log(f"  skip {r['home']} v {r['away']}: not enough matches for the model yet")
                continue
            probs[idx] = market_probs(model.score_matrix(r["home"], r["away"]))
            xg[idx] = model.expected_goals(r["home"], r["away"])
    return probs, xg


def save(store, fixtures: pd.DataFrame, probs: dict, xg: dict, targets: pd.DataFrame,
         season: str) -> pd.DataFrame:
    """Write matches, Pinnacle snapshots, predictions and targets. Returns targets with DB ids."""
    priced = fixtures.loc[list(probs)]
    ids = store.team_ids({lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in priced.groupby("league")})
    match_ids = store.upsert_matches([
        {"league": r["league"], "season": season, "kickoff": r["kickoff"], "home_id": ids[r["home"]],
         "away_id": ids[r["away"]], "oddspapi_id": r["fixture_id"]} for _, r in priced.iterrows()])
    snaps, preds = [], []
    for idx, r in priced.iterrows():
        mid = match_ids[r["fixture_id"]]
        snaps += [{"match_id": mid, "book": "pinnacle", "market": MARKET_OF[o], "outcome": o,
                   "odds": r[f"pin_{o}"]} for o in OUTCOMES if pd.notna(r[f"pin_{o}"])]
        preds.append({"match_id": mid, "model_version": MODEL_VERSION,
                      "probs": {k: round(float(v), 4) for k, v in probs[idx].items()},
                      "xg_home": round(float(xg[idx][0]), 2), "xg_away": round(float(xg[idx][1]), 2)})
    store.add_snapshots(snaps)
    pred_ids = store.add_predictions(preds)
    if targets.empty:
        return targets.assign(id=pd.Series(dtype=int))
    rows = []
    for _, t in targets.iterrows():
        mid = match_ids[t["fixture_id"]]
        rows.append({"match_id": mid, "prediction_id": pred_ids.get(mid), "market": t["market"],
                     "outcome": t["outcome"], "target_odds": t["target_odds"], "pin_odds": t["pin_odds"],
                     "p_model": round(t["p_model"], 4), "p_pinnacle": round(t["p_pin"], 4)})
    saved = {(s["match_id"], s["market"], s["outcome"]): s["id"] for s in store.upsert_targets(rows)}
    targets = targets.copy()
    targets["id"] = [saved[(match_ids[t["fixture_id"]], t["market"], t["outcome"])]
                     for _, t in targets.iterrows()]
    return targets


def weekly_summary(store, now: pd.Timestamp) -> str:
    from predictor.db import picks_frame

    allp = picks_frame(store.picks())
    if allp.empty:
        return format_weekly_summary(allp.assign(status=[]), allp.assign(status=[]),
                                     gate_status(pd.Series({"with_clv": 0, "avg_clv": float("nan")})))
    week = allp[pd.to_datetime(allp["kickoff"], utc=True) >= now - pd.Timedelta(days=7)]
    log = allp.rename(columns={"profit_units": "profit", "stake_units": "stake"})
    return format_weekly_summary(week, allp, gate_status(report(log)["overall"]))


def run(args, store=None, bot=None, client=None, history=None, now=None, log=print) -> str:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if bot is not None and store is not None:
        process_took(store, bot, log=log)
    history = history or load_history(args.league, refresh_current=not args.offline)
    client = client or OddsPapi()
    fixtures, unmatched = live_fixtures(client, args.league, team_names(history),
                                        days=args.hours / 24, now=now)
    for u in unmatched:
        log(f"WARNING: unmapped OddsPapi team {u}; add it to ALIASES in predictor/teams.py")
    log(f"{len(fixtures)} matches in the next {args.hours:g}h. OddsPapi requests: {client.calls} "
        f"(month: {client.month_usage()})")
    probs, xg = model_probs(history, fixtures, log=log)
    targets = all_targets(fixtures, probs, CFG)
    season = next(iter(history.values()))[1]["season"].iloc[0] if history else None
    if store is not None and len(probs):
        targets = save(store, fixtures, probs, xg, targets, season)
    elif not targets.empty:
        targets["id"] = range(1, len(targets) + 1)   # dry run: placeholder numbers
    message = format_targets(targets, now)
    if bot is not None:
        bot.send(message, html=True)
        if now.tz_convert("Africa/Johannesburg").day_of_week == 0 and store is not None:
            bot.send(weekly_summary(store, now), html=True)
    return message


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", nargs="+", default=ALL_LEAGUES)
    ap.add_argument("--hours", type=float, default=24, help="post targets for matches kicking off within this")
    ap.add_argument("--dry-run", action="store_true", help="no database, no Telegram: just print")
    ap.add_argument("--offline", action="store_true", help="use cached football-data files")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # emoji on Windows consoles

    bot = store = None
    if not args.dry_run:
        from predictor.db import Store, Supabase
        from predictor.telegram_bot import Bot
        bot = Bot()
        store = Store(Supabase())
    try:
        message = run(args, store=store, bot=bot)
        if args.dry_run:
            print("\n" + message)
    except Exception:
        if bot is not None:
            bot.alert("⚠️ daily_picks failed:\n" + traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
