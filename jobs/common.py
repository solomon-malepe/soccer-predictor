"""Pieces shared by the scheduled jobs."""

from __future__ import annotations

import pandas as pd

from predictor import LEAGUES
from predictor.data import load_seasons, season_codes
from predictor.targets import qualifies
from predictor.telegram_bot import format_pick_logged, parse_took, took_commands
from predictor.value import StrategyConfig

# Locked in Phase 1 (2026-11-15): both, model >= 5%, Pinnacle >= 2%, odds 1.30-5.00
CFG = StrategyConfig(name="both", strategy="both", min_edge=0.05, min_edge_pin=0.02,
                     min_odds=1.30, max_odds=5.00, max_disagreement=0.15)
MODEL_VERSION = "dc-v1-xi0019-r3"
ALL_LEAGUES = list(LEAGUES)


def current_season_start_year(today=None) -> int:
    today = today or pd.Timestamp.now(tz="UTC")
    return today.year if today.month >= 7 else today.year - 1


def load_history(leagues, history_seasons: int = 3, data_dir: str = "data/raw",
                 refresh_current: bool = True) -> dict:
    """league -> (older seasons, current season) from football-data."""
    end = current_season_start_year()
    seasons = season_codes(end - history_seasons + 1, end)
    out = {}
    for lg in leagues:
        cur = load_seasons(lg, seasons[-1:], cache_dir=data_dir, refresh=refresh_current)
        hist = load_seasons(lg, seasons[:-1], cache_dir=data_dir) if seasons[:-1] else cur.iloc[:0]
        out[lg] = (hist, cur)
    return out


def team_names(history: dict) -> dict[str, list[str]]:
    return {lg: sorted(set(cur["home"]) | set(cur["away"])) for lg, (_, cur) in history.items()}


def process_took(store, bot, cfg: StrategyConfig = CFG, log=print) -> int:
    """Read /took replies since the last run, log valid ones as paper picks, reply to each.
    Returns the number of picks logged."""
    last = store.state("telegram_offset")
    updates = bot.updates(int(last) + 1 if last else None)
    logged = 0
    for c in took_commands(updates, bot.chat_id):
        parsed = parse_took(c["text"])
        if isinstance(parsed, str):
            bot.send(parsed, reply_to=c["message_id"])
            continue
        target_id, odds, book = parsed
        t = store.target(target_id)
        if not t:
            bot.send(f"There's no target #{target_id}.", reply_to=c["message_id"])
            continue
        kickoff = pd.Timestamp(t["matches"]["kickoff"])
        if c["date"] >= kickoff:
            bot.send(f"#{target_id} had already kicked off when you replied, so it isn't logged.",
                     reply_to=c["message_id"])
            continue
        p_model, p_pin = float(t["p_model"]), float(t["p_pinnacle"])
        if not qualifies(odds, p_model, p_pin, cfg):
            bot.send(f"{odds:.2f} is below the target of {float(t['target_odds']):.2f} for #{target_id}, "
                     f"so it isn't a value bet and isn't logged.", reply_to=c["message_id"])
            continue
        pick = store.add_pick({
            "match_id": t["match_id"], "prediction_id": t.get("prediction_id"), "target_id": target_id,
            "strategy": cfg.strategy, "market": t["market"], "outcome": t["outcome"], "book": book,
            "odds": odds, "p_model": p_model, "p_pinnacle": p_pin, "edge": round(p_model * odds - 1, 4),
            "stake_units": 1, "is_paper": True, "status": "open", "logged_by": c["user"],
            "sent_at": c["date"].isoformat(),
        })
        if pick is None:
            bot.send("That match already has a pick logged (one pick per match). Bet it if you like, "
                     "but the record keeps the first one.", reply_to=c["message_id"])
            continue
        bot.send(format_pick_logged(c["user"], t, odds, book), reply_to=c["message_id"])
        logged += 1
    if updates:
        store.set_state("telegram_offset", max(u["update_id"] for u in updates))
    if logged:
        log(f"Logged {logged} pick(s) from Telegram")
    return logged
