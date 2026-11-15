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
GATE_PICKS_TEXT = ("Real money starts only after 150+ paper picks with closing odds show average CLV "
                   "above 0. CLV below 0 after 150: back to Phase 1.")
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


def log_pick(store, target_id: int, odds: float, book: str, user: str, when,
             cfg: StrategyConfig = CFG) -> tuple[dict | None, str]:
    """Log a paper pick against a posted target, enforcing the betting rules.
    Used by Telegram /took and the dashboard. Returns (pick or None, message)."""
    t = store.target(target_id)
    if not t:
        return None, f"There's no target #{target_id}."
    when = pd.Timestamp(when)
    when = when.tz_localize("UTC") if when.tzinfo is None else when
    if when >= pd.Timestamp(t["matches"]["kickoff"]):
        return None, f"#{target_id} had already kicked off, so it isn't logged."
    p_model, p_pin = float(t["p_model"]), float(t["p_pinnacle"])
    if not qualifies(odds, p_model, p_pin, cfg):
        return None, (f"{odds:.2f} is below the target of {float(t['target_odds']):.2f} for #{target_id}, "
                      f"so it isn't a value bet and isn't logged.")
    pick = store.add_pick({
        "match_id": t["match_id"], "prediction_id": t.get("prediction_id"), "target_id": target_id,
        "strategy": cfg.strategy, "market": t["market"], "outcome": t["outcome"], "book": book,
        "odds": odds, "p_model": p_model, "p_pinnacle": p_pin, "edge": round(p_model * odds - 1, 4),
        "stake_units": 1, "is_paper": True, "status": "open", "logged_by": user,
        "sent_at": when.isoformat(),
    })
    if pick is None:
        return None, ("That match already has a pick logged (one pick per match). Bet it if you like, "
                      "but the record keeps the first one.")
    return pick, format_pick_logged(user, t, odds, book)


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
        pick, message = log_pick(store, target_id, odds, book, c["user"], c["date"], cfg)
        bot.send(message, reply_to=c["message_id"])
        logged += pick is not None
    if updates:
        store.set_state("telegram_offset", max(u["update_id"] for u in updates))
    if logged:
        log(f"Logged {logged} pick(s) from Telegram")
    return logged
