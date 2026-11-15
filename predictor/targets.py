"""Target prices: the lowest Betway / SportyBet price at which a bet becomes value.

The system can't see Betway SA or SportyBet prices, so instead of picks it posts
targets. For each outcome it works out the price that satisfies the locked
`both` rule:

    model edge    >= 5%   ->  price >= 1.05 / p_model
    Pinnacle edge >= 2%   ->  price >= 1.02 / p_pinnacle
    1.30 <= price <= 5.00, and model and Pinnacle within 15 points of each other

target = the larger of the two. A friend checks their apps; if either book offers
the target or better, that's the bet.

Most targets are far above anything a bookmaker will offer (they need to beat
Pinnacle's own price). Only targets within 10% of Pinnacle's actual price are
posted, at most 2 per match. On 2021-24 data that kept 51 of the 54 bets the
backtest found, at about 1.8 lines per match; a target is actually on offer for
roughly 1 match in 100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .value import MARKET_GROUPS, StrategyConfig, devig

LABELS = {"H": "Home win", "D": "Draw", "A": "Away win", "O25": "Over 2.5", "U25": "Under 2.5"}


def target_price(p_model: float, p_pin: float, cfg: StrategyConfig) -> float:
    """Lowest price that passes the `both` rule, or NaN if no price in range does."""
    if not (np.isfinite(p_model) and np.isfinite(p_pin)) or p_model <= 0 or p_pin <= 0:
        return np.nan
    if abs(p_model - p_pin) > cfg.max_disagreement:
        return np.nan
    t = max((1 + cfg.min_edge) / p_model, (1 + cfg.min_edge_pin) / p_pin, cfg.min_odds)
    return float(t) if t <= cfg.max_odds else np.nan


def round_up_price(price: float) -> float:
    """Round up to a price bookmakers actually show (2 decimals below 3.0, 0.05 above)."""
    if not np.isfinite(price):
        return price
    step = 0.01 if price < 3 else 0.05
    return float(np.ceil(round(price / step, 6)) * step).__round__(2)


def match_targets(row, model_p: dict[str, float], cfg: StrategyConfig,
                  max_over_pinnacle: float = 0.10) -> list[dict]:
    """Postable targets for one fixture row (needs pin_* columns), best first."""
    out = []
    for group in cfg.markets:
        outcomes = MARKET_GROUPS[group]
        pin_odds = [row.get(f"pin_{o}", np.nan) for o in outcomes]
        fair = dict(zip(outcomes, devig(pin_odds, cfg.devig_method)))
        for o, pin in zip(outcomes, pin_odds):
            t = target_price(model_p.get(o, np.nan), fair[o], cfg)
            if not np.isfinite(t) or not np.isfinite(pin):
                continue
            t = round_up_price(t)
            over = t / pin - 1
            if over > max_over_pinnacle:
                continue
            out.append({"market": group, "outcome": o, "target_odds": t, "pin_odds": float(pin),
                        "p_model": float(model_p[o]), "p_pin": float(fair[o]), "over_pinnacle": over})
    return sorted(out, key=lambda r: r["over_pinnacle"])


def all_targets(fixtures: pd.DataFrame, model_probs: dict, cfg: StrategyConfig,
                max_over_pinnacle: float = 0.10, per_match: int = 2) -> pd.DataFrame:
    """Targets for every fixture. `model_probs` maps fixture index -> {outcome: p}."""
    rows = []
    for idx, r in fixtures.iterrows():
        if idx not in model_probs:
            continue
        for t in match_targets(r, model_probs[idx], cfg, max_over_pinnacle)[:per_match]:
            rows.append({"fixture_index": idx, **{k: r.get(k) for k in
                         ("league", "date", "kickoff", "home", "away", "fixture_id")}, **t})
    return pd.DataFrame(rows)


def qualifies(price: float, p_model: float, p_pin: float, cfg: StrategyConfig) -> bool:
    """Does a price someone actually got pass the `both` rule?"""
    t = target_price(p_model, p_pin, cfg)
    return bool(np.isfinite(t) and price >= t - 1e-9 and price <= cfg.max_odds)
