"""Value engine: remove bookmaker margin, compute edges, choose the bet.

Strategies
----------
model     bet when  model_prob x soft_odds - 1 >= min_edge
pinnacle  bet when  pinnacle_fair_prob x soft_odds - 1 >= min_edge  (no model at all)
both      bet when  the model edge >= min_edge AND the Pinnacle edge >= min_edge_pin
blend     bet when  (w x model + (1-w) x pinnacle) x soft_odds - 1 >= min_edge

At most one bet per match: the qualifying market with the biggest edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

MARKET_GROUPS = {
    "1X2": ["H", "D", "A"],
    "OU25": ["O25", "U25"],
    # Experimental (added 2026-11-20): never backtested, because football-data has no
    # historical odds for them. Tracked separately so they can't muddy the main record.
    "BTTS": ["BTTS_Y", "BTTS_N"],
    "OU15": ["O15", "U15"],
    "OU35": ["O35", "U35"],
}
EXPERIMENTAL_MARKETS = ("BTTS", "OU15", "OU35")
MARKET_OF = {o: group for group, outs in MARKET_GROUPS.items() for o in outs}   # outcome -> market


def is_experimental(market: str) -> bool:
    return market in EXPERIMENTAL_MARKETS


def devig(odds: list[float] | np.ndarray, method: str = "multiplicative") -> np.ndarray:
    """Fair probabilities from a full set of odds for one market (e.g. H/D/A)."""
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        return np.full(len(odds), np.nan)
    implied = 1.0 / odds
    if method == "multiplicative":
        return implied / implied.sum()
    if method == "power":
        if implied.sum() <= 1.0:
            return implied / implied.sum()
        k = brentq(lambda k: np.sum(implied ** k) - 1.0, 1.0, 10.0)
        return implied ** k
    raise ValueError(f"Unknown devig method: {method}")


def margin(odds: list[float]) -> float:
    """Bookmaker overround, e.g. 0.05 for a 5% margin."""
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)) - 1.0)


def kelly_fraction(p: float, odds: float) -> float:
    if odds <= 1:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


@dataclass
class StrategyConfig:
    name: str = "both"
    strategy: str = "both"            # model | pinnacle | both | blend
    min_edge: float = 0.05
    min_edge_pin: float = 0.0          # only used by "both"
    blend_weight: float = 0.5          # model weight, only used by "blend"
    min_odds: float = 1.30
    max_odds: float = 5.00
    markets: list[str] = field(default_factory=lambda: ["1X2", "OU25"])
    devig_method: str = "multiplicative"
    # If the model and Pinnacle disagree by more than this, something is usually
    # wrong (team-name mismatch, injury news, promoted team) -> no bet.
    max_disagreement: float = 0.15


def fair_probs(row, prefix: str, method: str) -> dict[str, float]:
    """De-vigged probabilities for every outcome, from columns like pin_H, pinc_O25."""
    out = {}
    for outcomes in MARKET_GROUPS.values():
        probs = devig([row.get(f"{prefix}_{o}", np.nan) for o in outcomes], method)
        out.update(dict(zip(outcomes, probs)))
    return out


def candidates(row, model_p: dict[str, float], cfg: StrategyConfig) -> list[dict]:
    """Every outcome on this match, with its edges under the strategy."""
    pin = fair_probs(row, "pin", cfg.devig_method)
    out = []
    for group in cfg.markets:
        for o in MARKET_GROUPS[group]:
            odds = row.get(f"soft_{o}", np.nan)
            if not np.isfinite(odds) or not (cfg.min_odds <= odds <= cfg.max_odds):
                continue
            pm, pp = model_p.get(o, np.nan), pin.get(o, np.nan)
            edge_model = pm * odds - 1 if np.isfinite(pm) else np.nan
            edge_pin = pp * odds - 1 if np.isfinite(pp) else np.nan

            if cfg.strategy == "model":
                p_used, ok = pm, edge_model >= cfg.min_edge
            elif cfg.strategy == "pinnacle":
                p_used, ok = pp, edge_pin >= cfg.min_edge
            elif cfg.strategy == "both":
                p_used = pm
                ok = (edge_model >= cfg.min_edge) and (edge_pin >= cfg.min_edge_pin)
            elif cfg.strategy == "blend":
                p_used = cfg.blend_weight * pm + (1 - cfg.blend_weight) * pp
                ok = np.isfinite(p_used) and (p_used * odds - 1 >= cfg.min_edge)
            else:
                raise ValueError(f"Unknown strategy: {cfg.strategy}")

            if not np.isfinite(p_used):
                continue
            disagree = abs(pm - pp) if (np.isfinite(pm) and np.isfinite(pp)) else np.nan
            if cfg.strategy != "pinnacle" and np.isfinite(disagree) and disagree > cfg.max_disagreement:
                ok = False
            out.append({
                "market": group, "outcome": o, "odds": float(odds),
                "book": row.get(f"softbook_{o}", ""),
                "p_model": pm, "p_pin": pp, "p_used": float(p_used),
                "edge": float(p_used * odds - 1),
                "edge_model": edge_model, "edge_pin": edge_pin, "disagreement": disagree,
                "qualifies": bool(ok),
            })
    return out


def select_bet(row, model_p: dict[str, float], cfg: StrategyConfig) -> dict | None:
    """The single best qualifying bet on this match, or None."""
    qualifying = [c for c in candidates(row, model_p, cfg) if c["qualifies"]]
    if not qualifying:
        return None
    return max(qualifying, key=lambda c: c["edge"])
