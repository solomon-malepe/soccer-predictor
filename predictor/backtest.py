"""Walk-forward backtest.

For every test match the model is trained ONLY on games played before that
match's date, refitting once a week, exactly as it would run live. Each
strategy is then replayed on the same predictions so they can be compared fairly.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .model import DixonColes, market_probs
from .value import StrategyConfig, fair_probs, kelly_fraction, select_bet

PRED_OUTCOMES = ["H", "D", "A", "O25", "U25", "BTTS_Y", "O15", "O35"]


def settle(outcome: str, hg: int, ag: int) -> bool:
    return {
        "H": hg > ag, "D": hg == ag, "A": hg < ag,
        "O25": hg + ag > 2.5, "U25": hg + ag < 2.5,
    }[outcome]


def walk_forward_predictions(df: pd.DataFrame, test_start: pd.Timestamp,
                             refit_days: int = 7, train_window_days: int = 1095,
                             min_team_matches: int = 8, xi: float | None = None,
                             ridge: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Model probabilities for every played match on/after test_start."""
    played = df.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
    test = played[played["date"] >= test_start]
    model, last_fit, rows = None, None, []
    kwargs = {k: v for k, v in (("xi", xi), ("ridge", ridge)) if v is not None}

    for date, day_games in test.groupby("date", sort=True):
        if model is None or (date - last_fit).days >= refit_days:
            train = played[(played["date"] < date) &
                           (played["date"] >= date - pd.Timedelta(days=train_window_days))]
            model = DixonColes(**kwargs).fit(train, as_of=date, warm_start=model)
            last_fit = date
            if verbose and sys.stdout.isatty():
                print(f"  refit {date.date()}  on {len(train)} matches", end="\r")
        for _, g in day_games.iterrows():
            if not (model.knows(g["home"], min_team_matches) and
                    model.knows(g["away"], min_team_matches)):
                continue
            p = market_probs(model.score_matrix(g["home"], g["away"]))
            xg_h, xg_a = model.expected_goals(g["home"], g["away"])
            rec = g.to_dict()
            rec.update({f"m_{o}": p[o] for o in PRED_OUTCOMES})
            rec.update({"m_xg_home": xg_h, "m_xg_away": xg_a})
            rows.append(rec)
    if verbose and sys.stdout.isatty():
        print()
    return pd.DataFrame(rows)


def run_strategy(preds: pd.DataFrame, cfg: StrategyConfig, kelly_frac: float = 0.25,
                 start_bank: float = 100.0, max_stake_pct: float = 0.05) -> pd.DataFrame:
    """Replay a strategy over the predictions. Returns one row per bet."""
    bets, bank = [], start_bank
    for _, row in preds.iterrows():
        model_p = {o: row[f"m_{o}"] for o in ["H", "D", "A", "O25", "U25"]}
        bet = select_bet(row, model_p, cfg)
        if bet is None:
            continue
        won = settle(bet["outcome"], int(row["hg"]), int(row["ag"]))
        close = fair_probs(row, "pinc", cfg.devig_method).get(bet["outcome"], np.nan)

        stake_k = min(kelly_frac * kelly_fraction(bet["p_used"], bet["odds"]) * bank,
                      max_stake_pct * bank)
        pnl_k = stake_k * (bet["odds"] - 1) if won else -stake_k
        bank += pnl_k

        bets.append({
            "date": row["date"], "season": row.get("season"), "home": row["home"],
            "away": row["away"], "score": f"{int(row['hg'])}-{int(row['ag'])}",
            **bet, "won": won,
            "profit": (bet["odds"] - 1) if won else -1.0,   # flat 1 unit
            "pin_close_odds": row.get(f"pinc_{bet['outcome']}", np.nan),
            "clv": bet["odds"] * close - 1 if np.isfinite(close) else np.nan,
            "kelly_stake": stake_k, "kelly_bank": bank,
        })
    return pd.DataFrame(bets)


def _max_drawdown(profits: pd.Series) -> float:
    curve = profits.cumsum()
    return float((curve.cummax() - curve).max()) if len(curve) else 0.0


def summarise(bets: pd.DataFrame, name: str, start_bank: float = 100.0) -> dict:
    if bets.empty:
        return {"strategy": name, "bets": 0}
    n = len(bets)
    profit = bets["profit"].sum()
    se = bets["profit"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {
        "strategy": name,
        "bets": n,
        "hit_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "avg_edge": bets["edge"].mean(),
        "profit_units": profit,
        "roi": profit / n,
        "roi_tstat": (profit / n) / se if se and se > 0 else np.nan,
        "avg_clv": bets["clv"].mean(),
        "clv_positive": (bets["clv"] > 0).mean(),
        "max_drawdown_units": _max_drawdown(bets["profit"]),
        "kelly_final_bank": bets["kelly_bank"].iloc[-1],
        "kelly_growth": bets["kelly_bank"].iloc[-1] / start_bank - 1,
    }


def calibration_report(preds: pd.DataFrame) -> pd.DataFrame:
    """Compare the model's 1X2 and O/U 2.5 accuracy with Pinnacle's (lower = better)."""
    rows = []
    res_1x2 = np.select([preds["hg"] > preds["ag"], preds["hg"] == preds["ag"]], [0, 1], 2)
    res_ou = (preds["hg"] + preds["ag"] > 2.5).astype(int).to_numpy()  # 1 = over

    def score(p1x2: np.ndarray, pou: np.ndarray, label: str):
        ok = np.isfinite(p1x2).all(axis=1) & np.isfinite(pou)
        y = np.eye(3)[res_1x2[ok]]
        p = np.clip(p1x2[ok], 1e-9, 1)
        po = np.clip(pou[ok], 1e-9, 1 - 1e-9)
        rows.append({
            "source": label, "matches": int(ok.sum()),
            "logloss_1x2": float(-np.mean(np.sum(y * np.log(p), axis=1))),
            "brier_1x2": float(np.mean(np.sum((p - y) ** 2, axis=1))),
            "logloss_ou25": float(-np.mean(res_ou[ok] * np.log(po) + (1 - res_ou[ok]) * np.log(1 - po))),
        })

    score(preds[["m_H", "m_D", "m_A"]].to_numpy(), preds["m_O25"].to_numpy(), "model")
    for prefix, label in (("pin", "pinnacle_open"), ("pinc", "pinnacle_close")):
        fp = preds.apply(lambda r: fair_probs(r, prefix, "multiplicative"), axis=1)
        p1x2 = np.array([[d["H"], d["D"], d["A"]] for d in fp])
        pou = np.array([d["O25"] for d in fp])
        score(p1x2, pou, label)
    return pd.DataFrame(rows)


def by_group(bets: pd.DataFrame, col: str) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    g = bets.groupby(col)
    return pd.DataFrame({
        "bets": g.size(),
        "hit_rate": g["won"].mean(),
        "profit_units": g["profit"].sum(),
        "roi": g["profit"].mean(),
        "avg_clv": g["clv"].mean(),
    })
