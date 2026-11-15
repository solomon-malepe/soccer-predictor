"""Walk-forward backtest of the Dixon-Coles value-betting system.

Examples
--------
    python run_backtest.py                              # EPL, 2019-20 to now
    python run_backtest.py --league E0 SP1 I1 D1 F1     # all five leagues
    python run_backtest.py --min-edge 0.07 --max-odds 4
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from predictor import LEAGUES
from predictor.backtest import (by_group, calibration_report, run_strategy, summarise,
                                walk_forward_predictions)
from predictor.data import DEFAULT_SOFT_BOOKS, load_seasons, season_codes
from predictor.value import StrategyConfig


def current_season_start_year(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def build_strategies(args) -> list[StrategyConfig]:
    common = dict(min_odds=args.min_odds, max_odds=args.max_odds, markets=args.markets,
                  max_disagreement=args.max_disagreement)
    return [
        StrategyConfig(name="model", strategy="model", min_edge=args.min_edge, **common),
        StrategyConfig(name="pinnacle", strategy="pinnacle", min_edge=args.min_edge_pin_only, **common),
        StrategyConfig(name="both", strategy="both", min_edge=args.min_edge,
                       min_edge_pin=args.min_edge_pin, **common),
        StrategyConfig(name="blend", strategy="blend", min_edge=args.min_edge_blend,
                       blend_weight=args.blend_weight, **common),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", nargs="+", default=["E0"], choices=list(LEAGUES))
    ap.add_argument("--from-year", type=int, default=2019, help="first season start year (2019 = 2019-20)")
    ap.add_argument("--to-year", type=int, default=current_season_start_year())
    ap.add_argument("--train-seasons", type=int, default=2, help="seasons used only for training before testing starts")
    ap.add_argument("--soft-books", nargs="+", default=DEFAULT_SOFT_BOOKS,
                    help="stand-in for your Betway/SportyBet accounts (best price is taken)")
    ap.add_argument("--markets", nargs="+", default=["1X2", "OU25"], choices=["1X2", "OU25"])
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--min-edge-pin", type=float, default=0.0, help="'both' strategy: required edge vs Pinnacle")
    ap.add_argument("--min-edge-pin-only", type=float, default=0.03, help="'pinnacle' strategy threshold")
    ap.add_argument("--min-edge-blend", type=float, default=0.03)
    ap.add_argument("--blend-weight", type=float, default=0.5)
    ap.add_argument("--min-odds", type=float, default=1.30)
    ap.add_argument("--max-odds", type=float, default=5.00)
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--xi", type=float, default=None, help="time-decay per day (default 0.0019)")
    ap.add_argument("--ridge", type=float, default=None, help="rating shrinkage (default 3.0)")
    ap.add_argument("--max-disagreement", type=float, default=0.15,
                    help="skip bets where model and Pinnacle probabilities differ by more than this")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refresh", action="store_true", help="re-download season files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seasons = season_codes(args.from_year, args.to_year)
    strategies = build_strategies(args)

    all_preds, all_bets = [], {s.name: [] for s in strategies}
    for league in args.league:
        print(f"\n=== {LEAGUES[league]} ({league}) seasons {seasons[0]}..{seasons[-1]} ===")
        df = load_seasons(league, seasons, cache_dir=args.data_dir,
                          soft_books=args.soft_books, refresh=args.refresh)
        test_season = seasons[min(args.train_seasons, len(seasons) - 1)]
        test_start = df.loc[df["season"] == test_season, "date"].min()
        print(f"  {len(df)} matches loaded; testing from {test_start.date()} ({test_season})")
        preds = walk_forward_predictions(df, test_start, refit_days=args.refit_days, xi=args.xi,
                                         ridge=args.ridge)
        print(f"  {len(preds)} matches predicted")
        all_preds.append(preds)
        for s in strategies:
            bets = run_strategy(preds, s)
            if not bets.empty:
                bets.insert(0, "league", league)
            all_bets[s.name].append(bets)

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(out / "predictions.csv", index=False)

    print("\n--- Probability accuracy (lower is better; Pinnacle close is the benchmark) ---")
    cal = calibration_report(preds)
    cal.to_csv(out / "calibration.csv", index=False)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summaries = []
    for s in strategies:
        bets = pd.concat([b for b in all_bets[s.name] if not b.empty] or [pd.DataFrame()],
                         ignore_index=True)
        bets.to_csv(out / f"bets_{s.name}.csv", index=False)
        summaries.append(summarise(bets, s.name))
        if not bets.empty:
            by_group(bets, "season").to_csv(out / f"by_season_{s.name}.csv")
            by_group(bets, "outcome").to_csv(out / f"by_outcome_{s.name}.csv")
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)

    print("\n--- Strategy results (flat 1 unit per bet) ---")
    cols = ["strategy", "bets", "hit_rate", "avg_odds", "profit_units", "roi", "roi_tstat",
            "avg_clv", "clv_positive", "max_drawdown_units", "kelly_growth"]
    print(summary.reindex(columns=cols).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for s in strategies:
            path = out / f"bets_{s.name}.csv"
            b = pd.read_csv(path) if path.stat().st_size > 1 else pd.DataFrame()
            if not b.empty:
                ax.plot(range(1, len(b) + 1), b["profit"].cumsum(), label=s.name)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("bet number")
        ax.set_ylabel("profit (units)")
        ax.set_title("Cumulative profit by strategy")
        if ax.get_lines()[1:] or ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "profit_curve.png", dpi=120)
        print(f"\nChart saved to {out / 'profit_curve.png'}")
    except ImportError:
        pass
    print(f"All results written to {out}/")
    print("\nHow to read this: avg_clv > 0 across hundreds of bets is the best sign of a real edge.")
    print("ROI on its own is noisy; |roi_tstat| below ~2 means it could easily be luck.")


if __name__ == "__main__":
    main()
