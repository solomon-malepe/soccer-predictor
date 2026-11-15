"""Price upcoming matches and list value picks.

Live mode (recommended): fixtures and Pinnacle odds come from OddsPapi, and you
type only Betway + SportyBet prices.

    python predict.py --league E0 SP1 I1 D1 F1 --live --write-template my_odds.csv
    #   ...fill in Betway / SportyBet prices in my_odds.csv...
    python predict.py --league E0 SP1 I1 D1 F1 --live --my-odds my_odds.csv

Without --live, football-data.co.uk/fixtures.csv gives the fixture list and its
odds. To use YOUR Betway / SportyBet prices instead, pass
--my-odds with a CSV like my_odds_template.csv:

    home,away,book,H,D,A,O25,U25
    Arsenal,Chelsea,Betway,2.10,3.40,3.50,1.95,1.85
    Arsenal,Chelsea,SportyBet,2.15,3.30,3.45,2.05,1.78
    Arsenal,Chelsea,Pinnacle,2.20,3.45,3.60,2.02,1.88

Without --live the Pinnacle row is required for the 'both' strategy: fixtures.csv
has had no Pinnacle odds since January 2026. With --live a Pinnacle row is optional
and overrides the API price.

Team names must match football-data's spelling (e.g. "Man United", "Nott'm Forest").

Examples
--------
    python predict.py                         # EPL
    python predict.py --league E0 SP1 I1 D1 F1
    python predict.py --my-odds my_odds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import LEAGUES
from predictor.data import DEFAULT_SOFT_BOOKS, load_fixtures, load_seasons, season_codes
from predictor.model import DixonColes, market_probs
from predictor.value import StrategyConfig, select_bet
from run_backtest import current_season_start_year

OUTCOMES = ["H", "D", "A", "O25", "U25"]
LABELS = {"H": "Home win", "D": "Draw", "A": "Away win", "O25": "Over 2.5", "U25": "Under 2.5"}


def apply_my_odds(fixtures: pd.DataFrame, my_odds: pd.DataFrame) -> pd.DataFrame:
    """Replace soft_* columns with the best of your own bookmaker prices.

    A row with book "Pinnacle" fills the pin_* columns instead. football-data stopped
    publishing Pinnacle odds in January 2026, so the 'both' strategy needs these typed in.
    """
    fx = fixtures.copy()
    is_pin = my_odds["book"].astype(str).str.strip().str.lower() == "pinnacle"
    soft, pin = my_odds[~is_pin], my_odds[is_pin]
    for o in OUTCOMES:
        fx[f"soft_{o}"] = np.nan
        fx[f"softbook_{o}"] = ""
    for i, r in fx.iterrows():
        rows = soft[(soft["home"] == r["home"]) & (soft["away"] == r["away"])]
        for o in OUTCOMES:
            if o in rows and rows[o].notna().any():
                j = rows[o].idxmax()
                fx.at[i, f"soft_{o}"] = rows.at[j, o]
                fx.at[i, f"softbook_{o}"] = rows.at[j, "book"]
        prow = pin[(pin["home"] == r["home"]) & (pin["away"] == r["away"])]
        if not prow.empty:
            for o in OUTCOMES:
                if o in prow and pd.notna(prow[o].iloc[-1]):
                    fx.at[i, f"pin_{o}"] = prow[o].iloc[-1]
    return fx


def write_template(fixtures: pd.DataFrame, path: str) -> None:
    """Blank odds sheet: one Betway and one SportyBet row per fixture, names already
    in football-data spelling, kickoff in SA time."""
    out = Path(path)
    if out.exists():
        raise SystemExit(f"{out} already exists. Delete or rename it first so typed odds aren't overwritten.")
    rows = []
    for _, r in fixtures.iterrows():
        ko = pd.Timestamp(r["kickoff"]).tz_convert("Africa/Johannesburg").strftime("%a %d %b %H:%M")
        for book in ("Betway", "SportyBet"):
            rows.append({"league": r["league"], "kickoff_sast": ko, "home": r["home"],
                         "away": r["away"], "book": book, **{o: "" for o in OUTCOMES}})
    pd.DataFrame(rows, columns=["league", "kickoff_sast", "home", "away", "book", *OUTCOMES]).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows ({len(fixtures)} matches) to {out}. Fill in the prices; "
          f"leave a cell blank if the book doesn't offer it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", nargs="+", default=["E0"], choices=list(LEAGUES))
    ap.add_argument("--history-seasons", type=int, default=3)
    ap.add_argument("--soft-books", nargs="+", default=DEFAULT_SOFT_BOOKS)
    ap.add_argument("--my-odds", help="CSV with your own Betway/SportyBet prices")
    ap.add_argument("--fixtures-file", help="use a local fixtures.csv instead of downloading")
    ap.add_argument("--live", action="store_true",
                    help="fixtures + Pinnacle odds from OddsPapi (needs ODDS_API_KEY in .env)")
    ap.add_argument("--days", type=float, default=7, help="--live: how many days ahead to price")
    ap.add_argument("--write-template", metavar="CSV",
                    help="--live: write a blank Betway/SportyBet odds sheet for the fixtures, then stop")
    ap.add_argument("--strategy", default="both", choices=["model", "pinnacle", "both", "blend"])
    ap.add_argument("--min-edge", type=float, default=0.05)
    # Locked in Phase 1 (2026-11-15): both, model >=5%, Pinnacle >=2%, odds 1.30-5.00
    ap.add_argument("--min-edge-pin", type=float, default=0.02)
    ap.add_argument("--max-odds", type=float, default=5.0)
    ap.add_argument("--max-disagreement", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=3.0)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/upcoming.csv")
    ap.add_argument("--offline", action="store_true", help="use cached files only, no downloads")
    args = ap.parse_args()

    end = current_season_start_year()
    seasons = season_codes(end - args.history_seasons + 1, end)
    if args.write_template and not args.live:
        ap.error("--write-template needs --live")

    history = {}
    for league in args.league:
        # current season is always re-downloaded so the latest results are included
        hist = load_seasons(league, seasons[:-1], cache_dir=args.data_dir, soft_books=args.soft_books)
        cur = load_seasons(league, seasons[-1:], cache_dir=args.data_dir,
                           soft_books=args.soft_books, refresh=not args.offline)
        history[league] = (hist, cur)

    if args.live:
        from predictor.odds_feed import OddsPapi, live_fixtures
        client = OddsPapi()
        fd_names = {lg: sorted(set(cur["home"]) | set(cur["away"])) for lg, (_, cur) in history.items()}
        fixtures_all, unmatched = live_fixtures(client, args.league, fd_names, days=args.days)
        for u in unmatched:
            print(f"WARNING: no football-data name for OddsPapi team {u}. Add it to ALIASES in predictor/teams.py")
        print(f"OddsPapi: {len(fixtures_all)} fixtures in the next {args.days:g} days. "
              f"Requests this run: {client.calls}, this month: {client.month_usage()}/250")
        if args.write_template:
            write_template(fixtures_all, args.write_template)
            return
        if not args.my_odds:
            print("Note: --live gives Pinnacle only. Add --my-odds with your Betway/SportyBet prices to get picks.")
    else:
        fixtures_all = load_fixtures(args.league, soft_books=args.soft_books, path=args.fixtures_file)
    if args.my_odds:
        fixtures_all = apply_my_odds(fixtures_all, pd.read_csv(args.my_odds))
    if args.strategy != "model" and len(fixtures_all):
        no_pin = fixtures_all[[f"pin_{o}" for o in OUTCOMES]].isna().all(axis=1)
        if no_pin.any():
            print(f"WARNING: {no_pin.sum()} of {len(fixtures_all)} matches have no Pinnacle odds, so the "
                  f"'{args.strategy}' rule can't pick them. Add a 'Pinnacle' row per match to --my-odds.")
    cfg = StrategyConfig(name=args.strategy, strategy=args.strategy, min_edge=args.min_edge,
                         min_edge_pin=args.min_edge_pin, max_odds=args.max_odds,
                         max_disagreement=args.max_disagreement)

    rows, picks = [], []
    for league in args.league:
        hist, cur = history[league]
        model = DixonColes(ridge=args.ridge).fit(pd.concat([hist, cur], ignore_index=True))
        fx = fixtures_all[fixtures_all["league"] == league] if len(fixtures_all) else fixtures_all
        print(f"\n=== {LEAGUES[league]}: {len(fx)} upcoming matches ===")
        for _, r in fx.iterrows():
            if not (model.knows(r["home"], 5) and model.knows(r["away"], 5)):
                print(f"  skip {r['home']} vs {r['away']} (not enough data / name mismatch)")
                continue
            p = market_probs(model.score_matrix(r["home"], r["away"]))
            xg_h, xg_a = model.expected_goals(r["home"], r["away"])
            rec = {"league": league, "date": r["date"].date(), "home": r["home"], "away": r["away"],
                   "xg_home": round(xg_h, 2), "xg_away": round(xg_a, 2)}
            for k, v in p.items():
                rec[f"p_{k}"] = round(v, 4)
                rec[f"fair_{k}"] = round(1 / v, 2) if v > 0 else np.nan
            rows.append(rec)
            bet = select_bet(r, {o: p[o] for o in OUTCOMES}, cfg)
            if bet:
                extra = {k: r[k] for k in ("fixture_id", "kickoff") if k in r and pd.notna(r[k])}
                picks.append({**{k: rec[k] for k in ("league", "date", "home", "away")}, **extra, **bet})

    if not rows:
        print("No matches priced.")
        return
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nFull probabilities and fair odds for every market saved to {args.out}")

    if not picks:
        # don't leave an old picks.csv behind for `paper.py add` to pick up
        Path(args.out).with_name("picks.csv").unlink(missing_ok=True)
        print(f"\nNo value picks today under the '{args.strategy}' rule. That's normal; most games have none.")
        return
    print(f"\n*** Value picks ({args.strategy}, min edge {args.min_edge:.0%}) ***")
    for b in sorted(picks, key=lambda b: -b["edge"]):
        print(f"{b['date']}  {b['home']} vs {b['away']}  ·  {LABELS[b['outcome']]} @ {b['odds']:.2f} "
              f"({b['book']})  ·  model {b['p_model']:.0%} / Pinnacle {b['p_pin']:.0%}  ·  "
              f"edge {b['edge']:+.1%}  ·  1 unit")
    pd.DataFrame(picks).to_csv(Path(args.out).with_name("picks.csv"), index=False)


if __name__ == "__main__":
    main()
