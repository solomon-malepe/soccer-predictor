"""Phase 2 paper-trading log.

    python paper.py add        # after predict.py: copy today's picks into paper_log.csv
    python paper.py settle     # after the games: fill scores from football-data, work out CLV
    python paper.py report     # record so far + progress towards the 150-pick gate

Workflow each matchday
----------------------
1. python predict.py --league E0 SP1 I1 D1 F1 --live --write-template my_odds.csv
   (fill in Betway + SportyBet prices)
2. python predict.py --league E0 SP1 I1 D1 F1 --live --my-odds my_odds.csv
3. python paper.py add
4. After the games: python paper.py settle, then python paper.py report
   settle fetches Pinnacle's closing odds from OddsPapi. Without an API key, type
   them into paper_log.csv (close_H/close_D/close_A or close_O25/close_U25).

football-data.co.uk usually posts results within a day or two, so settle may
leave some picks open at first; just run it again later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predictor import LEAGUES
from predictor.data import load_seasons, season_codes
from predictor.odds_feed import load_api_key
from predictor.paper import (add_picks, fill_closing, gate_status, missing_close, read_log, report,
                             settle_log)
from run_backtest import current_season_start_year

FMT = {"hit_rate": "{:.0%}", "units": "{:+.2f}", "roi": "{:+.1%}", "avg_clv": "{:+.2%}", "clv_t": "{:.2f}"}


def show(df: pd.DataFrame) -> str:
    df = df.copy()
    for c, f in FMT.items():
        if c in df:
            df[c] = df[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    for c in ("picks", "settled", "with_clv"):
        if c in df:
            df[c] = df[c].astype(int)
    return df.to_string()


def cmd_add(args) -> None:
    picks_path = Path(args.picks)
    if not picks_path.exists():
        print(f"No picks file at {picks_path}. Run predict.py first (it only writes one when there are picks).")
        return
    log, added, skipped = add_picks(read_log(args.log), pd.read_csv(picks_path))
    log.to_csv(args.log, index=False)
    print(f"Added {added} pick(s) to {args.log}" + (f"; skipped {skipped} already logged." if skipped else "."))
    if added and not load_api_key():
        print("Remember: type Pinnacle's closing odds into the close_* columns near kickoff.")
    elif added:
        print("After the games, `python paper.py settle` fetches results and Pinnacle closing odds.")


def cmd_settle(args) -> None:
    log = read_log(args.log)
    need = log[(log["status"] == "open") | (log.apply(missing_close, axis=1) & (log["status"] != "void"))]
    frames = []
    season = season_codes(current_season_start_year(), current_season_start_year())
    for league in sorted(set(need["league"]) & set(LEAGUES)):
        frames.append(load_seasons(league, season, cache_dir=args.data_dir, refresh=not args.offline))
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["league", "home", "away", "date", "hg", "ag"])

    if not args.no_closing and len(need) and load_api_key():
        from predictor.odds_feed import OddsPapi
        client = OddsPapi()
        fd_names = {lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in results.groupby("league")}
        log, filled = fill_closing(log, client, fd_names)
        print(f"Pinnacle closing odds fetched for {filled} pick(s) "
              f"(requests: {client.calls}, this month {client.month_usage()}/250).")
    elif not args.no_closing and len(need):
        print("No ODDS_API_KEY in .env, so closing odds must be typed in by hand.")

    log, settled = settle_log(log, results)
    log.to_csv(args.log, index=False)
    still_open = (log["status"] == "open").sum()
    no_close = (log["clv"].isna() & (log["status"] != "void")).sum()
    print(f"Settled {settled} pick(s). Still open: {still_open}. Missing closing odds: {no_close}.")


def cmd_report(args) -> None:
    log = read_log(args.log)
    if log.empty:
        print(f"{args.log} is empty. Log some picks with: python paper.py add")
        return
    r = report(log)
    print("=== Paper record (flat stakes, units) ===")
    print(show(r["overall"].to_frame("all").T))
    for name in ("by_league", "by_outcome"):
        if name in r:
            print(f"\n--- {name.replace('_', ' ')} ---")
            print(show(r[name]))
    print("\n" + gate_status(r["overall"]))
    print("CLV decides this, not profit: |clv_t| above ~2 means it's unlikely to be luck.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="paper_log.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="copy picks from predict.py into the log")
    a.add_argument("--picks", default="results/picks.csv")
    s = sub.add_parser("settle", help="fill results and CLV")
    s.add_argument("--data-dir", default="data/raw")
    s.add_argument("--offline", action="store_true", help="use cached result files, no download")
    s.add_argument("--no-closing", action="store_true", help="don't fetch closing odds from OddsPapi")
    sub.add_parser("report", help="show the record and gate progress")
    args = ap.parse_args()
    {"add": cmd_add, "settle": cmd_settle, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
