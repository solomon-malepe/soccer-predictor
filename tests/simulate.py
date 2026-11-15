"""Generate fake seasons in football-data.co.uk's CSV format, from a known
Dixon-Coles "true" model, so the whole pipeline can be tested offline.

Pinnacle odds = true probabilities + small noise + ~2.5% margin (sharp).
Soft books    = true probabilities + bigger noise + ~6% margin (soft).
This tests the MECHANICS only. Real-world profitability can only come from
running on real data.

    python tests/simulate.py --out data/sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predictor.model import DixonColes, market_probs


def make_true_model(teams, rng, attack=None, defence=None):
    m = DixonColes()
    m.teams_ = list(teams)
    m.attack_ = attack if attack is not None else rng.normal(0, 0.3, len(teams))
    m.defence_ = defence if defence is not None else rng.normal(0, 0.25, len(teams))
    m.attack_ -= m.attack_.mean()
    m.defence_ -= m.defence_.mean()
    m.mu0_, m.home_adv_, m.rho_ = 0.15, 0.25, -0.08
    return m


def book_odds(p: np.ndarray, rng, noise: float, margin: float) -> np.ndarray:
    logit = np.log(p) + rng.normal(0, noise, len(p))
    q = np.exp(logit)
    q = q / q.sum()
    return np.round(1 / (q * (1 + margin)), 2)


def simulate(out: Path, league="E0", start_year=2019, n_seasons=8, n_teams=20, seed=7,
             fixtures_rounds=1):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    pool = [f"Team {chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(60)]
    teams = pool[:n_teams]
    next_new = n_teams
    truth = make_true_model(teams, rng)
    truths = {}

    for s in range(n_seasons):
        year = start_year + s
        code = f"{year % 100:02d}{(year + 1) % 100:02d}"
        if s > 0:  # relegate 3 weakest, promote 3 new teams, ratings drift
            strength = truth.attack_ - truth.defence_
            keep = np.argsort(strength)[3:]
            teams = [truth.teams_[i] for i in keep] + pool[next_new:next_new + 3]
            att = np.r_[truth.attack_[keep] + rng.normal(0, 0.08, len(keep)), rng.normal(-0.25, 0.1, 3)]
            dfn = np.r_[truth.defence_[keep] + rng.normal(0, 0.08, len(keep)), rng.normal(0.2, 0.1, 3)]
            next_new += 3
            truth = make_true_model(teams, rng, att, dfn)
        truths[code] = truth

        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        per_round = n_teams // 2
        start = pd.Timestamp(f"{year}-08-10")
        is_last = s == n_seasons - 1
        n_rounds = len(pairs) // per_round
        played_rounds = n_rounds // 3 if is_last else n_rounds  # current season half-played

        rows, fixture_rows = [], []
        for r in range(n_rounds):
            date = start + pd.Timedelta(days=7 * r + int(rng.integers(0, 3)))
            for h, a in pairs[r * per_round:(r + 1) * per_round]:
                mat = truth.score_matrix(h, a)
                p = market_probs(mat)
                p1x2 = np.array([p["H"], p["D"], p["A"]])
                pou = np.array([p["O25"], p["U25"]])
                row = {"Div": league, "Date": date.strftime("%d/%m/%Y"), "HomeTeam": h, "AwayTeam": a}
                for book, noise, mg in (("B365", 0.08, 0.06), ("WH", 0.08, 0.065), ("PS", 0.02, 0.025)):
                    o1 = book_odds(p1x2, rng, noise, mg)
                    oo = book_odds(pou, rng, noise, mg)
                    ou_prefix = "P" if book == "PS" else book
                    row.update({f"{book}H": o1[0], f"{book}D": o1[1], f"{book}A": o1[2],
                                f"{ou_prefix}>2.5": oo[0], f"{ou_prefix}<2.5": oo[1]})
                row.update({"AvgH": row["B365H"], "AvgD": row["B365D"], "AvgA": row["B365A"],
                            "Avg>2.5": row["B365>2.5"], "Avg<2.5": row["B365<2.5"]})
                c1 = book_odds(p1x2, rng, 0.01, 0.022)
                co = book_odds(pou, rng, 0.01, 0.022)
                row.update({"PSCH": c1[0], "PSCD": c1[1], "PSCA": c1[2], "PC>2.5": co[0], "PC<2.5": co[1]})
                if r < played_rounds:
                    flat = rng.choice(mat.size, p=mat.ravel())
                    hg, ag = divmod(int(flat), mat.shape[1])
                    row.update({"FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else "D" if hg == ag else "A"})
                    rows.append(row)
                elif is_last and r < played_rounds + fixtures_rounds:
                    fixture_rows.append(row)
        cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        df = pd.DataFrame(rows)
        df = df[cols + [c for c in df.columns if c not in cols]]
        df.to_csv(out / f"{league}_{code}.csv", index=False)
        if fixture_rows:
            pd.DataFrame(fixture_rows).to_csv(out / "fixtures.csv", index=False)
    return truths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--seasons", type=int, default=8)
    args = ap.parse_args()
    simulate(Path(args.out), start_year=args.start_year, n_seasons=args.seasons)
    print(f"Simulated data written to {args.out}/")
