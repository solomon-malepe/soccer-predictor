# Soccer value-betting predictor (v1)

Predicts EPL, La Liga, Serie A, Bundesliga and Ligue 1 matches with a
Dixon-Coles model and only bets when the price at your bookmaker is
**better than both the model's fair odds and Pinnacle's fair odds**.

```
data (football-data.co.uk) -> Dixon-Coles model -> probabilities for every scoreline
  -> 1X2 / double chance / O/U 1.5, 2.5, 3.5 / BTTS
  -> compare with Pinnacle (sharp) and your soft books -> value bet or no bet
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Backtest (do this first)

```bash
python run_backtest.py                              # EPL, 2019-20 to now
python run_backtest.py --league E0 SP1 I1 D1 F1     # all five leagues
```

The first run downloads the season CSVs into `data/raw/`. The first 2 seasons
are used only for training; every later match is predicted using **only games
played before it** (refit weekly), the same way it would run live.

Four strategies are replayed on the same predictions:

| strategy | bets when |
|---|---|
| `model` | model prob x soft odds - 1 >= 5% |
| `pinnacle` | Pinnacle fair prob x soft odds - 1 >= 3% (no model at all) |
| `both` | model edge >= 5% **and** Pinnacle edge >= 0% (the main one) |
| `blend` | 50/50 average of model + Pinnacle gives edge >= 3% |

Every strategy skips a bet when the model and Pinnacle disagree by more than
15 percentage points. That usually means bad data, a team-name mismatch,
injury news the model can't see, or a promoted team with few games.

### Reading the output

- **Probability accuracy**: model vs Pinnacle log loss. Pinnacle closing odds
  are the benchmark. Getting close to Pinnacle is already good; beating it is rare.
- **avg_clv**: how much better your odds were than Pinnacle's closing price.
  **This is the number to watch.** Positive CLV over hundreds of bets is the
  best evidence of a real edge.
- **roi / roi_tstat**: profit per unit staked. ROI is noisy; a t-stat below ~2
  means the result could easily be luck.
- **max_drawdown_units**: the worst losing run. Expect it to happen.
- `results/` has every bet (`bets_*.csv`), results per season and per outcome,
  and `profit_curve.png`.

### Things to try

```bash
python run_backtest.py --min-edge 0.07           # pickier
python run_backtest.py --max-odds 3.5            # avoid long shots
python run_backtest.py --markets OU25            # goals markets only
python run_backtest.py --soft-books B365 BW      # which books stand in for yours
python run_backtest.py --xi 0.003 --ridge 2      # model tuning
```

Don't tune on every season and then trust the result. Pick settings on
2021-2024, then check them once on 2025 onwards.

**Limitation:** Betway and SportyBet aren't in football-data's history, so the
backtest uses Bet365 / William Hill (best of the two) as a stand-in for your
two accounts. Your real prices will differ a bit.

## 2. This week's picks

```bash
python predict.py                          # EPL
python predict.py --league E0 SP1 I1 D1 F1
```

This writes `results/upcoming.csv` with probabilities **and fair odds for every
market** (1X2, double chance, O/U 1.5/2.5/3.5, BTTS). If Betway offers more
than the fair odds, that's value according to the model. Value picks under the
`both` rule are printed and saved to `results/picks.csv`.

### Using your own Betway / SportyBet prices

Copy `my_odds_template.csv`, fill in the odds from your apps, then:

```bash
python predict.py --my-odds my_odds.csv
```

Team names must match football-data's spelling (`Man United`, `Nott'm Forest`,
`Wolves`, ...). Check `results/upcoming.csv` for the exact names.

Add a `Pinnacle` row for each match too. football-data stopped publishing Pinnacle
odds in January 2026, and the `both` rule can't pick a match without them.

### Paper trading (Phase 2)

Every matchday:

```bash
python predict.py --league E0 SP1 I1 D1 F1 --my-odds my_odds.csv
python paper.py add        # copies results/picks.csv into paper_log.csv
```

Just before kickoff, type Pinnacle's closing odds into `paper_log.csv`
(`close_H`/`close_D`/`close_A` for 1X2 picks, `close_O25`/`close_U25` for
Over/Under). After the games:

```bash
python paper.py settle     # scores from football-data, profit, CLV
python paper.py report     # record + progress towards the 150-pick gate
```

You can correct `odds` if you got a different price, or set `status` to `void`
for a postponed match. Commit `paper_log.csv` so the record is kept honestly.

## Staking

Picks are in **units**. Each person decides what 1 unit is (1-2% of the money
they've set aside). The backtest also simulates quarter-Kelly staking
(`kelly_growth`), capped at 5% of bankroll per bet.

## Offline test

```bash
python -m tests.simulate --out data/sim
python run_backtest.py --data-dir data/sim --out results_sim
python predict.py --data-dir data/sim --fixtures-file data/sim/fixtures.csv --offline
python -m pytest -q
```

This generates fake seasons from a known model to check the pipeline works.
Profits on simulated data mean nothing.

## Project layout

```
predictor/data.py      download + normalise football-data CSVs
predictor/model.py     Dixon-Coles fit, score matrix, market probabilities
predictor/value.py     de-vig, edges, strategies, bet selection
predictor/backtest.py  walk-forward backtest, CLV, calibration
run_backtest.py        backtest CLI
predict.py             upcoming-fixtures CLI
tests/                 simulator + unit tests
```

## Next steps

1. Run the backtest on real data and look at CLV per strategy.
2. Paper-trade `predict.py` for 1-2 months.
3. Live odds feed (OddsPapi / odds-api.io) for Betway, SportyBet and Pinnacle.
4. Supabase storage + daily GitHub Actions job + Telegram bot.
5. Add xG (Understat) as an extra model input.
