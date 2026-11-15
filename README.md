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

### Paper trading (Phase 2) with live Pinnacle odds (Phase 3)

Put your OddsPapi key in `.env` as `ODDS_API_KEY=...` (free tier: 250 requests
a month; a normal week uses about 5). OddsPapi supplies fixtures and Pinnacle
odds. Its Betway feed is betway.com, not Betway SA, so you still type Betway and
SportyBet prices.

Every matchday:

```bash
# 1. blank odds sheet with this week's fixtures, names already spelled right
python predict.py --league E0 SP1 I1 D1 F1 --live --write-template my_odds.csv
# 2. fill in Betway + SportyBet prices, then:
python predict.py --league E0 SP1 I1 D1 F1 --live --my-odds my_odds.csv
python paper.py add        # copies results/picks.csv into paper_log.csv
```

After the games:

```bash
python paper.py settle     # scores from football-data, Pinnacle closing odds, CLV
python paper.py report     # record + progress towards the 150-pick gate
```

Without an API key, add a `Pinnacle` row per match to `my_odds.csv` and type
Pinnacle's closing odds into `paper_log.csv` (`close_H`/`close_D`/`close_A` or
`close_O25`/`close_U25`). If OddsPapi names a team that isn't mapped yet (a newly
promoted club), predict.py warns you; add it to `ALIASES` in `predictor/teams.py`.

You can correct `odds` if you got a different price, or set `status` to `void`
for a postponed match. Commit `paper_log.csv` so the record is kept honestly.

## Automation: Supabase + GitHub Actions + Telegram (Phases 4-6)

The odds API can't see Betway SA or SportyBet, so the daily job posts **target
prices** instead of picks: for each match, the lowest Betway/SportyBet price at
which a bet passes the locked `both` rule (at most 2 per match, only targets
within 10% of Pinnacle's price). Friends check their apps; if a book offers the
target or better they bet 1 unit and reply in the group:

```text
/took 57 2.90 sportybet
```

The bot checks the price really meets the target, keeps one pick per match, and
logs it as a paper pick. `/took` replies are read when the jobs run (08:00 and
23:30 SAST), not instantly.

| Job | When (SAST) | What it does |
|---|---|---|
| `python -m jobs.daily_picks` | 08:00 daily | reads `/took`, fits the model, fetches Pinnacle, posts targets for the next 24h; Monday adds the weekly summary |
| `python -m jobs.settle` | 23:30 daily | reads `/took`, fetches Pinnacle closing odds, works out CLV, settles results, posts them |
| `python -m jobs.daily_picks --dry-run --hours 168` | any time | prints next week's targets, saves and sends nothing |

One-time setup:

1. **Supabase:** create a free project. In SQL Editor, run
   `supabase/migrations/001_init.sql`. From Project Settings, then API, copy the
   Project URL and the `service_role` key into `.env` as `SUPABASE_URL` and
   `SUPABASE_SERVICE_KEY`.
2. **Telegram:** in @BotFather send `/newbot` and put the token in `.env` as
   `TELEGRAM_BOT_TOKEN`. Add the bot to the group and send `/start` there, and
   also in a private chat with the bot. Then
   `https://api.telegram.org/bot<TOKEN>/getUpdates` shows both chat ids: the group
   (negative number) goes in `TELEGRAM_CHAT_ID`, your private chat in
   `TELEGRAM_ADMIN_CHAT_ID` (failure alerts).
3. **GitHub:** `git push`, then `gh secret set -f .env` copies every key from
   `.env` into the repo's Actions secrets. Test it with
   `gh workflow run daily.yml -f job=picks-dry-run` and watch it under the Actions tab.

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
