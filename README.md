# Basketball Value Boards

The basketball version of [rhyman7/fb-value-board](https://github.com/rhyman7/fb-value-board): a
model that rates teams, projects every game and compares it with the betting market. NBA first;
college (rhyman7/cbb) follows with the same code.

## NBA: what it does

1. **Team ratings** (`nbamodel/ratings.py`). Offense and defense in points per 100 possessions and
   pace in possessions, solved by ridge regression over every earlier game at once, so they are
   adjusted for opponents and home court. This season's games are weighted by recency (45-day
   half-life); last season's count 35% and fade as this season fills in. Same engine as the
   football boards' EPA ratings.
2. **Schedule spots** (`nbamodel/data.py`): days of rest (back-to-backs), and games in the last 7 days.
3. **Missing players** (`nbamodel/availability.py`): each team's regulars who don't play, valued by
   their minutes times their production per minute above replacement (Hollinger's Game Score,
   shrunk toward replacement). The backtest knows who played; the live board reads ESPN's injury
   report (players listed Out or Doubtful).
4. **Game model** (`nbamodel/model.py`): ridge regression for margin and total on the ratings'
   projection, rest, back-to-backs and missing players, refitted each season on earlier seasons only.
5. **Market blend**: projection = line + w × (model − line), with w measured on earlier seasons
   against closing lines. It is 0 today (below).

Data: sportsdataverse's [hoopR NBA data](https://github.com/sportsdataverse/hoopR-nba-data) (team
and player box scores, schedules) and the consensus closing lines it publishes (The Odds API, from
the 2020 restart on). No API key needed.

## What it found (Nov 17, 2026)

Walk-forward, 2021 through 2026, 7,221 regular-season games, graded at the consensus close
(`docs/nba/backtest.json`). Rating settings were chosen on 2017-2020, seasons with no closing lines.

| | Model | Ratings alone | Closing line |
|---|---|---|---|
| Average miss on the margin | 10.79 pts | 11.01 pts | 10.49 pts |
| Average miss on the total | 14.82 pts | | 14.25 pts |
| Picks the winner | 66.4% | | 68.1% |

| One bet on the model's side at the close | Record | Win rate | Return at −110 |
|---|---|---|---|
| Every game | 3,566-3,544-111 | 50.2% | −4.2% |
| Model 2+ points off the close | 1,633-1,688-56 | 49.2% | −6.0% |
| Model 3+ points off | 971-1,017-32 | 48.8% | −6.6% |
| Totals, every game | 3,486-3,653-82 | 48.8% | −6.7% |

**The model trails the closing line by about 0.3 points a game and earns no weight against it**
(side weight −0.03 ± 0.06; totals −0.13 ± 0.05, so the model's totals disagreements have been
slightly wrong more often than right). The same was true of the football model before quarterback
and injury inputs. Missing players already cut the gap from 0.52 to 0.30 points. In every season
the model trailed the close by 0.2 to 0.35 points.

What it learned (2026 fit): a back-to-back costs the tired team about 1.6 to 1.9 points; each point
of missing Game Score value costs about a third of a point; the ratings' margin carries through at
about 1.07.

So for now the board shows the model's number next to the market's for context, not as a bet signal.

## What to try next

- **Opening lines.** The football model's edge lives early in the week; NBA openers would need a
  line history (The Odds API historical endpoint, or saving ESPN's line from the dashboard builds,
  which started Nov 17, 2026).
- **Better player values** (plus-minus based, lineup-aware) and **returning players** (a team whose
  ratings were built while a star was out).
- **Totals**: why the disagreements run the wrong way (league-wide scoring drift within a season is
  the first suspect).

Every change follows the football boards' keep rule: written down before the run, better on every
measure on tuning and holdout seasons, at least one standard error on the holdout, and Ryan's
go-ahead before it goes on the board.

## Run it

```bash
pip install -r requirements.txt
python -m nbamodel.backtest          # the walk-forward backtest, about 2 minutes
python -m nbamodel.pipeline          # tonight's projections -> docs/feed/nba.json
python -m pytest -q
```

The **Update NBA board** Action runs the projections twice a day in season and the full backtest
on Mondays. `docs/feed/nba.json` is the feed the NBA dashboard (rhyman7/nba) will read.
