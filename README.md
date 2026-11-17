# Basketball Value Boards

The basketball version of [rhyman7/fb-value-board](https://github.com/rhyman7/fb-value-board): a
model that rates teams, projects every game and compares it with the betting market. NBA (`nbamodel/`) and men's college
(`cbbmodel/`).

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
4. **Market ratings** (`nbamodel/market.py`): team power ratings solved from earlier closing lines
   (version 2, below).
5. **Game model** (`nbamodel/model.py`): ridge regression for margin and total on the ratings'
   projection, rest, back-to-backs and missing players, refitted each season on earlier seasons only.
6. **Market blend**: projection = line + w × (model − line), with w measured on earlier seasons
   against closing lines. It is 0 today (below).

Data: sportsdataverse's [hoopR NBA data](https://github.com/sportsdataverse/hoopR-nba-data) (team
and player box scores, schedules) and the consensus closing lines it publishes (The Odds API, from
the 2020 restart on). No API key needed.

## What it found (Nov 17, 2026)

### NBA: version 2, the market's own ratings

Version 2 adds what a book's line-maker starts from: **market-implied power ratings**
(`nbamodel/market.py`). Each day, every earlier closing spread and total is fit as home court plus
two team ratings, recency-weighted (30-day half-life), with the players who sat each game added back,
so each team's number is its full-strength rating as the market priced it. The game model blends it
with the box-score ratings, rest, back-to-backs and tonight's missing players.

Walk-forward, 2021 through 2026, 7,221 games, graded at the consensus close (`docs/nba/backtest.json`):

| | Version 2 | Version 1 | Closing line |
|---|---|---|---|
| Average miss on the margin | 10.74 pts | 10.79 pts | 10.49 pts |
| Average miss on the total | 14.85 pts | 14.82 pts | 14.25 pts |
| Picks the winner | 66.6% | 66.4% | 68.1% |
| Bet the model's side, every game | 49.2% (−6.0%) | 50.2% (−4.2%) | |
| Model 3+ points off the close | 47.7% (−8.7%) | 48.8% (−6.6%) | |

**It still trails the closing line, now by 0.25 points a game, and earns no weight against it**
(side −0.04 ± 0.06). On the margin the model learns to lean mostly on the market's ratings (0.84)
and a little on the box score (0.41). Totals are still anti-predictive (−0.13 ± 0.05): when the
model's total is off the close, the close is right more often. That is a real pattern, but
fading it hasn't been tested for a keep and it isn't on the board.

The close is the hardest number in sports betting: it already holds the injury news, the rest spot
and the sharp money. The data here has no opening lines, so the place a model like this usually wins
(early lines, before the market has moved) can't be tested yet. The pipeline now logs every ESPN line
it reads (`docs/lines/nba_lines.csv`), which also feeds this season's market ratings, since hoopR's
closing lines only arrive after the season.

### College: version 1

`cbbmodel/` covers men's Division I, built the way college power ratings are:

- **Opponent-adjusted efficiency and pace** (`cbbmodel/ratings.py`), the NBA engine solved directly
  from normal equations so 365 teams stay fast. Neutral floors carry no home court; home court is
  about 3.5 points.
- **A preseason prior**: each team starts at 85% of its final rating from last season...
- **...moved by roster turnover** (`cbbmodel/roster.py`): last season's value of every player on this
  season's roster, wherever he played (transfers count for their new team), against what the team
  produced last season. Losing your best players pulls the prior down; landing transfers pushes it up.
- **Early-season stretch**: the ratings run compressed in November, so the game model gets a fading
  second copy of them, plus back-to-backs (early-season tournaments) and rest.

Graded against ESPN's spread, which hoopR keeps in its play-by-play files for 2015-2023 and this
season (none for 2023-24 to 2025-26). Settings were chosen on 2016-2018; 2019-2023 is the holdout
(`docs/cbb/backtest.json`).

| Home-court games, 2019-2023 (21,708) | Model | ESPN line |
|---|---|---|
| Average miss on the margin | 8.98 pts | 8.77 pts |
| Bet the model's side, every game | 49.6% (−5.1%) | |
| Model 5+ points off the line | 614-546, 52.9% (+1.0%) | |
| Model 6+ points off the line | 357-308, 53.7% (+2.4%) | |

The model trails ESPN's line by about 0.2 points and does earn a small weight against it
(0.10 ± 0.03), the first positive weight either model has earned, steady in four of five seasons.
At the biggest disagreements it has been just over the 52.4% a −110 bet needs, but within one
standard error of break-even: not a bet signal yet.

Two cautions. **Neutral-site games are left out**: ESPN sometimes lists the wrong team as favored in
them (a 10-point favorite shown as a 10-point underdog), which makes any model look brilliant there
(66% at 4+ points). And **this season so far (608 games) the model trails ESPN by a full point**:
two weeks in, the line knows rosters better than last season's numbers do.

So for now both boards show the model's number next to the market's for context, not as a bet signal.

## What to try next

- **Opening lines** (running): both pipelines log every ESPN line they read with the model's number
  at that moment (`docs/lines/`, see `nbamodel/lines.py`). `python -m nbamodel.openers` grades the
  model against the first line logged for each game, and whether the line then moved toward the
  model (closing line value), in `docs/openers.json` every Monday. It needs a few weeks of games
  before the numbers mean anything.
- **College preseason**: recruiting rankings and transfer ratings; this season's two-week deficit
  says the roster prior is the weakest link.
- **NBA totals**: why the model's disagreements run the wrong way.
- **Player ratings** (plus-minus based, lineup-aware) for both leagues.

Every change follows the football boards' keep rule: written down before the run, better on every
measure on tuning and holdout seasons, at least one standard error on the holdout, and Ryan's
go-ahead before it goes on the board.

## Run it

```bash
pip install -r requirements.txt
python -m nbamodel.backtest          # the walk-forward backtest, about 2 minutes
python -m nbamodel.pipeline          # tonight's projections -> docs/feed/nba.json
python -m cbbmodel.backtest          # the college backtest, about 30 seconds
python -m cbbmodel.pipeline          # college projections -> docs/feed/cbb.json
python -m pytest -q
```

The **Update NBA and college boards** Action runs the projections twice a day in season (college
November to April) and the full backtests on Mondays. `docs/feed/nba.json` and `docs/feed/cbb.json` are the feeds the dashboards (rhyman7/nba,
rhyman7/cbb) will read.

## The value board

GitHub Pages serves this repo from its root: `index.html` is the hub (rhyman7.github.io/bb-value-board/),
and `docs/nba/` and `docs/cbb/` are the boards, drawn in the browser by `docs/board.js` from
`docs/feed/*.json`, `docs/*/backtest.json` and `docs/openers.json`. The workflow only has to update
those JSON files. The boards open with a warning that the models don't beat the market, and the
warning's numbers come from the latest backtest. There is no bet list until the opening-line test
or a model change passes the keep rule.
