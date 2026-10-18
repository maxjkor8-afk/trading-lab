# trading-lab

A point-in-time backtest harness for crypto strategies, written so that
look-ahead bias raises an exception instead of inflating a result. It is a
measuring instrument: an event-driven bar-by-bar loop, a venue-accurate cost
model, and a data layer where asking for a bar that has not closed yet is a
structural error rather than a silent wrong number.

[![tests](https://github.com/maxjkor8-afk/trading-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/maxjkor8-afk/trading-lab/actions/workflows/tests.yml)

Python 3.11+. **Zero third-party runtime dependencies** — standard library only,
including the data fetcher. `pytest` is needed for the tests and nothing else.
That is deliberate: the one component whose correctness everything else rests on
should not depend on a dependency tree, and it means the harness runs anywhere a
bare interpreter does.

---

## Does it measure correctly?

That question has to be answered before any strategy result out of a harness is
worth reading, so it is answered first here.

Buy-and-hold on real Coinbase BTC-USD daily bars for calendar 2023, with costs
zeroed, returns **+155.7978%**. That is exactly `42,288.06 / 16,531.83 - 1` —
the final close over the entry open, checkable by hand from two cells of the
committed CSV.

Three things make that a real check rather than the harness marking its own
homework:

| Check | How it is independent |
|---|---|
| **Hand calculation** | The test reads the two prices straight out of the CSV with the `csv` module, not through the harness. If the loader mis-parsed the file, the expected value is still right and the test still catches it. |
| **Published figures** | Bitcoin's calendar-2023 USD return is independently reported as **+155.41%** ([DQYDJ](https://dqydj.com/2023-bitcoin-return/)), **+155.5%** ([ChartRow](https://chartrow.com/bitcoin/returns)) and **+156%** ([Look Into Bitcoin](https://www.lookintobitcoin.com/blog/a-decade-of-bitcoin-annual-returns-and-insights-for-investors/)). The harness lands inside that band. |
| **A second, unrelated venue** | The same window run against Bitstamp BTC/USD returns **+155.64%** through the same code path. A harness bug moves both venues; a data problem at one venue moves only one. Agreement across both is what separates those cases. |

With the pessimistic cost defaults switched on instead of zeroed, the same run
returns +155.41% — the difference being fees and slippage on a single entry.

The full gate is in
[`tests/test_acceptance_buy_and_hold.py`](tests/test_acceptance_buy_and_hold.py).

---

## Quickstart

Three commands on a clean machine. No network access needed.

```bash
python -m pip install -r requirements-dev.txt    # pytest, the only dependency
python -m pytest                                 # 159 tests, all offline
python scripts/run_backtest.py --data data/BTC-USD-1d-2023.csv --symbol BTC-USD
```

No build step and no install step — the scripts and tests put the repository
root on `sys.path` themselves. The CSVs in `data/` are committed so the
acceptance gate runs offline and deterministically.

The third command prints:

```
strategy          buy_and_hold
BTC-USD  [fees=binance/spot maker=10.0bps taker=10.0bps, slippage=LinearBookSlippage, funding=adverse 8:00:00, assume=taker]
period            2023-01-01 -> 2024-01-01 (366 bars)
equity            10,000.00 -> 25,540.95
cumulative return 155.41%
sharpe            2.35
sortino           4.11
max drawdown      20.08%
trades            0
turnover          0.57x equity
net pnl           15,540.95
gross pnl         15,556.13
  fees            9.99
  slippage        5.19
  funding         0.00
  total costs     15.18
costs / gross pnl 0.10%
```

### What this is not

No LLM code. No exchange connection. No API keys. No order placement of any
kind. Data comes from public historical endpoints over read-only HTTP GETs, and
the committed CSVs mean you never have to make even those requests.

---

## Why build the instrument first

Reported results for LLM trading agents are inflated by three specific things:

1. **Look-ahead leakage** — the backtest shows the strategy data it could not
   have had at decision time.
2. **Optimistic execution** — fills assumed at prices you would not get.
3. **Risk mandates described but not enforced** — the prompt says "be
   conservative"; nothing in the code makes it so.

The published [TradingAgents](https://arxiv.org/abs/2412.20138) paper reports a
Sharpe of 8.21 on a three-month equity backtest with no transaction cost model.
Renaissance's Medallion fund ran roughly 2.5 net over decades. A Sharpe of 8 is
not a result, it is a diagnostic of a bad test.

This harness attacks items 1 and 2 structurally — not by convention or by code
review, but by making the mistake raise. Item 3 is risk-limit code that belongs
outside any agent loop, and it is not in this repository.

---

## How the no-look-ahead guarantee is enforced

Four mechanisms, in order of how much work they do.

### 1. A bar is invisible until it closes

`Bar` carries both `open_time` and `close_time`, and visibility is governed by
`close_time`. A daily candle labelled 2023-01-01 is not knowable at
2023-01-01 00:00 — its high, low and close are only settled at 2023-01-02 00:00.

Treating the open timestamp as the moment of knowledge is the most common
look-ahead bug in hand-rolled backtests, and it is not expressible here: the
loader requires both timestamps, and every visibility decision uses the close.

### 2. The clock owns "now", and only the engine can move it

`Clock` holds a single `now`. It advances strictly forward (`advance_to` refuses
backwards *and* sideways moves) and rejects naive datetimes outright, because
guessing a timezone is how a bar ends up on the wrong side of a funding
timestamp.

Crucially, the data API is not parameterised by a caller-supplied timestamp.
`PointInTimeFeed.view(symbol)` takes **no** `ts` argument. There is no
`feed.get(symbol, some_time)` for a strategy to pass the wrong thing to — the
clock decides what exists, not the caller.

### 3. Every data access is bounds-checked against the clock, at access time

`PointInTimeView` wraps the series and the clock, and resolves the visible
cursor on *every* access rather than caching it — so it cannot go stale. Its
`__len__` is "bars known right now", and iteration and slicing stop at the
present:

- `view[2]` when two bars have closed → `LookAheadError`
- `view.at(+1)` → `LookAheadError`
- `view[999]` → `LookAheadError`, **not** `IndexError`
- `view.history(20)` with 12 bars available → `LookAheadError`

That last one matters as much as the obvious cases. Silently returning 12 bars
to a 20-period indicator produces a plausible, wrong number, which is the worst
possible failure mode. And raising `LookAheadError` rather than `IndexError`
means the failure names itself instead of being swallowed by a bare `except`.

### 4. Orders cannot fill at a price the strategy has seen

The loop processes each bar in this order:

1. advance the clock to the bar's **close** (the bar is now known)
2. fill orders queued on *previous* bars, at **this** bar's open
3. settle funding timestamps crossed since the previous bar close
4. record the equity point, marked at this bar's close
5. **call the strategy**, which sees bars `0..i` and may queue orders

Step 5 happening last is the whole trick: a decision taken on the close of bar
`i` cannot execute before the open of bar `i+1`. There is deliberately **no
option** to fill at the close of the current bar. Offering that as a flag would
be offering a way to quietly produce the wrong number.

One consequence worth stating: a strategy cannot know which bar is the last one
without look-ahead, so "sell everything at the end" is not a decision a strategy
can make. An order queued on the final bar has no next bar to fill on, and the
result reports it as unfilled rather than pretending.

### The honest caveat

Python has no enforceable private state. A determined caller can reach
`view._series`. The guarantee is that **no public API returns future data**, that
**every access is bounds-checked**, and that **tests prove the obvious leak
attempts raise** — including an end-to-end cheating strategy that must crash the
run rather than post a great Sharpe. That converts leakage from an easy accident
into a deliberate act that shows up in review.

What this cannot fix is the deeper form of failure mode #1. A pretrained LLM
asked about BTC in March 2023 already knows how 2023 ended, and no amount of
clock discipline removes that. This harness makes the *mechanical* leak
impossible; the *epistemic* leak is not a plumbing problem, and the only honest
test for it is whether forward results land within a sane distance of the
backtest. Point-in-time plumbing is necessary, not sufficient.

---

## Writing a strategy

```python
from harness import BacktestConfig, BarSeries, run_backtest

class BuyTheDip:
    def on_bar(self, ctx):
        closes = ctx.view.closes(20)          # raises if 20 bars have not closed
        if ctx.is_flat and ctx.bar.close < min(closes):
            ctx.buy(percent_of_equity=0.5)    # fills at the NEXT bar's open

series = BarSeries.from_csv("data/BTC-USD-1d-2023.csv", symbol="BTC-USD")
result = run_backtest(series, BuyTheDip(), BacktestConfig(warmup_bars=20))
print(result.report())
```

`ctx.view` is the only market data a strategy ever receives, and `ctx` exposes no
route to anything wider.

---

## The cost model

Pessimistic by default, configurable everywhere. `BacktestConfig()` with no
arguments is not a free lunch, and a forgotten venue argument picks the *worst*
taker fee in the table rather than the best.

**Fees** — published base tiers (VIP 0, no token discount) as understood on
2026-10-18:

| Schedule | Maker | Taker |
|---|---|---|
| `binance_spot` | 0.10% | 0.10% |
| `binance_perp` (USD-M) | 0.02% | 0.05% |
| `bybit_spot` | 0.10% | 0.10% |
| `bybit_perp` (USDT linear) | 0.02% | 0.055% |

These are a starting point, not a source of truth — venues change them and your
tier changes them. Verify before trusting a result. **Taker is assumed** unless
an order says otherwise: a resting maker order only fills when the market comes
to you, which is exactly when you least want it to.

**Slippage** — always adverse; a buy never fills below the reference price. The
default `LinearBookSlippage` walks a uniform-density book: `depth_quote` of
notional rests within `depth_bps` of mid, so filling notional `N` pays
`half_spread + depth_bps * N / (2 * depth_quote)`, and past the band the book is
assumed `overflow_density` times as liquid (default 0.25, four times steeper).
The point is that **size costs something**, so a strategy cannot earn its Sharpe
by pushing arbitrary notional through the top of the book. A `walk_book()`
helper prices against real depth levels, for whenever order books are being
persisted; it is deliberately not wired into order execution yet.

**Funding** — perp positions pay `rate * notional` at every funding timestamp
crossed, every 8h at 00:00/08:00/16:00 UTC by default (Binance and Bybit both
settle BTC perps on that schedule). Positive rate means longs pay. Rates can be
a constant, a per-timestamp mapping, or a callable. The pessimistic default
charges 1bp per 8h — roughly the historical mean of positive BTC funding, about
11% a year, and a deliberately unhelpful assumption for a long-biased strategy.

`always_adverse` (on in `pessimistic_costs()`) charges funding against the
position *whatever its sign*. That is **not** how a venue works — it is a
worst-case switch so that a result cannot depend on having been on the receiving
side of funding. If a strategy only works with it off, say so when reporting the
number.

No real funding history is bundled: Binance's funding endpoint returns HTTP 451
from US IP addresses and Bybit's returned 403. The funding *maths* is
unit-tested against hand calculations.

---

## Metrics

Cumulative and annualised return, Sharpe, Sortino, max drawdown, win rate,
turnover, and the one that matters most:

**`cost_ratio` — total costs paid as a share of gross PnL.** It is how you see a
strategy that is only profitable before fees, which is the usual condition of a
strategy that looks profitable. Above 1 means costs ate more than the edge
produced. There is a test for exactly that case: a +0.1% move cannot pay a 0.2%
round trip, and the ratio comes out at 2.0 with gross PnL positive and net PnL
negative.

Conventions, since they change the numbers:

- **365 periods per year**, inferred from bar spacing. Crypto trades every day;
  the 252 of an equities harness is wrong here.
- **Sharpe and Sortino use sample stdev (ddof=1)**, annualised by `sqrt(periods)`.
- **Win rate is over closed round trips**, not winning days — the share of
  *trades* that made money net of costs. It does not flatter a strategy that
  holds a loser through many bars.
- **Anything uncomputable returns `None`, never `0.0` and never `inf`.** A
  Sharpe of `0.0` reads as "flat"; `None` reads as "not enough data", which is
  the truth when a run has two bars in it. A zero-variance return series gets
  `None` too, not infinity.
- A run whose equity touches zero raises rather than reporting statistics
  through the blowup.

---

## Running backtests

```bash
python scripts/run_backtest.py --data data/BTC-USD-1d-2023.csv --symbol BTC-USD
```

| Flag | Effect |
|---|---|
| `--zero-costs` | control run: no fees, slippage or funding. For checking arithmetic, **not** for evaluating a strategy — the output says so. |
| `--instrument perp` | enables funding settlement and allows shorts |
| `--venue binance\|bybit` | pick a published fee schedule; default is the worst in the table |
| `--half-spread-bps`, `--depth-quote`, `--depth-bps` | slippage parameters |
| `--funding-rate` | per-8h rate for perp runs |
| `--equity-out path.csv` | write the equity curve |

### Bundled data

| File | Window | Source |
|---|---|---|
| `data/BTC-USD-1d-2023.csv` | 2022-12-31 → 2023-12-31 | Coinbase BTC-USD |
| `data/BTCUSD-1d-2023-bitstamp.csv` | 2022-12-31 → 2023-12-31 | Bitstamp BTC/USD — the cross-check venue |
| `data/BTC-USD-1d-2022.csv` | 2021-12-31 → 2022-12-31 | Coinbase BTC-USD — a down year, for checking a strategy that only works in a bull market |

Each has a `.meta.json` sidecar recording source, symbol, bar count, window and
fetch time. Each series starts one bar before the measured window: that first
bar is the warm-up, the strategy decides on its close, and the fill lands on the
open of the first in-window bar. Fetching more data is a single command — see
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## Tests

```bash
python -m pytest
```

159 tests, no network access, well under a second. The suite is organised around
what could silently go wrong:

| File | Covers |
|---|---|
| `test_leakage_guard.py` | every deliberate look-ahead attempt raises, including an end-to-end cheating strategy that must crash the run rather than post a great Sharpe |
| `test_costs_fees.py` | fee arithmetic, the venue table, pessimistic defaults, the adverse-only invariant |
| `test_costs_slippage.py` | size-dependence, direction, book walking, thin-book refusal |
| `test_funding.py` | which timestamps settle, half-open intervals, sign convention, end-to-end settlement counts |
| `test_metrics.py` | Sharpe/Sortino/drawdown against hand arithmetic, cost ratio on a hand-calculable round trip |
| `test_engine.py` | fill timing, sizing, rejections, position accounting including flips |
| `test_acceptance_buy_and_hold.py` | **the gate** |

Expected values are written out as explicit arithmetic on the sample rather than
read back from the implementation, so the tests check the formulas instead of
echoing them.

---

## Known limitations

Stated plainly, because a harness whose gaps are undocumented is a harness that
will mislead someone later.

- **No margin or liquidation model.** Nothing stops a perp position that would
  have been liquidated. Perp leverage results are optimistic in that specific
  way.
- **One instrument per run.** The data layer is multi-symbol; the portfolio is
  not. No cross-asset or portfolio-level backtests.
- **Market orders only.** No limit orders, no partial fills against a real book
  in the main loop (`walk_book` exists but is not wired into order execution),
  no queue position modelling.
- **Slippage is a model, not a measurement.** `LinearBookSlippage` is a
  defensible shape with invented parameters. It makes size cost something; it
  does not tell you what size actually costs on Binance at 03:00 UTC.
- **Funding is marked at the bar close** for every settlement inside that bar.
  With daily bars that is up to 8h of drift from the true mark.
- **Bundled data is daily bars from one venue** plus one cross-check. No
  intraday bars, no order books, no funding history.
- **No defence against epistemic leakage.** See the caveat above. This is the
  limitation that most constrains what a backtest here can prove about an LLM
  strategy.
- **Costs dominate at small account sizes.** Worth saying out loud, since this
  harness is what reveals it: a daily-rebalancing strategy on a few thousand
  dollars of capital pays a large fraction of that capital per year in fees. The
  `cost_ratio` metric exists to make that unmissable, not to make it go away.

---

## Repository layout

```
harness/
  clock.py        monotonic UTC clock; the only thing that moves "now"
  data.py         Bar, BarSeries, PointInTimeFeed, PointInTimeView, LookAheadError
  costs.py        fee schedules, slippage models, funding
  engine.py       the bar-by-bar loop and next-open fills
  portfolio.py    cash, position and equity accounting
  metrics.py      returns, Sharpe, Sortino, drawdown, cost_ratio
  strategies/     buy_and_hold reference strategy
scripts/
  fetch_ohlcv.py  stdlib-only OHLCV fetcher - the only networked code here
  run_backtest.py CLI
tests/            159 tests, offline
data/             committed CSVs plus .meta.json sidecars
```

## Background

[RESEARCH.md](RESEARCH.md) is the literature review this design came out of:
what the published LLM-trading papers actually report, why those numbers are not
credible, and what a staged build would have to prove at each step. It is
background reading, not required to use the harness.

[CONTRIBUTING.md](CONTRIBUTING.md) has the development setup, the ground rules a
change has to respect, and how to regenerate the bundled data.

## License

[MIT](LICENSE).
