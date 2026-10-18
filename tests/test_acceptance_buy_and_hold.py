"""THE ACCEPTANCE GATE (RESEARCH.md section 7, Stage 0).

    Gate: the harness reproduces a known buy-and-hold return correctly.

If these tests do not pass, the harness is not trustworthy and nothing should be
built on top of it. Every later stage's numbers are downstream of this file.

What is being verified, and how it is independent:

1. **Hand calculation.** With costs zeroed, buy-and-hold must return exactly
   `final_close / entry_open - 1`. The test reads those two numbers straight out
   of the CSV with the `csv` module - not through the harness - so the harness
   cannot mark its own homework.

2. **A published figure.** Bitcoin's calendar-2023 return in USD is independently
   reported as +155.41% (DQYDJ), +155.5% (ChartRow) and +156% (Look Into
   Bitcoin). Small differences between those come from which venue's prices and
   which daily snapshot each uses. The harness lands at +155.80% on Coinbase
   BTC-USD, inside that band.

3. **Two unrelated venues.** The same window is checked against Coinbase BTC-USD
   and Bitstamp BTC/USD. A bug in the harness moves both; a bug in one venue's
   data moves only one. Agreement across both is what separates those cases.

The entry convention, stated once: the series starts with a 2022-12-31 warm-up
bar. The strategy decides on that bar's close and fills at the open of
2023-01-01, so the measured window is exactly the 2023 calendar year.
"""

from __future__ import annotations

import csv

import pytest
from conftest import BTC_2023_CROSSCHECK_CSV, BTC_2023_CSV

from harness.costs import ZERO_COSTS, pessimistic_costs
from harness.data import BarSeries
from harness.engine import BacktestConfig, run_backtest
from harness.enums import InstrumentType
from harness.strategies import BuyAndHold

INITIAL_CASH = 10_000.0

# Published calendar-2023 BTC/USD returns: 155.41% (DQYDJ), 155.5% (ChartRow),
# 156% (Look Into Bitcoin). The band is wide enough to absorb venue and
# snapshot differences and far too tight to absorb a harness bug.
PUBLISHED_LOW = 1.50
PUBLISHED_HIGH = 1.61


def hand_calculated_return(csv_path) -> tuple[float, float, float]:
    """Read the CSV directly and work the answer out by hand.

    Deliberately does not use BarSeries: if the loader mis-parsed the file, this
    function must still produce the right number so the test can catch it.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    entry_open = float(rows[1]["open"])  # bar 0 is the warm-up bar
    final_close = float(rows[-1]["close"])
    return final_close / entry_open - 1.0, entry_open, final_close


def zero_cost_run(series: BarSeries):
    return run_backtest(
        series,
        BuyAndHold(),
        BacktestConfig(initial_cash=INITIAL_CASH, costs=ZERO_COSTS),
    )


# -- 1. the hand calculation -----------------------------------------------


def test_harness_reproduces_the_hand_calculated_buy_and_hold_return(btc_2023):
    """The gate. Zero costs, real data, exact agreement with two CSV cells."""
    expected, entry_open, final_close = hand_calculated_return(BTC_2023_CSV)
    result = zero_cost_run(btc_2023)

    assert result.metrics.cumulative_return == pytest.approx(expected, rel=1e-9)
    assert result.metrics.final_equity == pytest.approx(
        INITIAL_CASH * (1.0 + expected), rel=1e-9
    )
    # And the pieces it was built from.
    assert result.fills[0].price == pytest.approx(entry_open)
    assert result.equity_curve[-1].mark_price == pytest.approx(final_close)


def test_the_number_is_the_one_we_expect_to_the_fourth_decimal(btc_2023):
    """A hard-coded regression anchor: 42,288.06 / 16,531.83 - 1."""
    result = zero_cost_run(btc_2023)
    assert result.metrics.cumulative_return == pytest.approx(1.5579781549, rel=1e-9)
    assert result.metrics.final_equity == pytest.approx(25_579.781549, rel=1e-9)


def test_quantity_held_is_cash_divided_by_the_entry_price(btc_2023):
    """With no fee there is nothing to solve for: 10,000 / 16,531.83 units."""
    _, entry_open, _ = hand_calculated_return(BTC_2023_CSV)
    result = zero_cost_run(btc_2023)
    assert result.portfolio.quantity == pytest.approx(INITIAL_CASH / entry_open, rel=1e-12)
    assert result.portfolio.cash == pytest.approx(0.0, abs=1e-6)


# -- 2. the published figure ------------------------------------------------


def test_return_sits_inside_the_independently_published_band(btc_2023):
    """+155.8% against published figures of +155.41%, +155.5% and +156%."""
    result = zero_cost_run(btc_2023)
    assert PUBLISHED_LOW < result.metrics.cumulative_return < PUBLISHED_HIGH


def test_the_measured_window_is_the_2023_calendar_year(btc_2023):
    """Entry on 2023-01-01's open, final mark on 2023-12-31's close."""
    result = zero_cost_run(btc_2023)
    fill = result.fills[0]
    assert fill.ts.year == 2023 and fill.ts.month == 1 and fill.ts.day == 1
    assert btc_2023[-1].open_time.year == 2023
    assert btc_2023[-1].open_time.month == 12
    assert btc_2023[-1].open_time.day == 31
    assert len(btc_2023) == 366  # 2022-12-31 warm-up + 365 days of 2023


# -- 3. the second venue ----------------------------------------------------


def test_an_unrelated_venue_reproduces_the_same_return(btc_2023_crosscheck):
    """Bitstamp BTC/USD must land on its own hand calculation, and near Coinbase.

    A harness bug moves both venues; a data problem moves one. Checking both is
    how those two cases are told apart.
    """
    expected, _, _ = hand_calculated_return(BTC_2023_CROSSCHECK_CSV)
    result = zero_cost_run(btc_2023_crosscheck)
    assert result.metrics.cumulative_return == pytest.approx(expected, rel=1e-9)
    assert PUBLISHED_LOW < result.metrics.cumulative_return < PUBLISHED_HIGH


def test_the_two_venues_agree_with_each_other(btc_2023, btc_2023_crosscheck):
    coinbase = zero_cost_run(btc_2023).metrics.cumulative_return
    bitstamp = zero_cost_run(btc_2023_crosscheck).metrics.cumulative_return
    assert abs(coinbase - bitstamp) < 0.01  # within 1 percentage point


# -- no look-ahead on real data --------------------------------------------


def test_the_entry_is_the_next_open_not_the_observed_close(btc_2023):
    """The leak this harness is built to prevent, checked on real prices.

    Note how small the difference is on daily data: a daily open sits very close
    to the previous close, so leaking one bar moves the headline return by only
    about 0.02 percentage points here. That is exactly why the guarantee has to
    be structural. On a strategy that trades often, or on intraday bars, the
    same one-bar leak is what produces a Sharpe of 8.
    """
    with open(BTC_2023_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    observed_close = float(rows[0]["close"])  # what the strategy could see
    next_open = float(rows[1]["open"])  # what it actually paid
    assert observed_close != next_open

    result = zero_cost_run(btc_2023)
    assert result.fills[0].price == pytest.approx(next_open)

    leaked = float(rows[-1]["close"]) / observed_close - 1.0
    assert result.metrics.cumulative_return != pytest.approx(leaked, rel=1e-5)


# -- costs move the number in the right direction --------------------------


def test_pessimistic_costs_reduce_the_return_by_exactly_one_entry(btc_2023):
    """Buy-and-hold trades once, so its entire cost is one fee plus one slippage.

    This is the cheapest available check that the cost model is wired into the
    loop rather than merely configured: the drag has to equal the recorded cost
    of the single fill, to the cent.
    """
    costed = run_backtest(
        btc_2023,
        BuyAndHold(),
        BacktestConfig(
            initial_cash=INITIAL_CASH,
            instrument=InstrumentType.SPOT,
            costs=pessimistic_costs("binance", InstrumentType.SPOT),
        ),
    )
    free = zero_cost_run(btc_2023)

    assert costed.metrics.cumulative_return < free.metrics.cumulative_return
    assert len(costed.fills) == 1
    fill = costed.fills[0]
    assert fill.price > fill.reference_price  # paid up to get in
    assert costed.metrics.total_costs == pytest.approx(fill.fee + fill.slippage_cost)
    # 10bps fee plus ~5bps of spread and impact on a single entry: a drag of
    # well under 1% of gross PnL over a year.
    assert 0.0 < costed.metrics.cost_ratio < 0.01


def test_spot_buy_and_hold_pays_no_funding(btc_2023):
    costed = run_backtest(
        btc_2023,
        BuyAndHold(),
        BacktestConfig(initial_cash=INITIAL_CASH, costs=pessimistic_costs()),
    )
    assert costed.portfolio.funding_paid == 0.0


def test_a_perp_version_is_dragged_down_by_a_year_of_funding(btc_2023):
    """The same year on a perp, paying 1bp per 8h, is materially worse.

    ~1,095 settlements a year at 1bp is roughly 11% of notional, and it compounds
    against a growing position. Leaving funding out of a perp backtest is not a
    rounding error.
    """
    perp = run_backtest(
        btc_2023,
        BuyAndHold(),
        BacktestConfig(
            initial_cash=INITIAL_CASH,
            instrument=InstrumentType.PERP,
            costs=pessimistic_costs("binance", InstrumentType.PERP),
        ),
    )
    spot = zero_cost_run(btc_2023)
    assert perp.portfolio.funding_paid > 0
    assert perp.metrics.cumulative_return < spot.metrics.cumulative_return
    assert perp.metrics.funding_paid > perp.metrics.fees_paid


# -- metric sanity on real data --------------------------------------------


def test_metrics_on_the_real_series_are_plausible(btc_2023):
    """Shape checks on a year everyone has an intuition for.

    BTC in 2023 roughly tripled with a drawdown around 20%. A buy-and-hold
    Sharpe of about 2.3 is what that looks like.
    """
    m = zero_cost_run(btc_2023).metrics
    assert m.bars == 366
    assert m.periods_per_year == pytest.approx(365.0)
    assert 0.05 < m.max_drawdown < 0.40
    assert m.sharpe is not None and 1.0 < m.sharpe < 4.0
    assert m.sortino is not None and m.sortino > m.sharpe
    # Annualised and cumulative coincide because the window is one year.
    assert m.annualized_return == pytest.approx(m.cumulative_return, rel=1e-3)


def test_a_sharpe_above_five_would_mean_the_harness_is_broken(btc_2023):
    """A standing guard, pointed at the failure mode in RESEARCH.md section 2a.

    The TradingAgents paper reports 8.21 on a 3-month equity window with no cost
    model. On a daily crypto series, a buy-and-hold Sharpe anywhere near that
    would mean the equity curve is not being built from real price changes.
    """
    assert zero_cost_run(btc_2023).metrics.sharpe < 5.0


def test_buy_and_hold_never_closes_a_trade(btc_2023):
    """It holds, so there is no round trip and no win rate to report."""
    result = zero_cost_run(btc_2023)
    assert result.metrics.trades == 0
    assert result.metrics.win_rate is None
    assert result.portfolio.quantity > 0
    assert len(result.fills) == 1


def test_turnover_reflects_a_single_entry(btc_2023):
    """One entry against a rising average equity: well under 1x."""
    result = zero_cost_run(btc_2023)
    assert 0.0 < result.metrics.turnover < 1.0


def test_equity_curve_is_continuous_and_starts_at_the_initial_cash(btc_2023):
    result = zero_cost_run(btc_2023)
    curve = result.equity_curve
    assert len(curve) == len(btc_2023)
    assert curve[0].equity == pytest.approx(INITIAL_CASH)
    assert curve[0].position == 0.0  # not yet filled on the warm-up bar
    assert all(p.equity > 0 for p in curve)
    assert all(a.ts < b.ts for a, b in zip(curve, curve[1:]))
    # After entry the position never changes again.
    positions = {round(p.position, 12) for p in curve[1:]}
    assert len(positions) == 1
