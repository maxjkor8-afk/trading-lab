"""Perp funding: which timestamps settle, and who pays whom.

Funding is the cost most often left out of a crypto backtest, and for a
long-biased strategy in a bull market it is a persistent drag that runs in the
tens of percent a year. It is also the signal RESEARCH.md section 6 argues is
the cleanest real-time crowding proxy, so getting its sign right matters twice.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from conftest import flat_series

from harness.costs import CostModel, FundingModel, NoSlippage, ZERO_FEES
from harness.engine import BacktestConfig, run_backtest
from harness.enums import InstrumentType, Side

UTC = timezone.utc


def ts(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def zero_fee_funding(rate, **kwargs) -> CostModel:
    """Isolate funding: no fees, no slippage, just the funding stream."""
    return CostModel(
        fees=ZERO_FEES,
        slippage=NoSlippage(),
        funding=FundingModel(rates=rate, **kwargs),
    )


# -- which timestamps settle ------------------------------------------------


def test_funding_settles_three_times_a_day_at_00_08_16_utc():
    """Binance and Bybit both settle BTC perps every 8h from midnight UTC."""
    model = FundingModel(rates=0.0001)
    events = model.events(ts(2023, 1, 2), ts(2023, 1, 3))
    assert [e.ts for e in events] == [
        ts(2023, 1, 2, 8),
        ts(2023, 1, 2, 16),
        ts(2023, 1, 3, 0),
    ]


def test_interval_is_half_open_so_nothing_is_counted_twice():
    """Walking bar to bar must never settle the same funding twice.

    Day 1 ends on the 00:00 stamp; day 2 must start after it.
    """
    model = FundingModel(rates=0.0001)
    day1 = model.events(ts(2023, 1, 2), ts(2023, 1, 3))
    day2 = model.events(ts(2023, 1, 3), ts(2023, 1, 4))
    stamps = [e.ts for e in day1] + [e.ts for e in day2]
    assert len(stamps) == len(set(stamps)) == 6
    assert ts(2023, 1, 3) in [e.ts for e in day1]
    assert ts(2023, 1, 3) not in [e.ts for e in day2]


def test_no_events_when_no_time_passes():
    model = FundingModel(rates=0.0001)
    assert model.events(ts(2023, 1, 2), ts(2023, 1, 2)) == ()
    assert model.events(ts(2023, 1, 2), ts(2023, 1, 1)) == ()


def test_partial_window_picks_up_only_the_stamps_inside_it():
    model = FundingModel(rates=0.0001)
    events = model.events(ts(2023, 1, 2, 9), ts(2023, 1, 2, 23))
    assert [e.ts for e in events] == [ts(2023, 1, 2, 16)]


def test_a_window_landing_exactly_on_a_stamp_includes_it():
    model = FundingModel(rates=0.0001)
    events = model.events(ts(2023, 1, 2, 1), ts(2023, 1, 2, 8))
    assert [e.ts for e in events] == [ts(2023, 1, 2, 8)]


def test_interval_and_anchor_are_configurable():
    """Not every venue settles on the same schedule."""
    model = FundingModel(
        rates=0.0001, interval=timedelta(hours=4), anchor=ts(1970, 1, 1, 2)
    )
    events = model.events(ts(2023, 1, 2), ts(2023, 1, 2, 12))
    assert [e.ts for e in events] == [
        ts(2023, 1, 2, 2),
        ts(2023, 1, 2, 6),
        ts(2023, 1, 2, 10),
    ]


def test_non_positive_interval_is_rejected():
    with pytest.raises(ValueError, match="interval must be positive"):
        FundingModel(rates=0.0, interval=timedelta(0))


# -- rate sources -----------------------------------------------------------


def test_rate_can_be_a_constant():
    assert FundingModel(rates=0.0003).rate_at(ts(2023, 5, 1)) == pytest.approx(0.0003)


def test_rate_can_be_a_per_timestamp_mapping():
    model = FundingModel(rates={ts(2023, 1, 2, 8): 0.01}, missing_rate=0.0)
    assert model.rate_at(ts(2023, 1, 2, 8)) == pytest.approx(0.01)
    assert model.rate_at(ts(2023, 1, 2, 16)) == 0.0


def test_missing_rate_is_explicit_rather_than_assumed_zero():
    """A gap in fetched funding history should be a visible choice."""
    model = FundingModel(rates={}, missing_rate=0.0005)
    assert model.rate_at(ts(2023, 1, 2, 8)) == pytest.approx(0.0005)


def test_rate_can_be_a_callable():
    model = FundingModel(rates=lambda t: 0.0001 * t.hour)
    assert model.rate_at(ts(2023, 1, 2, 16)) == pytest.approx(0.0016)


# -- sign convention --------------------------------------------------------


def test_positive_rate_means_longs_pay():
    """1bp on $10,000 long = $1 out. Negative return value means you paid."""
    model = FundingModel(rates=1e-4)
    assert model.payment(quantity=100.0, mark_price=100.0, rate=1e-4) == pytest.approx(-1.0)


def test_positive_rate_means_shorts_receive():
    model = FundingModel(rates=1e-4)
    assert model.payment(quantity=-100.0, mark_price=100.0, rate=1e-4) == pytest.approx(1.0)


def test_negative_rate_flips_who_pays():
    model = FundingModel(rates=-1e-4)
    assert model.payment(100.0, 100.0, -1e-4) == pytest.approx(1.0)
    assert model.payment(-100.0, 100.0, -1e-4) == pytest.approx(-1.0)


def test_flat_position_pays_no_funding():
    assert FundingModel(rates=1e-3).payment(0.0, 100.0, 1e-3) == 0.0


def test_always_adverse_charges_both_directions():
    """The pessimistic switch: a result must not depend on being on the right side.

    This is not venue mechanics - it is a worst case. Documented as such.
    """
    model = FundingModel(rates=1e-4, always_adverse=True)
    assert model.payment(100.0, 100.0, 1e-4) == pytest.approx(-1.0)
    assert model.payment(-100.0, 100.0, 1e-4) == pytest.approx(-1.0)
    assert model.payment(100.0, 100.0, -1e-4) == pytest.approx(-1.0)


# -- integration with the loop ---------------------------------------------


def test_perp_held_for_two_days_pays_six_fundings():
    """Hand calculation, end to end.

    Flat price 100, $10,000 of equity, entry on bar 1's open gives 100 units
    ($10,000 notional). Rate 10bp per 8h, so each settlement costs
    0.001 * 10,000 = $10. Two full days held after entry = 6 settlements = $60.
    """
    series = flat_series(3, price=100.0)
    config = BacktestConfig(
        initial_cash=10_000.0,
        instrument=InstrumentType.PERP,
        costs=zero_fee_funding(0.001),
    )

    def buy_once(ctx):
        if ctx.is_flat and not ctx.pending_orders:
            ctx.buy(percent_of_equity=1.0)

    result = run_backtest(series, buy_once, config)
    assert len(result.fills) == 1
    assert result.portfolio.quantity == pytest.approx(100.0)
    assert result.portfolio.funding_paid == pytest.approx(60.0)
    assert result.portfolio.fees_paid == 0.0
    assert result.metrics.final_equity == pytest.approx(9_940.0)
    assert result.metrics.cumulative_return == pytest.approx(-0.006)


def test_short_perp_receives_funding_when_the_rate_is_positive():
    series = flat_series(3, price=100.0)
    config = BacktestConfig(
        initial_cash=10_000.0,
        instrument=InstrumentType.PERP,
        costs=zero_fee_funding(0.001),
    )

    def sell_once(ctx):
        if ctx.is_flat and not ctx.pending_orders:
            ctx.sell(percent_of_equity=1.0)

    result = run_backtest(series, sell_once, config)
    assert result.portfolio.quantity == pytest.approx(-100.0)
    assert result.portfolio.funding_paid == pytest.approx(-60.0)  # received
    assert result.metrics.final_equity == pytest.approx(10_060.0)


def test_funding_is_not_charged_before_the_position_exists():
    """Entry is on bar 1's open, so bar 0 settles nothing."""
    series = flat_series(2, price=100.0)
    config = BacktestConfig(
        initial_cash=10_000.0,
        instrument=InstrumentType.PERP,
        costs=zero_fee_funding(0.001),
    )

    def buy_once(ctx):
        if ctx.is_flat and not ctx.pending_orders:
            ctx.buy(percent_of_equity=1.0)

    result = run_backtest(series, buy_once, config)
    # One bar held after entry = 3 settlements, not 6.
    assert result.portfolio.funding_paid == pytest.approx(30.0)


def test_spot_never_settles_funding():
    """Spot has no funding leg, so a non-zero rate on a spot run is inert."""
    series = flat_series(3, price=100.0)
    config = BacktestConfig(
        initial_cash=10_000.0,
        instrument=InstrumentType.SPOT,
        costs=zero_fee_funding(0.001),
    )

    def buy_once(ctx):
        if ctx.is_flat and not ctx.pending_orders:
            ctx.buy(percent_of_equity=1.0)

    result = run_backtest(series, buy_once, config)
    assert result.portfolio.funding_paid == 0.0
    assert result.metrics.final_equity == pytest.approx(10_000.0)


def test_applying_funding_to_a_spot_portfolio_is_an_error():
    from harness.portfolio import Portfolio

    spot = Portfolio(1_000.0, InstrumentType.SPOT)
    with pytest.raises(ValueError, match="perpetuals only"):
        spot.apply_funding(ts(2023, 1, 1), -1.0, 100.0)


def test_always_adverse_funding_costs_a_short_too():
    """With the pessimistic default on, the short no longer gets paid."""
    series = flat_series(3, price=100.0)
    config = BacktestConfig(
        initial_cash=10_000.0,
        instrument=InstrumentType.PERP,
        costs=zero_fee_funding(0.001, always_adverse=True),
    )

    def sell_once(ctx):
        if ctx.is_flat and not ctx.pending_orders:
            ctx.sell(percent_of_equity=1.0)

    result = run_backtest(series, sell_once, config)
    assert result.portfolio.funding_paid == pytest.approx(60.0)
    assert result.metrics.final_equity == pytest.approx(9_940.0)
