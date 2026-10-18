"""The event loop: when orders fill, what gets recorded, and what is refused."""

from __future__ import annotations

import pytest
from conftest import flat_series, make_series

from harness.costs import (
    CostModel,
    FeeSchedule,
    FixedSpreadSlippage,
    LinearBookSlippage,
    NoSlippage,
    ZERO_COSTS,
    ZERO_FUNDING,
)
from harness.engine import BacktestConfig, FillTiming, run_backtest
from harness.enums import InstrumentType, Liquidity, Side
from harness.portfolio import InsufficientFunds, Order, Portfolio


def buy_on_first_bar(**order_kwargs):
    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(**order_kwargs)

    return strategy


# -- fill timing ------------------------------------------------------------


def test_order_fills_at_the_next_bar_open():
    """The anti-leak rule, stated as a test.

    The order is decided on bar 0's close (price 100) and must fill at bar 1's
    open (105) - not at 100, which the strategy had already seen.
    """
    series = make_series([(100.0, 100.0), (105.0, 110.0), (110.0, 110.0)])
    result = run_backtest(
        series,
        buy_on_first_bar(quantity=1.0),
        BacktestConfig(costs=ZERO_COSTS),
    )
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.price == pytest.approx(105.0)
    assert fill.reference_price == pytest.approx(105.0)
    assert fill.ts == series[1].open_time


def test_no_order_can_fill_on_the_bar_that_produced_it():
    """Equity on the decision bar must still show a flat position."""
    series = make_series([(100.0, 100.0), (105.0, 105.0)])
    result = run_backtest(
        series, buy_on_first_bar(quantity=1.0), BacktestConfig(costs=ZERO_COSTS)
    )
    assert result.equity_curve[0].position == 0.0
    assert result.equity_curve[1].position == pytest.approx(1.0)


def test_next_close_timing_fills_at_the_following_close():
    series = make_series([(100.0, 100.0), (105.0, 110.0), (110.0, 110.0)])
    result = run_backtest(
        series,
        buy_on_first_bar(quantity=1.0),
        BacktestConfig(costs=ZERO_COSTS, fill_timing=FillTiming.NEXT_CLOSE),
    )
    assert result.fills[0].price == pytest.approx(110.0)
    assert result.fills[0].ts == series[1].close_time


def test_an_order_queued_on_the_final_bar_never_fills():
    """There is no next bar to fill it on, and the result says so."""
    series = flat_series(3)

    def always_buy(ctx):
        ctx.buy(quantity=0.01)

    result = run_backtest(series, always_buy, BacktestConfig(costs=ZERO_COSTS))
    assert len(result.fills) == 2  # bars 1 and 2
    assert len(result.unfilled_at_end) == 1
    assert "never filled" in result.report()


def test_a_single_bar_run_is_refused():
    series = flat_series(1)
    with pytest.raises(ValueError, match="at least 2 bars"):
        run_backtest(series, buy_on_first_bar(quantity=1.0))


# -- the equity curve -------------------------------------------------------


def test_equity_curve_has_one_point_per_bar_and_starts_at_initial_cash():
    series = flat_series(5, price=100.0)
    result = run_backtest(
        series, buy_on_first_bar(quantity=1.0), BacktestConfig(initial_cash=5_000.0, costs=ZERO_COSTS)
    )
    assert len(result.equity_curve) == len(series) == 5
    assert result.equity_curve[0].equity == pytest.approx(5_000.0)
    assert [p.ts for p in result.equity_curve] == [b.close_time for b in series]


def test_equity_is_marked_at_each_bar_close():
    series = make_series([(100.0, 100.0), (100.0, 120.0), (120.0, 90.0)])
    result = run_backtest(
        series,
        buy_on_first_bar(quantity=1.0),
        BacktestConfig(initial_cash=1_000.0, costs=ZERO_COSTS),
    )
    # bought 1 unit at bar 1's open of 100, leaving 900 cash
    assert result.equity_curve[1].equity == pytest.approx(900.0 + 120.0)
    assert result.equity_curve[2].equity == pytest.approx(900.0 + 90.0)
    assert result.metrics.max_drawdown == pytest.approx(
        (1020.0 - 990.0) / 1020.0
    )


def test_write_equity_csv_round_trips(tmp_path):
    series = flat_series(3)
    result = run_backtest(series, buy_on_first_bar(quantity=1.0), BacktestConfig(costs=ZERO_COSTS))
    path = result.write_equity_csv(tmp_path / "equity.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "ts,equity,mark_price,position,cash"
    assert len(lines) == 4


# -- sizing -----------------------------------------------------------------


def test_percent_of_equity_solves_for_the_fee_so_cash_is_not_overdrawn():
    """qty = cash / (price * (1 + fee_rate)), not cash / price.

    Ignoring the fee here overdraws the account by exactly the fee on the first
    bar of every fully-invested strategy.
    """
    series = flat_series(3, price=100.0)
    costs = CostModel(
        fees=FeeSchedule(maker_bps=10.0, taker_bps=10.0),
        slippage=NoSlippage(),
        funding=ZERO_FUNDING,
    )
    result = run_backtest(
        series,
        buy_on_first_bar(percent_of_equity=1.0),
        BacktestConfig(initial_cash=10_000.0, costs=costs),
    )
    assert result.portfolio.quantity == pytest.approx(10_000.0 / 100.1)
    assert result.portfolio.cash == pytest.approx(0.0, abs=1e-9)


def test_notional_sizing_divides_by_the_fill_price():
    series = make_series([(100.0, 100.0), (50.0, 50.0), (50.0, 50.0)])
    result = run_backtest(
        series, buy_on_first_bar(notional=1_000.0), BacktestConfig(costs=ZERO_COSTS)
    )
    assert result.portfolio.quantity == pytest.approx(20.0)


def test_sell_percent_of_equity_sells_that_share_of_the_position():
    series = flat_series(5, price=100.0)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(quantity=10.0)
        elif len(ctx.view) == 3:
            ctx.sell(percent_of_equity=0.5)

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert result.portfolio.quantity == pytest.approx(5.0)


def test_order_requires_exactly_one_sizing_field():
    with pytest.raises(ValueError, match="exactly one"):
        Order(side=Side.BUY)
    with pytest.raises(ValueError, match="exactly one"):
        Order(side=Side.BUY, quantity=1.0, notional=100.0)


def test_order_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="must be positive"):
        Order(side=Side.BUY, quantity=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        Order(side=Side.SELL, quantity=-1.0)


def test_percent_of_equity_above_one_is_refused_as_implicit_leverage():
    """Leverage should be asked for explicitly, never arrived at by accident."""
    with pytest.raises(ValueError, match="implies leverage"):
        Order(side=Side.BUY, percent_of_equity=1.5)


# -- rejections and refusals ------------------------------------------------


def test_reduce_only_with_no_position_is_rejected_not_executed():
    series = flat_series(3)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.submit(Order(side=Side.SELL, quantity=1.0, reduce_only=True))

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert result.fills == ()
    assert len(result.rejected) == 1
    assert "reduce_only" in result.rejected[0].reason


def test_a_vanishingly_small_order_is_rejected():
    series = flat_series(3, price=100.0)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(notional=1e-15)

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert result.fills == ()
    assert "zero" in result.rejected[0].reason


def test_spot_cannot_go_short():
    series = flat_series(5, price=100.0)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(quantity=10.0)
        elif len(ctx.view) == 3:
            ctx.sell(quantity=25.0)

    with pytest.raises(ValueError, match="spot cannot go short"):
        run_backtest(
            series, strategy, BacktestConfig(instrument=InstrumentType.SPOT, costs=ZERO_COSTS)
        )


def test_overspending_cash_raises_rather_than_silently_borrowing():
    series = flat_series(3, price=100.0)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(notional=50_000.0)

    with pytest.raises(InsufficientFunds):
        run_backtest(
            series, strategy, BacktestConfig(initial_cash=10_000.0, costs=ZERO_COSTS)
        )


# -- costs flowing through the loop ----------------------------------------


def test_slippage_is_charged_and_recorded():
    series = flat_series(3, price=100.0)
    costs = CostModel(
        fees=FeeSchedule(maker_bps=0.0, taker_bps=0.0),
        slippage=FixedSpreadSlippage(half_spread_bps=20.0),
        funding=ZERO_FUNDING,
    )
    result = run_backtest(
        series, buy_on_first_bar(quantity=2.0), BacktestConfig(costs=costs)
    )
    fill = result.fills[0]
    assert fill.price == pytest.approx(100.2)
    assert fill.slippage_bps == pytest.approx(20.0)
    assert fill.slippage_cost == pytest.approx(0.4)  # 0.2 * 2 units
    assert result.portfolio.slippage_paid == pytest.approx(0.4)
    assert result.metrics.total_costs == pytest.approx(0.4)


def test_size_dependent_slippage_resolves_circular_sizing_conservatively():
    """percent_of_equity sizing and size-dependent slippage are circular.

    Slippage depends on size; size depends on the slipped price. The loop prices
    against the probe quantity, which is computed at the unslipped reference and
    so is always at least the quantity finally filled. Two properties must hold:
    the charged slippage is no better than the exact slippage for the size that
    actually traded, and the order remains exactly affordable.
    """
    series = flat_series(3, price=100.0)
    costs = CostModel(
        fees=FeeSchedule(maker_bps=10.0, taker_bps=10.0),
        slippage=LinearBookSlippage(
            half_spread_bps=5.0, depth_quote=10_000.0, depth_bps=100.0
        ),
        funding=ZERO_FUNDING,
    )
    result = run_backtest(
        series,
        buy_on_first_bar(percent_of_equity=1.0),
        BacktestConfig(initial_cash=10_000.0, costs=costs),
    )
    fill = result.fills[0]
    exact_bps = costs.slippage.slippage_bps(Side.BUY, 100.0, fill.quantity)
    assert fill.price > 100.0
    assert fill.slippage_bps >= exact_bps  # erring pessimistic, by design
    assert result.portfolio.cash == pytest.approx(0.0, abs=1e-9)


def test_taker_is_assumed_unless_the_order_says_otherwise():
    series = flat_series(3, price=100.0)
    costs = CostModel(
        fees=FeeSchedule(maker_bps=2.0, taker_bps=10.0),
        slippage=NoSlippage(),
        funding=ZERO_FUNDING,
    )
    taker = run_backtest(
        series, buy_on_first_bar(quantity=1.0), BacktestConfig(costs=costs)
    )
    assert taker.fills[0].liquidity is Liquidity.TAKER
    assert taker.fills[0].fee == pytest.approx(0.10)

    def maker_strategy(ctx):
        if len(ctx.view) == 1:
            ctx.submit(Order(side=Side.BUY, quantity=1.0, liquidity=Liquidity.MAKER))

    maker = run_backtest(series, maker_strategy, BacktestConfig(costs=costs))
    assert maker.fills[0].fee == pytest.approx(0.02)


# -- position accounting ---------------------------------------------------


def test_perp_position_can_flip_through_flat_in_one_fill():
    """Sell 25 while long 10: one trade closes, a 15-unit short opens."""
    series = make_series(
        [(100.0, 100.0), (100.0, 100.0), (120.0, 120.0), (120.0, 120.0), (120.0, 120.0)]
    )

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(quantity=10.0)
        elif len(ctx.view) == 3:
            ctx.sell(quantity=25.0)

    result = run_backtest(
        series,
        strategy,
        BacktestConfig(
            initial_cash=10_000.0, instrument=InstrumentType.PERP, costs=ZERO_COSTS
        ),
    )
    p = result.portfolio
    assert p.quantity == pytest.approx(-15.0)
    assert p.avg_entry_price == pytest.approx(120.0)
    assert p.realized_pnl == pytest.approx(200.0)  # 10 units * 20
    assert len(p.closed_trades) == 1
    closed = p.closed_trades[0]
    assert closed.direction is Side.BUY
    assert closed.quantity == pytest.approx(10.0)
    assert closed.entry_price == pytest.approx(100.0)
    assert closed.exit_price == pytest.approx(120.0)
    assert closed.gross_pnl == pytest.approx(200.0)
    assert closed.is_win


def test_partial_reduce_keeps_the_trade_open():
    series = flat_series(6, price=100.0)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(quantity=10.0)
        elif len(ctx.view) == 3:
            ctx.sell(quantity=4.0)

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert result.portfolio.quantity == pytest.approx(6.0)
    assert result.portfolio.closed_trades == []
    assert result.metrics.trades == 0
    assert result.metrics.win_rate is None


def test_averaging_in_updates_the_average_entry():
    """Buy 1 at 100 then 1 at 200: average entry is 150.

    Orders go in on bars 0 and 2, so they fill at the opens of bars 1 (100) and
    3 (200).
    """
    series = make_series(
        [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (200.0, 200.0), (200.0, 200.0)]
    )

    def strategy(ctx):
        if len(ctx.view) in (1, 3):
            ctx.buy(quantity=1.0)

    result = run_backtest(
        series, strategy, BacktestConfig(initial_cash=10_000.0, costs=ZERO_COSTS)
    )
    assert result.portfolio.quantity == pytest.approx(2.0)
    assert result.portfolio.avg_entry_price == pytest.approx(150.0)


def test_closing_a_position_reports_a_closed_trade():
    series = make_series(
        [(100.0, 100.0), (100.0, 100.0), (110.0, 110.0), (110.0, 110.0)]
    )

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(quantity=5.0)
        elif len(ctx.view) == 3:
            ctx.close_position()

    result = run_backtest(
        series, strategy, BacktestConfig(initial_cash=10_000.0, costs=ZERO_COSTS)
    )
    assert result.portfolio.is_flat
    assert result.metrics.trades == 1
    assert result.metrics.win_rate == 1.0


def test_close_position_on_a_flat_book_queues_nothing():
    series = flat_series(3)

    def strategy(ctx):
        ctx.close_position()

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert result.fills == ()
    assert result.rejected == ()


# -- strategy plumbing ------------------------------------------------------


def test_a_plain_function_works_as_a_strategy():
    series = flat_series(3, price=100.0)
    calls = []
    result = run_backtest(
        series, lambda ctx: calls.append(ctx.now), BacktestConfig(costs=ZERO_COSTS)
    )
    assert len(calls) == 3
    assert result.metrics.bars == 3


def test_on_start_runs_once_before_the_first_on_bar():
    series = flat_series(4)
    events = []

    class Recorder:
        def on_start(self, ctx):
            events.append(("start", len(ctx.view)))

        def on_bar(self, ctx):
            events.append(("bar", len(ctx.view)))

        def on_finish(self, ctx):
            events.append(("finish", len(ctx.view)))

    run_backtest(series, Recorder(), BacktestConfig(costs=ZERO_COSTS))
    assert events[0] == ("start", 1)
    assert [e for e, _ in events].count("start") == 1
    assert [e for e, _ in events].count("finish") == 1
    assert events[-1] == ("finish", 4)


def test_warmup_bars_suppresses_the_callback_but_not_the_equity_curve():
    series = flat_series(5, price=100.0)
    seen = []

    def strategy(ctx):
        seen.append(len(ctx.view))

    result = run_backtest(
        series, strategy, BacktestConfig(costs=ZERO_COSTS, warmup_bars=2)
    )
    assert seen == [3, 4, 5]
    assert len(result.equity_curve) == 5


def test_negative_warmup_is_refused():
    with pytest.raises(ValueError, match="warmup_bars"):
        BacktestConfig(warmup_bars=-1)


def test_context_exposes_pending_orders_so_a_strategy_can_avoid_double_submitting():
    series = flat_series(4, price=100.0)
    snapshots = []

    def strategy(ctx):
        snapshots.append(len(ctx.pending_orders))
        if ctx.is_flat and not ctx.pending_orders:
            ctx.buy(quantity=1.0)

    result = run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert len(result.fills) == 1
    assert snapshots[0] == 0


def test_context_reports_equity_marked_at_the_current_close():
    series = make_series([(100.0, 100.0), (100.0, 150.0), (150.0, 150.0)])
    equities = []

    def strategy(ctx):
        equities.append(ctx.equity)
        if len(ctx.view) == 1:
            ctx.buy(quantity=1.0)

    run_backtest(
        series, strategy, BacktestConfig(initial_cash=1_000.0, costs=ZERO_COSTS)
    )
    assert equities[0] == pytest.approx(1_000.0)
    assert equities[1] == pytest.approx(900.0 + 150.0)


def test_portfolio_requires_positive_initial_cash():
    with pytest.raises(ValueError, match="initial_cash"):
        Portfolio(0.0)
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestConfig(initial_cash=-1.0)


def test_default_config_is_pessimistic_so_a_careless_run_overstates_costs():
    """A BacktestConfig() with no arguments must not be a free lunch."""
    series = flat_series(3, price=100.0)
    result = run_backtest(series, buy_on_first_bar(percent_of_equity=1.0))
    assert result.portfolio.fees_paid > 0
    assert result.portfolio.slippage_paid > 0
    assert result.metrics.cumulative_return < 0  # flat price, costs only
