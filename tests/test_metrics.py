"""Metrics arithmetic.

Expected values are written out as explicit arithmetic on the sample, not read
back from the implementation, so these tests check the formula rather than
echoing it. The Sharpe tests use sample standard deviation (ddof=1), which is
the convention the implementation documents.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_series

from harness.costs import (
    CostModel,
    FeeSchedule,
    NoSlippage,
    ZERO_COSTS,
    ZERO_FUNDING,
)
from harness.engine import BacktestConfig, run_backtest
from harness.metrics import (
    EquityPoint,
    SECONDS_PER_YEAR,
    annualized_return,
    infer_periods_per_year,
    max_drawdown,
    periodic_returns,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from harness.portfolio import ClosedTrade, Side

UTC = timezone.utc
START = datetime(2023, 1, 1, tzinfo=UTC)

# equity 100 -> 110 -> 99 -> 108.9, i.e. returns of +10%, -10%, +10%
SAMPLE_EQUITY = [100.0, 110.0, 99.0, 108.9]
SAMPLE_RETURNS = [0.1, -0.1, 0.1]


def points(equity: list[float], step: timedelta = timedelta(days=1)) -> list[EquityPoint]:
    return [
        EquityPoint(
            ts=START + i * step, equity=e, mark_price=e, position=0.0, cash=e
        )
        for i, e in enumerate(equity)
    ]


# -- returns ----------------------------------------------------------------


def test_periodic_returns_are_period_over_period():
    got = periodic_returns(SAMPLE_EQUITY)
    assert got == pytest.approx(SAMPLE_RETURNS)


def test_periodic_returns_refuses_a_ruined_account():
    """Equity through zero means the run blew up; statistics past it are fiction."""
    with pytest.raises(ValueError, match="zero or below"):
        periodic_returns([100.0, 0.0, 50.0])
    with pytest.raises(ValueError, match="zero or below"):
        periodic_returns([100.0, -10.0, 50.0])


def test_single_point_has_no_returns():
    assert periodic_returns([100.0]) == []


# -- sharpe -----------------------------------------------------------------


def test_sharpe_matches_hand_arithmetic():
    mean = (0.1 + -0.1 + 0.1) / 3
    variance = ((0.1 - mean) ** 2 + (-0.1 - mean) ** 2 + (0.1 - mean) ** 2) / 2
    expected = mean / math.sqrt(variance) * math.sqrt(365.0)
    assert sharpe_ratio(SAMPLE_RETURNS, 365.0) == pytest.approx(expected)
    # Sanity: roughly 5.5 for this sample at daily frequency.
    assert sharpe_ratio(SAMPLE_RETURNS, 365.0) == pytest.approx(5.5151, rel=1e-4)


def test_sharpe_scales_with_the_square_root_of_frequency():
    daily = sharpe_ratio(SAMPLE_RETURNS, 365.0)
    hourly = sharpe_ratio(SAMPLE_RETURNS, 365.0 * 24)
    assert hourly / daily == pytest.approx(math.sqrt(24.0))


def test_sharpe_subtracts_the_risk_free_rate_per_period():
    rf_annual = 0.0365
    per_period = rf_annual / 365.0
    excess = [r - per_period for r in SAMPLE_RETURNS]
    mean = sum(excess) / 3
    variance = sum((r - mean) ** 2 for r in excess) / 2
    expected = mean / math.sqrt(variance) * math.sqrt(365.0)
    assert sharpe_ratio(SAMPLE_RETURNS, 365.0, rf_annual) == pytest.approx(expected)


def test_sharpe_is_none_rather_than_infinite_when_there_is_no_variance():
    """A constant return series has no risk in the denominator, not zero risk."""
    assert sharpe_ratio([0.01, 0.01, 0.01], 365.0) is None


def test_sharpe_is_none_with_too_few_observations():
    assert sharpe_ratio([], 365.0) is None
    assert sharpe_ratio([0.01], 365.0) is None


def test_negative_mean_gives_negative_sharpe():
    assert sharpe_ratio([-0.01, -0.02, -0.005], 365.0) < 0


# -- sortino ----------------------------------------------------------------


def test_sortino_matches_hand_arithmetic():
    """Downside deviation is the RMS of shortfalls over the full sample."""
    mean = (0.1 + -0.1 + 0.1) / 3
    shortfalls = [0.0, -0.1, 0.0]
    downside = math.sqrt(sum(s * s for s in shortfalls) / 3)
    expected = mean / downside * math.sqrt(365.0)
    assert sortino_ratio(SAMPLE_RETURNS, 365.0) == pytest.approx(expected)
    assert sortino_ratio(SAMPLE_RETURNS, 365.0) == pytest.approx(11.0303, rel=1e-4)


def test_sortino_exceeds_sharpe_when_downside_is_the_smaller_risk():
    """Two up moves and one down: penalising only the down move flatters less."""
    assert sortino_ratio(SAMPLE_RETURNS, 365.0) > sharpe_ratio(SAMPLE_RETURNS, 365.0)


def test_sortino_is_none_when_nothing_fell_below_target():
    assert sortino_ratio([0.01, 0.02, 0.03], 365.0) is None


def test_sortino_is_none_with_too_few_observations():
    assert sortino_ratio([0.01], 365.0) is None


# -- drawdown ---------------------------------------------------------------


def test_max_drawdown_is_peak_to_trough():
    """Peak 110 to trough 99 is 11/110 = 10%."""
    dd, start, end = max_drawdown(points(SAMPLE_EQUITY))
    assert dd == pytest.approx(0.10)
    assert start == START + timedelta(days=1)
    assert end == START + timedelta(days=2)


def test_max_drawdown_is_zero_for_a_monotonic_rise():
    dd, start, end = max_drawdown(points([100.0, 110.0, 120.0]))
    assert dd == 0.0
    assert start is None and end is None


def test_max_drawdown_finds_the_worst_of_several():
    # 100 -> 90 (10%), then 120 -> 60 (50%)
    dd, _, end = max_drawdown(points([100.0, 90.0, 120.0, 60.0, 80.0]))
    assert dd == pytest.approx(0.50)
    assert end == START + timedelta(days=3)


def test_max_drawdown_handles_a_total_loss():
    dd, _, _ = max_drawdown(points([100.0, 0.0]))
    assert dd == pytest.approx(1.0)


# -- annualisation ----------------------------------------------------------


def test_annualized_return_compounds_over_elapsed_time():
    """10% in 365 days is 10% a year."""
    assert annualized_return(0.10, timedelta(days=365)) == pytest.approx(0.10)


def test_annualized_return_of_a_short_window_is_an_extrapolation():
    """A good quarter annualises into a number nobody should rely on.

    The TradingAgents result in RESEARCH.md section 2a is a 3-month window; this
    is what that does to a return.
    """
    quarterly = annualized_return(0.2662, timedelta(days=89))
    assert quarterly == pytest.approx(1.2662 ** (365.0 / 89.0) - 1.0)
    assert quarterly > 1.5  # 26.6% in a quarter annualises past +150%


def test_annualized_return_handles_a_wipeout():
    assert annualized_return(-1.0, timedelta(days=30)) == -1.0
    assert annualized_return(-1.5, timedelta(days=30)) == -1.0


def test_annualized_return_needs_elapsed_time():
    assert annualized_return(0.1, timedelta(0)) is None


def test_periods_per_year_is_inferred_from_the_bar_spacing():
    assert infer_periods_per_year(points(SAMPLE_EQUITY)) == pytest.approx(365.0)
    hourly = points(SAMPLE_EQUITY, step=timedelta(hours=1))
    assert infer_periods_per_year(hourly) == pytest.approx(365.0 * 24)
    assert SECONDS_PER_YEAR == 365.0 * 24 * 3600


# -- win rate ---------------------------------------------------------------


def trade(gross: float, costs: float) -> ClosedTrade:
    return ClosedTrade(
        opened_at=START,
        closed_at=START + timedelta(days=1),
        direction=Side.BUY,
        quantity=1.0,
        entry_price=100.0,
        exit_price=100.0 + gross,
        gross_pnl=gross,
        fees=costs,
    )


def test_win_rate_counts_trades_that_won_after_costs():
    """A trade that is only profitable before fees is not a win."""
    trades = [trade(10.0, 1.0), trade(5.0, 6.0), trade(-3.0, 1.0), trade(20.0, 2.0)]
    assert win_rate(trades) == pytest.approx(0.5)


def test_win_rate_is_none_with_no_closed_trades():
    """Nothing happened is not a 0% win rate."""
    assert win_rate([]) is None


def test_a_breakeven_trade_is_not_a_win():
    assert win_rate([trade(1.0, 1.0)]) == 0.0


# -- cost ratio, end to end -------------------------------------------------


def ten_bps_taker() -> CostModel:
    """10bps either way, no slippage, no funding: fees in isolation."""
    return CostModel(
        fees=FeeSchedule(maker_bps=10.0, taker_bps=10.0),
        slippage=NoSlippage(),
        funding=ZERO_FUNDING,
    )


def round_trip(opens_closes, cash=10_000.0, costs=None):
    """Buy on bar 0, sell on bar 2. Fills land on bars 1 and 3."""
    series = make_series(opens_closes)

    def strategy(ctx):
        if len(ctx.view) == 1:
            ctx.buy(percent_of_equity=1.0, tag="in")
        elif len(ctx.view) == 3 and not ctx.is_flat:
            ctx.sell(percent_of_equity=1.0, tag="out")

    return run_backtest(
        series, strategy, BacktestConfig(initial_cash=cash, costs=costs or ten_bps_taker())
    )


def test_cost_ratio_on_a_hand_calculable_round_trip():
    """Buy 100, sell 110, 10bps each way.

    qty = 10,000 / (100 * 1.001) = 99.9000999 units
    gross pnl = 99.9000999 * 10 = 999.000999
    fees = 0.001 * (9,990.00999 + 10,989.010989) = 20.979021
    costs / gross = 20.979021 / 999.000999 = exactly 0.021
    """
    result = round_trip([(100.0, 100.0), (100.0, 100.0), (110.0, 110.0), (110.0, 110.0)])
    m = result.metrics
    assert result.portfolio.quantity == 0.0
    assert m.gross_pnl == pytest.approx(999.000999)
    assert m.total_costs == pytest.approx(20.979021)
    assert m.net_pnl == pytest.approx(978.021978)
    assert m.cost_ratio == pytest.approx(0.021)
    assert m.trades == 1
    assert m.win_rate == 1.0
    assert m.turnover == pytest.approx(2.0, rel=1e-3)


def test_cost_ratio_above_one_exposes_a_strategy_that_only_works_before_fees():
    """The headline diagnostic: a +0.1% move cannot pay a 0.2% round trip.

    Gross PnL is positive, net PnL is negative, and the cost ratio says why.
    """
    result = round_trip(
        [(100.0, 100.0), (100.0, 100.0), (100.1, 100.1), (100.1, 100.1)]
    )
    m = result.metrics
    assert m.gross_pnl > 0
    assert m.net_pnl < 0
    assert m.cost_ratio is not None and m.cost_ratio > 1.0
    assert m.cost_ratio == pytest.approx(2.001, rel=1e-3)
    assert m.win_rate == 0.0


def test_cost_ratio_is_none_when_there_was_no_gross_profit_to_eat():
    """A losing strategy's cost ratio is arithmetic noise, so it is not reported."""
    result = round_trip([(100.0, 100.0), (100.0, 100.0), (90.0, 90.0), (90.0, 90.0)])
    assert result.metrics.gross_pnl < 0
    assert result.metrics.cost_ratio is None


def test_zero_cost_run_reports_no_costs_and_equal_gross_and_net():
    result = round_trip(
        [(100.0, 100.0), (100.0, 100.0), (110.0, 110.0), (110.0, 110.0)],
        costs=ZERO_COSTS,
    )
    m = result.metrics
    assert m.total_costs == 0.0
    assert m.gross_pnl == pytest.approx(m.net_pnl)
    assert m.cost_ratio == pytest.approx(0.0)
    assert m.net_pnl == pytest.approx(1_000.0)


def test_report_is_readable_and_shows_costs_next_to_returns():
    result = round_trip([(100.0, 100.0), (100.0, 100.0), (110.0, 110.0), (110.0, 110.0)])
    text = result.report()
    for label in ("cumulative return", "sharpe", "max drawdown", "costs / gross pnl"):
        assert label in text
    assert "n/a" not in text.split("sharpe")[1].split("\n")[0]


def test_metrics_on_an_empty_curve_is_an_error():
    from harness.metrics import compute_metrics
    from harness.portfolio import Portfolio

    with pytest.raises(ValueError, match="empty equity curve"):
        compute_metrics([], Portfolio(1_000.0))
