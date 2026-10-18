"""The leakage guard. These tests are the reason the harness exists.

Each one is a deliberate attempt to see the future through the public API a
strategy is handed. Every attempt must raise. If any of these ever starts
passing silently, every number the harness produces afterwards is suspect.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from conftest import DAY, EPOCH, make_series

from harness.clock import Clock, ClockError, LookAheadError
from harness.data import Bar, BarSeries, PointInTimeFeed, PointInTimeView
from harness.engine import BacktestConfig, run_backtest
from harness.costs import ZERO_COSTS


@pytest.fixture
def series() -> BarSeries:
    return make_series([(10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 14.0)])


@pytest.fixture
def view_at_bar_1(series: BarSeries) -> PointInTimeView:
    """A view positioned so bars 0 and 1 are visible and bars 2-3 are not."""
    clock = Clock(series[1].close_time)
    return PointInTimeView(series, clock)


# -- the headline test ------------------------------------------------------


def test_deliberate_lookahead_by_index_raises(view_at_bar_1):
    """Index the next bar directly. This is the leak in its simplest form."""
    assert len(view_at_bar_1) == 2
    with pytest.raises(LookAheadError, match="look-ahead blocked"):
        view_at_bar_1[2]


def test_deliberate_lookahead_by_offset_raises(view_at_bar_1):
    with pytest.raises(LookAheadError, match=r"at\(\+1\)"):
        view_at_bar_1.at(1)


def test_far_future_index_raises_lookahead_not_indexerror(view_at_bar_1):
    """A wild index must read as look-ahead, not as an ordinary bounds error.

    IndexError invites a try/except that hides the bug; LookAheadError names it.
    """
    with pytest.raises(LookAheadError):
        view_at_bar_1[999]


def test_history_longer_than_available_raises(view_at_bar_1):
    """Asking for 20 bars when 2 exist must raise, not quietly return 2.

    A 20-period indicator silently fed 2 bars produces a plausible, wrong number
    - the worst possible failure mode.
    """
    with pytest.raises(LookAheadError, match="history"):
        view_at_bar_1.history(20)


def test_history_within_bounds_is_fine(view_at_bar_1):
    assert view_at_bar_1.closes(2) == (11.0, 12.0)
    assert view_at_bar_1.current.close == 12.0
    assert view_at_bar_1.at(0).close == 12.0
    assert view_at_bar_1.at(-1).close == 11.0


def test_len_and_iteration_stop_at_the_present(view_at_bar_1):
    """Iteration must not walk off the end of the present."""
    assert [b.close for b in view_at_bar_1] == [11.0, 12.0]
    assert len(list(view_at_bar_1)) == len(view_at_bar_1) == 2


def test_slicing_cannot_reach_past_the_present(view_at_bar_1):
    """An open-ended slice is bounded by the clock, not by the raw series."""
    assert [b.close for b in view_at_bar_1[:]] == [11.0, 12.0]
    assert [b.close for b in view_at_bar_1[0:99]] == [11.0, 12.0]


def test_negative_index_before_series_start_raises(view_at_bar_1):
    with pytest.raises(IndexError):
        view_at_bar_1[-5]
    with pytest.raises(IndexError):
        view_at_bar_1.at(-5)


# -- the clock itself -------------------------------------------------------


def test_clock_refuses_to_run_backwards():
    clock = Clock(EPOCH)
    with pytest.raises(ClockError, match="strictly forward"):
        clock.advance_to(EPOCH - DAY)
    with pytest.raises(ClockError, match="strictly forward"):
        clock.advance_to(EPOCH)  # sideways is also forbidden


def test_clock_rejects_naive_datetimes():
    """Guessing a timezone puts bars on the wrong side of funding timestamps."""
    with pytest.raises(ClockError, match="timezone-aware"):
        Clock(datetime(2023, 1, 1))


def test_unstarted_clock_has_no_now():
    with pytest.raises(ClockError, match="not been started"):
        _ = Clock().now


def test_assert_visible_blocks_the_future():
    clock = Clock(EPOCH)
    clock.assert_visible(EPOCH)
    clock.assert_visible(EPOCH - DAY)
    with pytest.raises(LookAheadError, match="look-ahead blocked"):
        clock.assert_visible(EPOCH + DAY)


def test_view_follows_the_clock_rather_than_a_cached_cursor(series):
    """Visibility is recomputed per access, so it cannot go stale."""
    clock = Clock(series[0].close_time)
    view = PointInTimeView(series, clock)
    assert len(view) == 1
    clock.advance_to(series[2].close_time)
    assert len(view) == 3
    assert view.current.close == 13.0


def test_bar_is_invisible_until_it_closes(series):
    """The core convention: a bar is knowable at its close, not at its open.

    Treating the open timestamp as the moment of knowledge would mean knowing a
    day's close at the start of the day.
    """
    bar = series[1]
    clock = Clock(bar.open_time)
    view = PointInTimeView(series, clock)
    assert len(view) == 1  # only bar 0 has closed
    assert view.current is series[0]
    clock.advance_to(bar.close_time)
    assert view.current is bar


def test_view_before_any_bar_closes_raises():
    series = make_series([(10.0, 11.0), (11.0, 12.0)])
    clock = Clock(series[0].open_time - DAY)
    view = PointInTimeView(series, clock)
    with pytest.raises(LookAheadError, match="have closed"):
        _ = view.current


# -- the feed ---------------------------------------------------------------


def test_feed_has_no_api_for_fetching_an_arbitrary_timestamp(series):
    """`view(symbol)` takes no timestamp; the clock decides what is visible."""
    clock = Clock(series[1].close_time)
    feed = PointInTimeFeed(clock, series)
    assert not hasattr(feed, "get")
    assert len(feed.view(series.symbol)) == 2


def test_feed_bars_until_rejects_a_future_timestamp(series):
    clock = Clock(series[1].close_time)
    feed = PointInTimeFeed(clock, series)
    assert len(feed.bars_until(series.symbol, series[1].close_time)) == 2
    with pytest.raises(LookAheadError):
        feed.bars_until(series.symbol, series[3].close_time)


# -- end to end -------------------------------------------------------------


def test_strategy_attempting_lookahead_fails_the_backtest(series):
    """A cheating strategy must crash the run, not produce a great Sharpe.

    This is the whole design intent: leakage is loud.
    """

    class Cheater:
        def on_bar(self, ctx):
            ctx.view.at(1)  # tomorrow's bar, please

    with pytest.raises(LookAheadError):
        run_backtest(series, Cheater(), BacktestConfig(costs=ZERO_COSTS))


def test_strategy_context_exposes_no_route_to_the_raw_series(series):
    """The context hands over a bounded view and nothing wider."""
    seen = []

    class Inspector:
        def on_bar(self, ctx):
            seen.append(ctx)

    run_backtest(series, Inspector(), BacktestConfig(costs=ZERO_COSTS))
    ctx = seen[0]
    assert isinstance(ctx.view, PointInTimeView)
    public = {a for a in dir(ctx) if not a.startswith("_")}
    # Nothing on the public surface hands back a BarSeries.
    for name in public:
        value = getattr(ctx, name, None)
        assert not isinstance(value, BarSeries), f"ctx.{name} leaks the raw series"


def test_a_full_run_never_lets_the_view_outrun_the_clock(series):
    """Invariant check on every bar of a real run."""
    observed: list[tuple[datetime, int, float]] = []

    def strategy(ctx):
        assert ctx.view.current.close_time <= ctx.now
        with pytest.raises(LookAheadError):
            ctx.view.at(1)
        observed.append((ctx.now, len(ctx.view), ctx.view.current.close))

    run_backtest(series, strategy, BacktestConfig(costs=ZERO_COSTS))
    assert [n for _, n, _ in observed] == [1, 2, 3, 4]
    assert [c for _, _, c in observed] == [11.0, 12.0, 13.0, 14.0]


# -- series construction guards --------------------------------------------


def test_bar_rejects_close_before_open_time():
    with pytest.raises(ValueError, match="must be after"):
        Bar(
            open_time=EPOCH,
            close_time=EPOCH,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
        )


def test_bar_rejects_price_outside_high_low():
    with pytest.raises(ValueError, match="outside"):
        Bar(
            open_time=EPOCH,
            close_time=EPOCH + DAY,
            open=10.0,
            high=9.0,
            low=8.0,
            close=8.5,
        )


def test_series_rejects_out_of_order_bars():
    a = make_series([(10.0, 11.0), (11.0, 12.0)])
    with pytest.raises(ValueError, match="non-monotonic|overlapping"):
        BarSeries("BAD", [a[1], a[0]])


def test_series_rejects_duplicate_timestamps():
    """A repeated bar is caught as an overlap - it overlaps itself."""
    a = make_series([(10.0, 11.0)])
    with pytest.raises(ValueError, match="overlapping|non-monotonic"):
        BarSeries("BAD", [a[0], a[0]])


def test_index_visible_at_handles_gaps():
    """A gap in the data must not make a later bar visible early."""
    bars = [
        Bar(EPOCH, EPOCH + DAY, 10.0, 11.0, 10.0, 11.0),
        Bar(EPOCH + 5 * DAY, EPOCH + 6 * DAY, 11.0, 12.0, 11.0, 12.0),
    ]
    series = BarSeries("GAPPY", bars)
    assert series.index_visible_at(EPOCH + DAY) == 0
    assert series.index_visible_at(EPOCH + 3 * DAY) == 0
    assert series.index_visible_at(EPOCH + 6 * DAY) == 1
    assert series.index_visible_at(EPOCH - timedelta(seconds=1)) == -1
