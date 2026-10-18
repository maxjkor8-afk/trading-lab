"""Slippage: direction, size-dependence, and walking a real book.

The expected impact numbers are derived from the model's stated formula by hand:
with `depth_quote` resting uniformly within `depth_bps`, filling notional N walks
to `k*N` bps where `k = depth_bps/depth_quote`, and pays the average `k*N/2`.
"""

from __future__ import annotations

import pytest

from harness.costs import (
    FixedSpreadSlippage,
    InsufficientLiquidity,
    LinearBookSlippage,
    NoSlippage,
    walk_book,
)
from harness.enums import Side


# -- direction --------------------------------------------------------------


def test_slippage_is_always_adverse():
    """A buy fills above the reference, a sell below. Never the other way."""
    model = LinearBookSlippage(half_spread_bps=5.0, depth_quote=1_000_000.0, depth_bps=10.0)
    buy = model.fill_price(Side.BUY, 100.0, 10.0)
    sell = model.fill_price(Side.SELL, 100.0, 10.0)
    assert buy > 100.0
    assert sell < 100.0
    # Symmetric around the reference for the same size.
    assert buy - 100.0 == pytest.approx(100.0 - sell)


def test_no_slippage_fills_at_reference():
    assert NoSlippage().fill_price(Side.BUY, 123.45, 7.0) == 123.45
    assert NoSlippage().slippage_bps(Side.BUY, 123.45, 7.0) == 0.0


def test_fixed_spread_is_size_independent():
    model = FixedSpreadSlippage(half_spread_bps=5.0)
    small = model.fill_price(Side.BUY, 100.0, 0.001)
    large = model.fill_price(Side.BUY, 100.0, 1_000.0)
    assert small == large == pytest.approx(100.0 * 1.0005)


# -- size dependence --------------------------------------------------------


def test_impact_is_zero_for_a_zero_size_order():
    assert LinearBookSlippage().impact_bps(0.0) == 0.0


def test_impact_at_full_depth_is_half_the_band():
    """Consume exactly `depth_quote`: worst price is depth_bps, average is half."""
    model = LinearBookSlippage(half_spread_bps=0.0, depth_quote=100_000.0, depth_bps=10.0)
    assert model.impact_bps(100_000.0) == pytest.approx(5.0)


def test_impact_is_linear_in_size_inside_the_band():
    """Half the depth costs half the impact: k*N/2 with N halved."""
    model = LinearBookSlippage(half_spread_bps=0.0, depth_quote=100_000.0, depth_bps=10.0)
    assert model.impact_bps(50_000.0) == pytest.approx(2.5)
    assert model.impact_bps(10_000.0) == pytest.approx(0.5)


def test_impact_accelerates_beyond_the_band():
    """Past the top of the book liquidity thins, so cost grows faster than linearly.

    depth_quote=100k, depth_bps=10, overflow_density=0.25 so k_out = 4*k.
    Filling 200k: inside 100k averages 5bps; the 100k excess runs from 10bps to
    10 + 4*(10/100k)*100k = 50bps, averaging 30bps. Weighted:
    (5*100k + 30*100k)/200k = 17.5bps.
    """
    model = LinearBookSlippage(
        half_spread_bps=0.0, depth_quote=100_000.0, depth_bps=10.0, overflow_density=0.25
    )
    assert model.impact_bps(200_000.0) == pytest.approx(17.5)
    # Strictly worse than the linear extrapolation of 10bps.
    assert model.impact_bps(200_000.0) > 2 * model.impact_bps(100_000.0)


def test_impact_grows_monotonically_with_size():
    model = LinearBookSlippage(depth_quote=100_000.0, depth_bps=10.0)
    sizes = [1_000.0, 10_000.0, 100_000.0, 250_000.0, 1_000_000.0]
    impacts = [model.impact_bps(s) for s in sizes]
    assert impacts == sorted(impacts)
    assert all(a < b for a, b in zip(impacts, impacts[1:]))


def test_total_slippage_adds_half_spread_to_impact():
    model = LinearBookSlippage(half_spread_bps=5.0, depth_quote=100_000.0, depth_bps=10.0)
    # 1 unit at 50,000 = 50,000 notional = half the depth -> 2.5bps impact.
    assert model.slippage_bps(Side.BUY, 50_000.0, 1.0) == pytest.approx(7.5)
    assert model.fill_price(Side.BUY, 50_000.0, 1.0) == pytest.approx(50_000.0 * 1.00075)


def test_a_bigger_order_gets_a_worse_price():
    """The property that stops a strategy assuming infinite top-of-book size."""
    model = LinearBookSlippage(depth_quote=250_000.0, depth_bps=10.0)
    small = model.fill_price(Side.BUY, 30_000.0, 0.1)
    big = model.fill_price(Side.BUY, 30_000.0, 100.0)
    assert big > small


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="depth_quote"):
        LinearBookSlippage(depth_quote=0.0)
    with pytest.raises(ValueError, match="overflow_density"):
        LinearBookSlippage(overflow_density=0.0)
    with pytest.raises(ValueError, match="overflow_density"):
        LinearBookSlippage(overflow_density=1.5)
    with pytest.raises(ValueError, match="half_spread_bps"):
        LinearBookSlippage(half_spread_bps=-1.0)


# -- walking a real book ----------------------------------------------------


def test_walk_book_averages_across_levels():
    """2 at 100 and 3 at 101 = (200 + 303)/5 = 100.6."""
    walk = walk_book([(100.0, 2.0), (101.0, 3.0)], 5.0)
    assert walk.average_price == pytest.approx(100.6)
    assert walk.filled_quantity == pytest.approx(5.0)
    assert walk.levels_consumed == 2
    assert walk.unfilled_quantity == 0.0


def test_walk_book_stops_at_the_first_level_when_it_can():
    walk = walk_book([(100.0, 10.0), (101.0, 10.0)], 4.0)
    assert walk.average_price == pytest.approx(100.0)
    assert walk.levels_consumed == 1


def test_walk_book_raises_when_the_book_is_too_thin():
    """An order silently truncated to the size of the book is a fill you did not get."""
    with pytest.raises(InsufficientLiquidity, match="unfilled"):
        walk_book([(100.0, 1.0)], 5.0)


def test_walk_book_can_partially_fill_when_asked():
    walk = walk_book([(100.0, 1.0), (101.0, 1.0)], 5.0, allow_partial=True)
    assert walk.filled_quantity == pytest.approx(2.0)
    assert walk.unfilled_quantity == pytest.approx(3.0)
    assert walk.average_price == pytest.approx(100.5)


def test_walk_book_rejects_nonsense():
    with pytest.raises(ValueError, match="quantity must be positive"):
        walk_book([(100.0, 1.0)], 0.0)
    with pytest.raises(ValueError, match="invalid book level"):
        walk_book([(0.0, 1.0)], 1.0)
    with pytest.raises(InsufficientLiquidity, match="empty"):
        walk_book([], 1.0, allow_partial=True)
