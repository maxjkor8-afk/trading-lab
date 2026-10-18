"""Shared fixtures. Synthetic bars are built with explicit prices so that every
expected number in a test can be worked out on paper."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.data import Bar, BarSeries  # noqa: E402

DAY = timedelta(days=1)
EPOCH = datetime(2023, 1, 1, tzinfo=timezone.utc)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BTC_2023_CSV = DATA_DIR / "BTC-USD-1d-2023.csv"
BTC_2023_CROSSCHECK_CSV = DATA_DIR / "BTCUSD-1d-2023-bitstamp.csv"


def make_bar(
    index: int,
    open_: float,
    close: float,
    high: float | None = None,
    low: float | None = None,
    start: datetime = EPOCH,
    duration: timedelta = DAY,
) -> Bar:
    """A bar whose open/close you choose; high/low default to enclose them."""
    open_time = start + index * duration
    return Bar(
        open_time=open_time,
        close_time=open_time + duration,
        open=open_,
        high=high if high is not None else max(open_, close),
        low=low if low is not None else min(open_, close),
        close=close,
        volume=1.0,
    )


def make_series(
    prices: list[tuple[float, float]],
    symbol: str = "TEST",
    start: datetime = EPOCH,
    duration: timedelta = DAY,
) -> BarSeries:
    """Build a contiguous series from a list of (open, close) pairs."""
    return BarSeries(
        symbol,
        [
            make_bar(i, o, c, start=start, duration=duration)
            for i, (o, c) in enumerate(prices)
        ],
    )


def flat_series(n: int, price: float = 100.0, **kwargs) -> BarSeries:
    """`n` bars that never move. Isolates costs from price action."""
    return make_series([(price, price)] * n, **kwargs)


@pytest.fixture
def ramp() -> BarSeries:
    """Five bars, open == close, stepping 100 -> 140. Entirely hand-checkable."""
    return make_series([(100.0, 100.0), (110.0, 110.0), (120.0, 120.0), (130.0, 130.0), (140.0, 140.0)])


@pytest.fixture
def btc_2023() -> BarSeries:
    if not BTC_2023_CSV.exists():
        pytest.skip(
            f"{BTC_2023_CSV.name} missing; regenerate with "
            "scripts/fetch_ohlcv.py (see README)"
        )
    return BarSeries.from_csv(BTC_2023_CSV, symbol="BTC-USD")


@pytest.fixture
def btc_2023_crosscheck() -> BarSeries:
    if not BTC_2023_CROSSCHECK_CSV.exists():
        pytest.skip(f"{BTC_2023_CROSSCHECK_CSV.name} missing; see README")
    return BarSeries.from_csv(BTC_2023_CROSSCHECK_CSV, symbol="BTCUSD-bitstamp")
