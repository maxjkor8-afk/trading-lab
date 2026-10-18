"""Point-in-time market data.

Convention that does most of the anti-leakage work: a bar is identified by its
**close time**, and is invisible until the clock reaches that close time. A daily
candle labelled 2023-01-01 is not knowable at 2023-01-01 00:00 - its high, low
and close are only settled at 2023-01-02 00:00. Treating the open timestamp as
the moment of knowledge is the most common look-ahead bug in hand-rolled
backtests, so the types here do not let you express it: `Bar` carries both
timestamps and visibility is always governed by `close_time`.
"""

from __future__ import annotations

import bisect
import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from harness.clock import Clock, LookAheadError, ensure_utc


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV interval. Immutable, because a replayed bar must not be edited."""

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_time", ensure_utc(self.open_time))
        object.__setattr__(self, "close_time", ensure_utc(self.close_time))
        if self.close_time <= self.open_time:
            raise ValueError(
                f"bar close_time {self.close_time.isoformat()} must be after "
                f"open_time {self.open_time.isoformat()}"
            )
        if self.low > self.high:
            raise ValueError(f"bar low {self.low} exceeds high {self.high}")
        for name in ("open", "close"):
            px = getattr(self, name)
            if not (self.low <= px <= self.high):
                raise ValueError(
                    f"bar {name} {px} outside [low={self.low}, high={self.high}]"
                )
        if min(self.low, self.open, self.close) <= 0:
            raise ValueError("bar prices must be positive")

    @property
    def duration(self) -> timedelta:
        return self.close_time - self.open_time


class BarSeries(Sequence[Bar]):
    """An immutable, strictly time-ordered run of bars for one symbol."""

    __slots__ = ("_bars", "_close_times", "symbol")

    def __init__(self, symbol: str, bars: Sequence[Bar]) -> None:
        bars = tuple(bars)
        if not bars:
            raise ValueError(f"BarSeries({symbol!r}) requires at least one bar")
        for prev, nxt in zip(bars, bars[1:]):
            if nxt.open_time < prev.close_time:
                raise ValueError(
                    f"overlapping bars in {symbol!r}: {prev.close_time.isoformat()} "
                    f"then open {nxt.open_time.isoformat()}"
                )
            if nxt.close_time <= prev.close_time:
                raise ValueError(
                    f"non-monotonic bars in {symbol!r} at {nxt.close_time.isoformat()}"
                )
        self.symbol = symbol
        self._bars = bars
        self._close_times = tuple(b.close_time for b in bars)

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index):  # type: ignore[override]
        return self._bars[index]

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def index_visible_at(self, ts: datetime) -> int:
        """Index of the last bar whose close_time <= ts, or -1 if none."""
        ts = ensure_utc(ts)
        return bisect.bisect_right(self._close_times, ts) - 1

    @property
    def first_close(self) -> datetime:
        return self._close_times[0]

    @property
    def last_close(self) -> datetime:
        return self._close_times[-1]

    @classmethod
    def from_csv(cls, path: str | Path, symbol: str | None = None) -> BarSeries:
        """Load bars written by scripts/fetch_ohlcv.py.

        Expects ISO-8601 UTC open_time/close_time columns, so the knowledge
        timestamp survives the round trip to disk.
        """
        path = Path(path)
        bars: list[Bar] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"open_time", "close_time", "open", "high", "low", "close"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            for row in reader:
                bars.append(
                    Bar(
                        open_time=datetime.fromisoformat(row["open_time"]),
                        close_time=datetime.fromisoformat(row["close_time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0.0),
                    )
                )
        return cls(symbol or path.stem, bars)


class PointInTimeView(Sequence[Bar]):
    """The only market-data object a strategy ever receives.

    It wraps a BarSeries plus the clock and exposes it as a sequence whose length
    is "bars known right now". Every index is resolved against the clock at
    access time, so there is no cached cursor to go stale and no public route to
    a bar that has not closed yet. Indexing past the end raises LookAheadError,
    not IndexError, so the failure reads as what it is.

    Honest caveat: Python has no enforceable private state, so a determined
    caller can still reach `view._series`. The guarantee is that no public API
    returns future data, that every access is bounds-checked against the clock,
    and that tests prove the obvious leak attempts raise. That converts leakage
    from an easy accident into a deliberate act visible in review.
    """

    __slots__ = ("_series", "_clock")

    def __init__(self, series: BarSeries, clock: Clock) -> None:
        self._series = series
        self._clock = clock

    @property
    def symbol(self) -> str:
        return self._series.symbol

    @property
    def now(self) -> datetime:
        return self._clock.now

    def _cursor(self) -> int:
        cursor = self._series.index_visible_at(self._clock.now)
        if cursor < 0:
            raise LookAheadError(
                f"no bars for {self._series.symbol!r} have closed at "
                f"{self._clock.now.isoformat()}"
            )
        return cursor

    def __len__(self) -> int:
        return self._cursor() + 1

    def __getitem__(self, index):  # type: ignore[override]
        cursor = self._cursor()
        if isinstance(index, slice):
            start, stop, step = index.indices(cursor + 1)
            return tuple(self._series[i] for i in range(start, stop, step))
        index = int(index)
        absolute = index if index >= 0 else cursor + 1 + index
        if absolute > cursor:
            raise LookAheadError(
                f"look-ahead blocked: bar index {index} resolves to {absolute}, but "
                f"only bars 0..{cursor} have closed at "
                f"{self._clock.now.isoformat()}"
            )
        if absolute < 0:
            raise IndexError(f"bar index {index} is before the start of the series")
        return self._series[absolute]

    def __iter__(self) -> Iterator[Bar]:
        cursor = self._cursor()
        for i in range(cursor + 1):
            yield self._series[i]

    @property
    def current(self) -> Bar:
        """The most recently closed bar."""
        return self._series[self._cursor()]

    def at(self, offset: int) -> Bar:
        """Bar at `offset` relative to the present. 0 is now, -1 is previous.

        A positive offset is a request for the future and always raises.
        """
        if offset > 0:
            raise LookAheadError(
                f"look-ahead blocked: at(+{offset}) asks for a bar that has not "
                f"closed at {self._clock.now.isoformat()}"
            )
        cursor = self._cursor()
        absolute = cursor + offset
        if absolute < 0:
            raise IndexError(
                f"at({offset}) is before the start of {self._series.symbol!r}"
            )
        return self._series[absolute]

    def history(self, n: int) -> tuple[Bar, ...]:
        """The last `n` closed bars, oldest first. Raises if `n` are not there.

        Raising beats silently returning a short window: an indicator quietly fed
        12 bars instead of 20 produces plausible, wrong numbers.
        """
        if n <= 0:
            raise ValueError(f"history(n) requires n >= 1, got {n}")
        cursor = self._cursor()
        if n > cursor + 1:
            raise LookAheadError(
                f"look-ahead blocked: history({n}) requested but only {cursor + 1} "
                f"bars have closed at {self._clock.now.isoformat()}"
            )
        return tuple(self._series[i] for i in range(cursor + 1 - n, cursor + 1))

    def closes(self, n: int) -> tuple[float, ...]:
        return tuple(b.close for b in self.history(n))

    def __repr__(self) -> str:
        try:
            return (
                f"PointInTimeView({self._series.symbol!r}, {len(self)} bars visible "
                f"at {self._clock.now.isoformat()})"
            )
        except Exception:
            return f"PointInTimeView({self._series.symbol!r}, unstarted)"


class PointInTimeFeed:
    """Multi-symbol data access that cannot be asked for the future.

    Note the API shape: `view(symbol)` takes no timestamp. There is deliberately
    no `get(symbol, ts)` that trusts a caller-supplied timestamp - the clock
    decides. `bars_until` exists for tooling and asserts against the clock.
    """

    __slots__ = ("_clock", "_series")

    def __init__(
        self, clock: Clock, series: Mapping[str, BarSeries] | BarSeries
    ) -> None:
        if isinstance(series, BarSeries):
            series = {series.symbol: series}
        if not series:
            raise ValueError("PointInTimeFeed requires at least one BarSeries")
        self._clock = clock
        self._series = dict(series)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._series)

    def series_for(self, symbol: str) -> BarSeries:
        """Raw series. Engine and tooling only - never hand this to a strategy."""
        try:
            return self._series[symbol]
        except KeyError:
            raise KeyError(f"no data loaded for symbol {symbol!r}") from None

    def view(self, symbol: str) -> PointInTimeView:
        return PointInTimeView(self.series_for(symbol), self._clock)

    def bars_until(self, symbol: str, ts: datetime) -> tuple[Bar, ...]:
        """Bars closed at or before `ts`; raises if `ts` is in the future."""
        self._clock.assert_visible(ts, what=f"{symbol} bars")
        series = self.series_for(symbol)
        cursor = series.index_visible_at(ts)
        return tuple(series[i] for i in range(cursor + 1))
