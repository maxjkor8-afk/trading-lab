"""The clock owns "now". Everything time-dependent asks it.

This is the load-bearing object for the no-look-ahead guarantee. Data access is
never parameterised by an arbitrary timestamp the caller chose; it is derived
from `Clock.now`, which only the backtest engine can advance, and only forwards.
"""

from __future__ import annotations

from datetime import datetime, timezone


class ClockError(RuntimeError):
    """The clock was used in an impossible way (e.g. asked to run backwards)."""


class LookAheadError(ClockError):
    """A caller tried to observe information that was not available yet.

    Raised rather than returning a silently-wrong value, because a backtest that
    quietly leaks is worse than one that crashes.
    """


def ensure_utc(ts: datetime) -> datetime:
    """Normalise to tz-aware UTC. Naive datetimes are rejected, not assumed.

    Guessing a timezone is exactly the sort of silent assumption that puts a bar
    on the wrong side of a funding timestamp.
    """
    if not isinstance(ts, datetime):
        raise TypeError(f"expected datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        raise ClockError(
            f"naive datetime {ts!r}: timestamps must be timezone-aware UTC"
        )
    return ts.astimezone(timezone.utc)


class Clock:
    """A monotonic simulation clock.

    `now` is the instant the simulation has reached. Information timestamped
    later than `now` does not exist yet, and `assert_visible` enforces that.
    """

    __slots__ = ("_now", "_started")

    def __init__(self, now: datetime | None = None) -> None:
        self._now: datetime | None = None if now is None else ensure_utc(now)
        self._started = now is not None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise ClockError("clock has not been started; no 'now' exists yet")
        return self._now

    def advance_to(self, ts: datetime) -> None:
        """Move the clock forward to `ts`. Backwards or sideways moves raise.

        Only the engine should call this. Rewinding a clock mid-run is how a
        backtest ends up replaying a bar it has already acted on.
        """
        ts = ensure_utc(ts)
        if self._now is not None and ts <= self._now:
            raise ClockError(
                f"clock cannot move from {self._now.isoformat()} to "
                f"{ts.isoformat()}: time must move strictly forward"
            )
        self._now = ts
        self._started = True

    def is_visible(self, ts: datetime) -> bool:
        """True if information timestamped `ts` is known at `now`."""
        return ensure_utc(ts) <= self.now

    def assert_visible(self, ts: datetime, what: str = "information") -> None:
        ts = ensure_utc(ts)
        if ts > self.now:
            raise LookAheadError(
                f"look-ahead blocked: {what} timestamped {ts.isoformat()} is not "
                f"available at simulation time {self.now.isoformat()}"
            )

    def __repr__(self) -> str:
        state = self._now.isoformat() if self._now else "unstarted"
        return f"Clock({state})"
