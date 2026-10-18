"""Cost model: fees, slippage, funding.

This module exists because of OpenPM failure mode #2 (RESEARCH.md section 2f):
reported LLM-trading results assume fills at prices you would not get. The
TradingAgents paper reports a Sharpe of 8.21 and does not model transaction
costs at all.

Every number here is configurable and every default is pessimistic:

* fees default to the **taker** side of the base (VIP 0, no token discount) tier
* slippage is always adverse - it never improves your price
* funding defaults to being charged against you regardless of position sign

Fee schedules below are the published base tiers as understood on 2026-10-18.
They are a starting point, not a source of truth: venues change them, your tier
changes them, and a wrong fee assumption is exactly the kind of quiet error this
harness exists to catch. Verify against the venue before trusting a result, and
override via `FeeSchedule(...)`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from harness.clock import ensure_utc
from harness.enums import InstrumentType, Liquidity, Side

BPS = 1e-4
"""One basis point. Fees and slippage are quoted in bps throughout."""


class InsufficientLiquidity(RuntimeError):
    """The order was larger than the book supplied to the slippage model."""


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Maker/taker fees in basis points of traded notional.

    Negative bps (maker rebates, which some perp venues pay) are allowed, but
    note that modelling a rebate means assuming a maker fill, which the harness
    does not do by default.
    """

    maker_bps: float
    taker_bps: float
    venue: str = "custom"
    instrument: InstrumentType = InstrumentType.SPOT
    note: str = ""

    def rate(self, liquidity: Liquidity) -> float:
        bps = self.maker_bps if liquidity is Liquidity.MAKER else self.taker_bps
        return bps * BPS

    def fee(self, notional: float, liquidity: Liquidity = Liquidity.TAKER) -> float:
        """Fee on `notional` (absolute quote value). Positive means you pay."""
        return abs(notional) * self.rate(liquidity)


VENUE_FEES: Mapping[str, FeeSchedule] = {
    "binance_spot": FeeSchedule(
        maker_bps=10.0,
        taker_bps=10.0,
        venue="binance",
        instrument=InstrumentType.SPOT,
        note="base/VIP0 spot, no BNB discount (0.10%/0.10%)",
    ),
    "binance_perp": FeeSchedule(
        maker_bps=2.0,
        taker_bps=5.0,
        venue="binance",
        instrument=InstrumentType.PERP,
        note="base/VIP0 USD-M perpetual (0.02%/0.05%)",
    ),
    "bybit_spot": FeeSchedule(
        maker_bps=10.0,
        taker_bps=10.0,
        venue="bybit",
        instrument=InstrumentType.SPOT,
        note="non-VIP spot (0.10%/0.10%)",
    ),
    "bybit_perp": FeeSchedule(
        maker_bps=2.0,
        taker_bps=5.5,
        venue="bybit",
        instrument=InstrumentType.PERP,
        note="non-VIP USDT linear perpetual (0.02%/0.055%)",
    ),
}

ZERO_FEES = FeeSchedule(maker_bps=0.0, taker_bps=0.0, venue="none", note="fees off")


def worst_fees(instrument: InstrumentType | None = None) -> FeeSchedule:
    """The harshest taker fee in `VENUE_FEES`, optionally within an instrument.

    Used as the no-argument default so that forgetting to pick a venue can only
    ever overstate costs, never understate them.
    """
    candidates = [
        f
        for f in VENUE_FEES.values()
        if instrument is None or f.instrument is instrument
    ]
    if not candidates:
        raise ValueError(f"no fee schedule for instrument {instrument}")
    return max(candidates, key=lambda f: f.taker_bps)


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


@runtime_checkable
class SlippageModel(Protocol):
    """Maps an intended trade onto the price you actually get.

    Implementations must be adverse: a BUY never fills below the reference price
    and a SELL never fills above it.
    """

    def fill_price(
        self, side: Side, reference_price: float, quantity: float
    ) -> float: ...

    def slippage_bps(self, side: Side, reference_price: float, quantity: float) -> float: ...


@dataclass(frozen=True, slots=True)
class NoSlippage:
    """Fills exactly at the reference price. For the zero-cost control only.

    Do not use this to evaluate a strategy. It is here so the acceptance test can
    isolate the harness arithmetic from the cost model.
    """

    def slippage_bps(self, side: Side, reference_price: float, quantity: float) -> float:
        return 0.0

    def fill_price(self, side: Side, reference_price: float, quantity: float) -> float:
        return reference_price


@dataclass(frozen=True, slots=True)
class FixedSpreadSlippage:
    """Cross half the spread, size-independent.

    The honest baseline for small orders in a deep book. It understates cost for
    size, which is why `LinearBookSlippage` is the default.
    """

    half_spread_bps: float = 5.0

    def slippage_bps(self, side: Side, reference_price: float, quantity: float) -> float:
        return self.half_spread_bps

    def fill_price(self, side: Side, reference_price: float, quantity: float) -> float:
        return reference_price * (1.0 + side.sign * self.half_spread_bps * BPS)


@dataclass(frozen=True, slots=True)
class LinearBookSlippage:
    """Walk a uniform-density book: slippage grows with size relative to depth.

    Model: `depth_quote` of quote-currency notional rests within `depth_bps` of
    mid on each side, spread uniformly. Filling notional N therefore walks to a
    worst price of `k * N` bps where `k = depth_bps / depth_quote`, and pays the
    average over the walk, `k * N / 2`. Beyond the band the book is assumed
    `overflow_density` times as liquid (default 0.25, i.e. four times steeper),
    because real books thin out past the top levels.

    Total cost in bps is `half_spread_bps + average impact`.

    This is a model, not a measurement. Its purpose is to make size cost
    something, so that a strategy cannot earn its Sharpe by assuming it can push
    an arbitrary notional through the top of the book. Replace it with
    `walk_book` against real depth once Stage 1 is persisting order books -
    which, per RESEARCH.md section 2d, should be read with the knowledge that a
    large share of visible crypto depth is not real.
    """

    half_spread_bps: float = 5.0
    depth_quote: float = 250_000.0
    depth_bps: float = 10.0
    overflow_density: float = 0.25

    def __post_init__(self) -> None:
        if self.depth_quote <= 0:
            raise ValueError("depth_quote must be positive")
        if self.depth_bps < 0:
            raise ValueError("depth_bps cannot be negative")
        if not 0 < self.overflow_density <= 1:
            raise ValueError("overflow_density must be in (0, 1]")
        if self.half_spread_bps < 0:
            raise ValueError("half_spread_bps cannot be negative")

    def impact_bps(self, notional: float) -> float:
        """Average adverse impact, in bps, of pushing `notional` through the book."""
        n = abs(notional)
        if n == 0:
            return 0.0
        k = self.depth_bps / self.depth_quote
        if n <= self.depth_quote:
            return k * n / 2.0
        excess = n - self.depth_quote
        k_out = k / self.overflow_density
        inside_cost = (self.depth_bps / 2.0) * self.depth_quote
        outside_cost = (self.depth_bps + k_out * excess / 2.0) * excess
        return (inside_cost + outside_cost) / n

    def slippage_bps(self, side: Side, reference_price: float, quantity: float) -> float:
        return self.half_spread_bps + self.impact_bps(
            abs(quantity) * reference_price
        )

    def fill_price(self, side: Side, reference_price: float, quantity: float) -> float:
        bps = self.slippage_bps(side, reference_price, quantity)
        return reference_price * (1.0 + side.sign * bps * BPS)


@dataclass(frozen=True, slots=True)
class BookWalk:
    average_price: float
    filled_quantity: float
    levels_consumed: int
    unfilled_quantity: float


def walk_book(
    levels: Sequence[tuple[float, float]],
    quantity: float,
    *,
    allow_partial: bool = False,
) -> BookWalk:
    """Fill `quantity` against real book levels `[(price, size), ...]`, best first.

    Used when actual depth is available instead of a parametric model. Raises on
    insufficient depth unless `allow_partial`, because an order silently
    truncated to the size of the book is a fill you did not get.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    remaining = quantity
    cost = 0.0
    consumed = 0
    for price, size in levels:
        if price <= 0 or size < 0:
            raise ValueError(f"invalid book level (price={price}, size={size})")
        if remaining <= 0:
            break
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        consumed += 1
    filled = quantity - remaining
    if remaining > 0 and not allow_partial:
        raise InsufficientLiquidity(
            f"book holds {filled:g} of the requested {quantity:g}; "
            f"{remaining:g} unfilled"
        )
    if filled <= 0:
        raise InsufficientLiquidity("book is empty")
    return BookWalk(
        average_price=cost / filled,
        filled_quantity=filled,
        levels_consumed=consumed,
        unfilled_quantity=remaining,
    )


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------

RateSource = float | Mapping[datetime, float] | Callable[[datetime], float]


@dataclass(frozen=True, slots=True)
class FundingEvent:
    ts: datetime
    rate: float


@dataclass(frozen=True, slots=True)
class FundingModel:
    """Perpetual funding, charged on positions held across funding timestamps.

    Mechanics: every `interval`, a position pays `rate * notional` if the rate is
    positive and it is long, and receives it if short. Binance and Bybit both
    settle BTC perps every 8h at 00:00, 08:00 and 16:00 UTC, which is the
    default anchor here.

    `rates` may be a constant, a mapping of funding timestamp to rate, or a
    callable. The default constant of 1bp per 8h (~10.95%/yr paid by longs) is
    roughly the historical mean of positive BTC funding - a deliberately
    unhelpful assumption for a long-biased strategy.

    `always_adverse` (pessimistic default) charges `abs(rate) * abs(notional)`
    against the position whatever its sign. That is **not** how a venue works;
    it is a worst-case switch so that a result cannot depend on having been on
    the receiving side of funding. Turn it off for realistic mechanics, and if
    a strategy only works with it off, say so when reporting the number.
    """

    rates: RateSource = 1e-4
    interval: timedelta = timedelta(hours=8)
    anchor: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    always_adverse: bool = False
    missing_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.interval <= timedelta(0):
            raise ValueError("funding interval must be positive")
        object.__setattr__(self, "anchor", ensure_utc(self.anchor))

    def rate_at(self, ts: datetime) -> float:
        ts = ensure_utc(ts)
        rates = self.rates
        if isinstance(rates, (int, float)):
            return float(rates)
        if isinstance(rates, Mapping):
            return float(rates.get(ts, self.missing_rate))
        return float(rates(ts))

    def events(self, after: datetime, upto: datetime) -> tuple[FundingEvent, ...]:
        """Funding timestamps in the half-open interval (after, upto].

        Half-open on the left so that walking bar by bar never settles the same
        funding twice and never skips one.
        """
        after = ensure_utc(after)
        upto = ensure_utc(upto)
        if upto <= after:
            return ()
        step = self.interval.total_seconds()
        offset = (after - self.anchor).total_seconds()
        # first index strictly after `after`
        n = int(offset // step) + 1
        out: list[FundingEvent] = []
        while True:
            ts = self.anchor + timedelta(seconds=n * step)
            if ts > upto:
                break
            if ts > after:
                out.append(FundingEvent(ts=ts, rate=self.rate_at(ts)))
            n += 1
        return tuple(out)

    def payment(self, quantity: float, mark_price: float, rate: float) -> float:
        """Cash delta for holding `quantity` at `mark_price` through one funding.

        Negative is a cost to you. Sign convention: positive rate means longs
        pay, so a long position with a positive rate returns a negative number.
        """
        notional = quantity * mark_price
        if self.always_adverse:
            return -abs(rate) * abs(notional)
        return -rate * notional


ZERO_FUNDING = FundingModel(rates=0.0)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fees + slippage + funding, as one configurable object."""

    fees: FeeSchedule = field(default_factory=worst_fees)
    slippage: SlippageModel = field(default_factory=LinearBookSlippage)
    funding: FundingModel = field(default_factory=FundingModel)
    default_liquidity: Liquidity = Liquidity.TAKER

    def fill_price(self, side: Side, reference_price: float, quantity: float) -> float:
        if reference_price <= 0:
            raise ValueError(f"reference_price must be positive, got {reference_price}")
        price = self.slippage.fill_price(side, reference_price, quantity)
        # Adverse-only invariant: a slippage model must never improve a fill.
        if side is Side.BUY and price < reference_price:
            raise ValueError("slippage model improved a BUY price")
        if side is Side.SELL and price > reference_price:
            raise ValueError("slippage model improved a SELL price")
        return price

    def fee(self, notional: float, liquidity: Liquidity | None = None) -> float:
        return self.fees.fee(notional, liquidity or self.default_liquidity)

    def with_zero_fees(self) -> CostModel:
        return replace(self, fees=ZERO_FEES)

    def describe(self) -> str:
        slip = type(self.slippage).__name__
        return (
            f"fees={self.fees.venue}/{self.fees.instrument.value} "
            f"maker={self.fees.maker_bps}bps taker={self.fees.taker_bps}bps, "
            f"slippage={slip}, "
            f"funding={'adverse ' if self.funding.always_adverse else ''}"
            f"{self.funding.interval}, assume={self.default_liquidity.value}"
        )


ZERO_COSTS = CostModel(
    fees=ZERO_FEES,
    slippage=NoSlippage(),
    funding=ZERO_FUNDING,
)
"""The control case. Use it to verify harness arithmetic, never to evaluate.

Any strategy evaluated under ZERO_COSTS is being flattered. The acceptance test
uses it for exactly one purpose: proving buy-and-hold reproduces a return that
can be checked by hand from the raw CSV.
"""


def pessimistic_costs(
    venue: str | None = None,
    instrument: InstrumentType | None = None,
    *,
    half_spread_bps: float = 5.0,
    depth_quote: float = 250_000.0,
    depth_bps: float = 10.0,
    funding_rate: RateSource = 1e-4,
) -> CostModel:
    """Build a deliberately unfavourable cost model.

    With no arguments it picks the worst taker fee in the venue table, charges
    half a spread plus depth-aware impact, assumes taker fills, and charges
    funding adversely regardless of position sign. Each piece is overridable.
    """
    if venue is not None:
        key = venue if "_" in venue else f"{venue}_{(instrument or InstrumentType.SPOT).value}"
        try:
            fees = VENUE_FEES[key]
        except KeyError:
            raise KeyError(
                f"unknown venue {venue!r}; known: {sorted(VENUE_FEES)}"
            ) from None
    else:
        fees = worst_fees(instrument)
    return CostModel(
        fees=fees,
        slippage=LinearBookSlippage(
            half_spread_bps=half_spread_bps,
            depth_quote=depth_quote,
            depth_bps=depth_bps,
        ),
        funding=FundingModel(rates=funding_rate, always_adverse=True),
        default_liquidity=Liquidity.TAKER,
    )
