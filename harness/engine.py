"""The event-driven backtest loop.

One bar at a time, in this order:

1. advance the clock to the bar's **close** time (the bar is now known)
2. fill orders queued on the previous bar, at **this bar's open**
3. settle any funding timestamps crossed since the previous bar close
4. record the equity point, marked at this bar's close
5. call the strategy, which sees bars 0..i and may queue orders for the next bar

Step 5 happening last is what makes the loop leak-free: a decision taken on the
close of bar i cannot be executed before the open of bar i+1, so no order ever
fills at a price the strategy had already observed. There is deliberately no
option to fill at the close of the current bar. That option is the difference
between most published backtests and a true one, and offering it as a flag
would mean offering a way to quietly produce the wrong number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from harness.clock import Clock
from harness.costs import CostModel, pessimistic_costs
from harness.data import Bar, BarSeries, PointInTimeFeed, PointInTimeView
from harness.enums import InstrumentType, Liquidity, Side
from harness.metrics import EquityPoint, Metrics, compute_metrics
from harness.portfolio import Fill, Order, Portfolio

MIN_QUANTITY = 1e-12


class FillTiming(Enum):
    """When a queued order is allowed to fill. Both options are leak-free."""

    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


@dataclass(frozen=True, slots=True)
class RejectedOrder:
    ts: datetime
    order: Order
    reason: str


class StrategyContext:
    """Everything a strategy may touch, and nothing else.

    It holds no reference the strategy can follow to the full bar series: `view`
    is a `PointInTimeView` bounded by the clock, and the engine keeps the raw
    series to itself.
    """

    __slots__ = ("_engine", "view")

    def __init__(self, engine: "_Engine", view: PointInTimeView) -> None:
        self._engine = engine
        self.view = view

    # -- state you are allowed to see --------------------------------------

    @property
    def now(self) -> datetime:
        return self._engine.clock.now

    @property
    def bar(self) -> Bar:
        """The bar that just closed."""
        return self.view.current

    @property
    def position(self) -> float:
        return self._engine.portfolio.quantity

    @property
    def avg_entry_price(self) -> float:
        return self._engine.portfolio.avg_entry_price

    @property
    def cash(self) -> float:
        return self._engine.portfolio.cash

    @property
    def equity(self) -> float:
        """Equity marked at the close that just printed."""
        return self._engine.portfolio.equity(self.view.current.close)

    @property
    def is_flat(self) -> bool:
        return self._engine.portfolio.is_flat

    @property
    def pending_orders(self) -> tuple[Order, ...]:
        return tuple(self._engine.pending)

    # -- what you can do ---------------------------------------------------

    def submit(self, order: Order) -> None:
        """Queue an order. It fills on the next bar, never on this one."""
        self._engine.pending.append(order)

    def buy(self, **kwargs) -> None:
        self.submit(Order.buy(**kwargs))

    def sell(self, **kwargs) -> None:
        self.submit(Order.sell(**kwargs))

    def close_position(self, tag: str = "close") -> None:
        qty = self._engine.portfolio.quantity
        if abs(qty) <= MIN_QUANTITY:
            return
        side = Side.SELL if qty > 0 else Side.BUY
        self.submit(
            Order(side=side, quantity=abs(qty), reduce_only=True, tag=tag)
        )


@runtime_checkable
class Strategy(Protocol):
    def on_bar(self, ctx: StrategyContext) -> None: ...


StrategyLike = Strategy | Callable[[StrategyContext], None]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Run parameters. Costs default to pessimistic, deliberately.

    `warmup_bars` suppresses the strategy callback for the first N bars while
    still recording equity, so an indicator that needs 20 bars of history does
    not have to defend itself against being called on bar 3.
    """

    initial_cash: float = 10_000.0
    instrument: InstrumentType = InstrumentType.SPOT
    costs: CostModel = field(default_factory=pessimistic_costs)
    fill_timing: FillTiming = FillTiming.NEXT_OPEN
    warmup_bars: int = 0
    risk_free_rate: float = 0.0
    periods_per_year: float | None = None

    def __post_init__(self) -> None:
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars cannot be negative")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        # A non-zero funding rate on a SPOT run is inert rather than wrong: spot
        # never settles funding, so the engine skips it. Worth knowing when
        # reading a config that looks like it is charging funding.


@dataclass(frozen=True, slots=True)
class BacktestResult:
    symbol: str
    metrics: Metrics
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]
    rejected: tuple[RejectedOrder, ...]
    unfilled_at_end: tuple[Order, ...]
    portfolio: Portfolio
    config: BacktestConfig

    @property
    def cost_description(self) -> str:
        return self.config.costs.describe()

    def report(self) -> str:
        head = f"{self.symbol}  [{self.cost_description}]"
        tail = ""
        if self.unfilled_at_end:
            tail = (
                f"\nnote: {len(self.unfilled_at_end)} order(s) queued on the final "
                "bar never filled - there was no next bar to fill them on"
            )
        return f"{head}\n{self.metrics.report()}{tail}"

    def write_equity_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["ts,equity,mark_price,position,cash"]
        for p in self.equity_curve:
            lines.append(
                f"{p.ts.isoformat()},{p.equity:.10f},{p.mark_price:.10f},"
                f"{p.position:.12f},{p.cash:.10f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class _Engine:
    """Mutable run state. Not part of the public API."""

    __slots__ = ("clock", "portfolio", "pending", "config")

    def __init__(self, clock: Clock, portfolio: Portfolio, config: BacktestConfig) -> None:
        self.clock = clock
        self.portfolio = portfolio
        self.pending: list[Order] = []
        self.config = config


def _call_strategy(strategy: StrategyLike, hook: str, ctx: StrategyContext) -> None:
    fn = getattr(strategy, hook, None)
    if fn is not None:
        fn(ctx)
    elif hook == "on_bar" and callable(strategy):
        strategy(ctx)


def run_backtest(
    data: BarSeries | Sequence[Bar],
    strategy: StrategyLike,
    config: BacktestConfig | None = None,
    *,
    symbol: str = "ASSET",
) -> BacktestResult:
    """Walk `data` bar by bar, calling `strategy`, and record what happened."""
    config = config or BacktestConfig()
    series = data if isinstance(data, BarSeries) else BarSeries(symbol, data)
    if len(series) < 2:
        raise ValueError(
            "a backtest needs at least 2 bars: orders decided on one bar fill on "
            "the next, so a single-bar run can never execute anything"
        )

    clock = Clock()
    portfolio = Portfolio(config.initial_cash, config.instrument)
    engine = _Engine(clock, portfolio, config)
    feed = PointInTimeFeed(clock, series)
    view = feed.view(series.symbol)

    equity_curve: list[EquityPoint] = []
    rejected: list[RejectedOrder] = []
    started = False

    for index, bar in enumerate(series):
        clock.advance_to(bar.close_time)

        # 1. Fill what was decided on earlier bars.
        if engine.pending:
            reference = bar.open if config.fill_timing is FillTiming.NEXT_OPEN else bar.close
            fill_ts = bar.open_time if config.fill_timing is FillTiming.NEXT_OPEN else bar.close_time
            queued, engine.pending = engine.pending, []
            for order in queued:
                outcome = _execute(engine, order, reference, fill_ts)
                if outcome is not None:
                    rejected.append(outcome)

        # 2. Funding for the interval that just elapsed, on the position now held.
        if config.instrument is InstrumentType.PERP and not portfolio.is_flat:
            previous_close = series[index - 1].close_time if index > 0 else bar.open_time
            for event in config.costs.funding.events(previous_close, bar.close_time):
                payment = config.costs.funding.payment(
                    portfolio.quantity, bar.close, event.rate
                )
                portfolio.apply_funding(event.ts, payment, bar.close)

        # 3. Mark to this bar's close.
        equity_curve.append(
            EquityPoint(
                ts=bar.close_time,
                equity=portfolio.equity(bar.close),
                mark_price=bar.close,
                position=portfolio.quantity,
                cash=portfolio.cash,
            )
        )

        # 4. Hand the strategy a bounded view and let it decide.
        if index >= config.warmup_bars:
            ctx = StrategyContext(engine, view)
            if not started:
                _call_strategy(strategy, "on_start", ctx)
                started = True
            _call_strategy(strategy, "on_bar", ctx)

    final_ctx = StrategyContext(engine, view)
    _call_strategy(strategy, "on_finish", final_ctx)

    metrics = compute_metrics(
        equity_curve,
        portfolio,
        periods_per_year=config.periods_per_year,
        risk_free_rate=config.risk_free_rate,
    )
    return BacktestResult(
        symbol=series.symbol,
        metrics=metrics,
        equity_curve=tuple(equity_curve),
        fills=tuple(portfolio.fills),
        rejected=tuple(rejected),
        unfilled_at_end=tuple(engine.pending),
        portfolio=portfolio,
        config=config,
    )


def _execute(
    engine: _Engine, order: Order, reference_price: float, ts: datetime
) -> RejectedOrder | None:
    """Price, size and settle one order. Returns a rejection, or None on success."""
    costs = engine.config.costs
    portfolio = engine.portfolio
    liquidity = order.liquidity or costs.default_liquidity

    if order.reduce_only and portfolio.is_flat:
        return RejectedOrder(ts, order, "reduce_only order with no open position")

    # Sizing and slippage are circular: slippage depends on size, and
    # percent-of-equity size depends on the price after slippage. Resolve it by
    # pricing against the probe quantity, which is computed at the unslipped
    # reference price and is therefore always >= the quantity finally filled.
    # The fill is priced for a trade at least as large as the one that happens,
    # which keeps the error on the pessimistic side, and sizing against that
    # price leaves the order exactly affordable rather than overdrawing by the
    # fee. Re-pricing a second time at the smaller resolved quantity would be
    # more precise and slightly optimistic; precision is not worth that here.
    probe = _probe_quantity(order, reference_price, portfolio)
    price = costs.fill_price(order.side, reference_price, probe)
    fee_rate = costs.fees.rate(liquidity)
    quantity = portfolio.resolve_quantity(order, price, fee_rate)

    if quantity <= MIN_QUANTITY:
        return RejectedOrder(ts, order, f"resolved quantity {quantity:g} is zero")

    notional = quantity * price
    fee = costs.fee(notional, liquidity)
    slippage_cost = abs(price - reference_price) * quantity

    portfolio.apply_fill(
        Fill(
            ts=ts,
            side=order.side,
            quantity=quantity,
            reference_price=reference_price,
            price=price,
            fee=fee,
            slippage_cost=slippage_cost,
            liquidity=liquidity,
            tag=order.tag,
        )
    )
    return None


def _probe_quantity(order: Order, reference_price: float, portfolio: Portfolio) -> float:
    """A first guess at size, so the slippage model has something to price."""
    if order.quantity is not None:
        return order.quantity
    if order.notional is not None:
        return order.notional / reference_price
    pct = order.percent_of_equity or 0.0
    if order.side is Side.SELL and portfolio.quantity > 0:
        return abs(portfolio.quantity) * pct
    return portfolio.equity(reference_price) * pct / reference_price
