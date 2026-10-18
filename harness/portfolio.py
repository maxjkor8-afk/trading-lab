"""Orders, fills, and position/cash accounting.

Average-cost accounting, single instrument per backtest. Spot is long-only and
pays no funding; perp may go short and settles funding. There is no margin
engine and no liquidation model here - that belongs behind the Stage 4 risk
gate, which per RESEARCH.md section 4 must be plain code outside any agent loop.
Until that exists, treat perp leverage results from this harness as optimistic
in one specific way: nothing stops a position that would have been liquidated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from harness.clock import ensure_utc
from harness.enums import InstrumentType, Liquidity, Side

_EPS = 1e-12
"""Quantities below this are treated as flat. Floating-point residue from a
percent-of-equity sell should not leave a position of 1e-17 open."""

__all__ = [
    "ClosedTrade",
    "Fill",
    "InstrumentType",
    "Order",
    "Portfolio",
    "Side",
]


class InsufficientFunds(RuntimeError):
    """The order would have spent cash the portfolio does not have."""


@dataclass(frozen=True, slots=True)
class Order:
    """A market order to be filled on the next bar open.

    Exactly one sizing field must be set. `percent_of_equity` is resolved at
    fill time against the fill price, not at submission time, because the
    submitting strategy does not know the fill price yet - and must not.
    """

    side: Side
    quantity: float | None = None
    notional: float | None = None
    percent_of_equity: float | None = None
    liquidity: Liquidity | None = None
    reduce_only: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        sizes = [self.quantity, self.notional, self.percent_of_equity]
        given = [s for s in sizes if s is not None]
        if len(given) != 1:
            raise ValueError(
                "Order needs exactly one of quantity, notional or "
                f"percent_of_equity; got {len(given)}"
            )
        if given[0] <= 0:
            raise ValueError(
                f"order size must be positive, got {given[0]}; use side=SELL to sell"
            )
        if self.percent_of_equity is not None and self.percent_of_equity > 1.0:
            raise ValueError(
                f"percent_of_equity={self.percent_of_equity} implies leverage; "
                "size it explicitly with notional= if that is intended"
            )

    @classmethod
    def buy(cls, **kwargs) -> Order:
        return cls(side=Side.BUY, **kwargs)

    @classmethod
    def sell(cls, **kwargs) -> Order:
        return cls(side=Side.SELL, **kwargs)


@dataclass(frozen=True, slots=True)
class Fill:
    """What actually happened, including what it cost to make it happen."""

    ts: datetime
    side: Side
    quantity: float
    reference_price: float
    price: float
    fee: float
    slippage_cost: float
    liquidity: Liquidity
    tag: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def slippage_bps(self) -> float:
        if self.reference_price == 0:
            return 0.0
        return abs(self.price - self.reference_price) / self.reference_price / 1e-4

    @property
    def total_cost(self) -> float:
        return self.fee + self.slippage_cost


@dataclass(slots=True)
class ClosedTrade:
    """One flat-to-flat round trip, net of the costs incurred inside it.

    Win rate is computed over these rather than over daily equity changes: the
    share of *trades* that made money is the number a trader means, and it does
    not flatter a strategy that holds a losing position through many bars.
    """

    opened_at: datetime
    closed_at: datetime
    direction: Side
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0

    @property
    def costs(self) -> float:
        return self.fees + self.slippage + self.funding

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


@dataclass(slots=True)
class _OpenTrade:
    opened_at: datetime
    direction: Side
    quantity: float = 0.0
    entry_notional: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    last_exit_price: float = 0.0
    last_ts: datetime | None = None


class Portfolio:
    """Cash, one position, and a running tally of every cost paid.

    The cost tally is the point. `total_costs` divided by gross PnL is how you
    see a strategy that is only profitable before fees, which is the single most
    useful diagnostic this harness produces.
    """

    __slots__ = (
        "instrument",
        "initial_cash",
        "cash",
        "quantity",
        "avg_entry_price",
        "realized_pnl",
        "fees_paid",
        "slippage_paid",
        "funding_paid",
        "traded_notional",
        "fills",
        "closed_trades",
        "_open_trade",
    )

    def __init__(
        self, initial_cash: float, instrument: InstrumentType = InstrumentType.SPOT
    ) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")
        self.instrument = instrument
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.quantity = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        self.funding_paid = 0.0
        self.traded_notional = 0.0
        self.fills: list[Fill] = []
        self.closed_trades: list[ClosedTrade] = []
        self._open_trade: _OpenTrade | None = None

    # -- valuation ---------------------------------------------------------

    def position_value(self, mark_price: float) -> float:
        if self.instrument is InstrumentType.SPOT:
            return self.quantity * mark_price
        return (mark_price - self.avg_entry_price) * self.quantity

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position_value(mark_price)

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    @property
    def total_costs(self) -> float:
        return self.fees_paid + self.slippage_paid + self.funding_paid

    def net_pnl(self, mark_price: float) -> float:
        return self.equity(mark_price) - self.initial_cash

    def gross_pnl(self, mark_price: float) -> float:
        """PnL before costs. Net plus everything the costs took out."""
        return self.net_pnl(mark_price) + self.total_costs

    # -- sizing ------------------------------------------------------------

    def resolve_quantity(self, order: Order, fill_price: float, fee_rate: float) -> float:
        """Turn an order's sizing into base-unit quantity at `fill_price`.

        For `percent_of_equity` the fee is solved for, not ignored: spending the
        whole balance means `qty * price * (1 + fee_rate) == cash`, otherwise the
        first fill of a buy-and-hold run overdraws by the fee and the harness has
        to paper over it.
        """
        if order.quantity is not None:
            qty = order.quantity
        elif order.notional is not None:
            qty = order.notional / fill_price
        else:
            pct = order.percent_of_equity or 0.0
            if order.side is Side.SELL and self.quantity > 0:
                qty = abs(self.quantity) * pct
            else:
                budget = self.equity(fill_price) * pct
                qty = budget / (fill_price * (1.0 + fee_rate))
        if order.reduce_only:
            qty = min(qty, abs(self.quantity))
        return qty

    # -- mutation ----------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Settle a fill into cash, position, cost tallies and trade tracking."""
        if fill.quantity <= 0:
            raise ValueError(f"fill quantity must be positive, got {fill.quantity}")
        signed = fill.quantity * fill.side.sign
        old_qty = self.quantity
        new_qty = old_qty + signed

        if self.instrument is InstrumentType.SPOT:
            if new_qty < -_EPS:
                raise ValueError(
                    f"spot cannot go short: holding {old_qty:g}, "
                    f"fill of {signed:g} would leave {new_qty:g}"
                )
            cash_delta = -signed * fill.price - fill.fee
            if self.cash + cash_delta < -1e-9:
                raise InsufficientFunds(
                    f"fill needs {-cash_delta:,.2f} but only {self.cash:,.2f} in cash"
                )
            self.cash += cash_delta
        else:
            self.cash -= fill.fee

        opening = old_qty == 0.0 or (old_qty > 0) == (signed > 0)

        if opening:
            self.avg_entry_price = (
                self.avg_entry_price * old_qty + fill.price * signed
            ) / new_qty
            self.quantity = new_qty
            closing_qty = 0.0
            realized = 0.0
        else:
            closing_qty = min(abs(signed), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realized = (fill.price - self.avg_entry_price) * closing_qty * direction
            self.realized_pnl += realized
            if self.instrument is InstrumentType.PERP:
                self.cash += realized
            self.quantity = 0.0 if abs(new_qty) < _EPS else new_qty
            if self.quantity == 0.0:
                self.avg_entry_price = 0.0
            elif abs(signed) > abs(old_qty):
                # Flipped through flat: the remainder is a new position opened at
                # this fill's price.
                self.avg_entry_price = fill.price

        self.fees_paid += fill.fee
        self.slippage_paid += fill.slippage_cost
        self.traded_notional += fill.quantity * fill.price
        self.fills.append(fill)
        self._track(fill, opening=opening, closing_qty=closing_qty, realized=realized)

    def apply_funding(self, ts: datetime, payment: float, mark_price: float) -> None:
        """Settle one funding payment. `payment` negative means you paid."""
        if self.instrument is not InstrumentType.PERP:
            raise ValueError("funding applies to perpetuals only")
        self.cash += payment
        self.funding_paid += -payment
        if self._open_trade is not None:
            self._open_trade.funding += -payment
            self._open_trade.last_ts = ensure_utc(ts)

    # -- round-trip tracking ----------------------------------------------

    def _track(
        self, fill: Fill, *, opening: bool, closing_qty: float, realized: float
    ) -> None:
        """Maintain the flat-to-flat round trip that `win_rate` is computed over.

        A fill that flips the position (sell 25 while long 10) ends one trade and
        starts another, so its fee and slippage are split pro-rata by quantity
        between the two. Charging the whole cost to either side would misstate
        both.
        """
        if opening:
            if self._open_trade is None:
                self._open_trade = _OpenTrade(
                    opened_at=fill.ts, direction=fill.side, last_exit_price=fill.price
                )
            trade = self._open_trade
            trade.quantity += fill.quantity
            trade.entry_notional += fill.quantity * fill.price
            trade.fees += fill.fee
            trade.slippage += fill.slippage_cost
            trade.last_ts = fill.ts
            return

        trade = self._open_trade
        if trade is None:
            # Reducing a position the tracker never saw opened (a portfolio built
            # mid-flight). Nothing to attribute the round trip to.
            return

        closing_share = closing_qty / fill.quantity if fill.quantity else 1.0
        trade.gross_pnl += realized
        trade.fees += fill.fee * closing_share
        trade.slippage += fill.slippage_cost * closing_share
        trade.last_exit_price = fill.price
        trade.last_ts = fill.ts

        if self.quantity != 0.0 and closing_share >= 1.0:
            return  # partial reduce; the trade is still open

        self._close_trade(trade, fill)

        remainder = fill.quantity - closing_qty
        if remainder > _EPS:
            self._open_trade = _OpenTrade(
                opened_at=fill.ts,
                direction=fill.side,
                quantity=remainder,
                entry_notional=remainder * fill.price,
                fees=fill.fee * (1.0 - closing_share),
                slippage=fill.slippage_cost * (1.0 - closing_share),
                last_exit_price=fill.price,
                last_ts=fill.ts,
            )

    def _close_trade(self, trade: _OpenTrade, fill: Fill) -> None:
        entry_price = (
            trade.entry_notional / trade.quantity if trade.quantity else fill.price
        )
        self.closed_trades.append(
            ClosedTrade(
                opened_at=trade.opened_at,
                closed_at=fill.ts,
                direction=trade.direction,
                quantity=trade.quantity,
                entry_price=entry_price,
                exit_price=trade.last_exit_price,
                gross_pnl=trade.gross_pnl,
                fees=trade.fees,
                slippage=trade.slippage,
                funding=trade.funding,
            )
        )
        self._open_trade = None

    def __repr__(self) -> str:
        return (
            f"Portfolio({self.instrument.value}, cash={self.cash:,.2f}, "
            f"qty={self.quantity:g} @ {self.avg_entry_price:,.2f}, "
            f"costs={self.total_costs:,.2f})"
        )
