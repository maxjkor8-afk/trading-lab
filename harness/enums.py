"""Small shared enums, kept in their own module so `costs` and `portfolio` can
both import them without a cycle."""

from __future__ import annotations

from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Used for both position and price direction."""
        return 1 if self is Side.BUY else -1

    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class InstrumentType(Enum):
    """Spot cannot be shorted and pays no funding; perp can and does."""

    SPOT = "spot"
    PERP = "perp"


class Liquidity(Enum):
    """Whether an order took liquidity (crossed the spread) or made it.

    The harness assumes TAKER unless told otherwise. Assuming maker fills is the
    second OpenPM failure mode (optimistic execution): a resting order only fills
    if the market comes to you, which is precisely when you least want it to.
    """

    MAKER = "maker"
    TAKER = "taker"
