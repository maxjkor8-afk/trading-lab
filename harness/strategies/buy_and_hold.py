"""Buy once, hold to the end. The reference every strategy is measured against.

It is also the harness's own test instrument: with costs zeroed, its return must
equal `final_close / first_fill_price - 1`, a number anyone can check by hand
from two cells of the CSV. See tests/test_acceptance_buy_and_hold.py.
"""

from __future__ import annotations

from harness.engine import StrategyContext


class BuyAndHold:
    """Allocate `allocation` of equity on the first bar, then do nothing.

    The buy is submitted on the first bar the strategy sees and fills on the
    next bar's open, so the entry price is one the strategy had not observed when
    it decided.

    There is no liquidate-at-end option, and the omission is deliberate. A
    strategy cannot know which bar is the last one without look-ahead, so
    "sell everything on the final bar" is not a decision a strategy is able to
    make - and an order queued on the final bar has no next bar to fill on. The
    terminal position is instead marked to market at the final close, which is
    also the correct treatment for a benchmark nobody actually sells.
    """

    def __init__(self, allocation: float = 1.0) -> None:
        if not 0 < allocation <= 1:
            raise ValueError(f"allocation must be in (0, 1], got {allocation}")
        self.allocation = allocation
        self.entered = False

    def on_bar(self, ctx: StrategyContext) -> None:
        if self.entered:
            return
        ctx.buy(percent_of_equity=self.allocation, tag="buy_and_hold_entry")
        self.entered = True
