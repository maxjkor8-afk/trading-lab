"""Stage 0 evaluation harness: point-in-time, cost-aware, leakage-resistant.

Nothing in this package talks to an exchange, holds a key, or places an order.
It reads historical bars from disk and simulates fills against them.

The design answers the three failure modes catalogued in RESEARCH.md section 2f
(OpenPM, SSRN 7262158):

1. look-ahead leakage -> `clock` + `data`. The Clock owns "now"; every data
   access is bounds-checked against it, so a strategy cannot index the future.
2. optimistic execution -> `costs` + `engine`. Fees, depth-aware slippage and
   perp funding, pessimistic by default, with next-bar-open fills.
3. risk mandates described but not enforced -> out of scope for Stage 0 by
   design. The risk gate is Stage 4 and must be plain code outside any agent
   loop, not a prompt and not an agent that can be argued with.
"""

from harness.clock import Clock, ClockError, LookAheadError, ensure_utc
from harness.costs import (
    BookWalk,
    CostModel,
    FeeSchedule,
    FixedSpreadSlippage,
    FundingModel,
    InsufficientLiquidity,
    LinearBookSlippage,
    NoSlippage,
    VENUE_FEES,
    ZERO_COSTS,
    ZERO_FEES,
    ZERO_FUNDING,
    pessimistic_costs,
    walk_book,
    worst_fees,
)
from harness.data import Bar, BarSeries, PointInTimeFeed, PointInTimeView
from harness.engine import (
    BacktestConfig,
    BacktestResult,
    FillTiming,
    RejectedOrder,
    StrategyContext,
    run_backtest,
)
from harness.enums import InstrumentType, Liquidity, Side
from harness.metrics import EquityPoint, Metrics, compute_metrics
from harness.portfolio import ClosedTrade, Fill, InsufficientFunds, Order, Portfolio

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Bar",
    "BarSeries",
    "BookWalk",
    "Clock",
    "ClockError",
    "ClosedTrade",
    "CostModel",
    "EquityPoint",
    "FeeSchedule",
    "Fill",
    "FillTiming",
    "FixedSpreadSlippage",
    "FundingModel",
    "InstrumentType",
    "InsufficientFunds",
    "InsufficientLiquidity",
    "LinearBookSlippage",
    "Liquidity",
    "LookAheadError",
    "Metrics",
    "NoSlippage",
    "Order",
    "PointInTimeFeed",
    "PointInTimeView",
    "Portfolio",
    "RejectedOrder",
    "Side",
    "StrategyContext",
    "VENUE_FEES",
    "ZERO_COSTS",
    "ZERO_FEES",
    "ZERO_FUNDING",
    "compute_metrics",
    "ensure_utc",
    "pessimistic_costs",
    "run_backtest",
    "walk_book",
    "worst_fees",
]
