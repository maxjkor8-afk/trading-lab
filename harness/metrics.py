"""Performance metrics.

The one that matters most is `cost_ratio`: total costs paid as a share of gross
PnL. It is how you see a strategy that is only profitable before fees, which is
the usual condition of a strategy that looks profitable. A cost ratio above 1
means the costs ate more than the edge produced.

Every metric that cannot be computed returns `None` rather than 0.0 or inf. A
Sharpe of 0.0 reads as "flat"; a Sharpe of None reads as "not enough data",
which is the truth when a run has two bars in it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from harness.portfolio import ClosedTrade, Portfolio

SECONDS_PER_YEAR = 365.0 * 24 * 3600
"""Crypto trades every day, so a calendar year is the right denominator here -
not the 252 trading days an equities harness would use."""


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: datetime
    equity: float
    mark_price: float
    position: float
    cash: float


@dataclass(frozen=True, slots=True)
class Metrics:
    bars: int
    start: datetime | None
    end: datetime | None
    initial_equity: float
    final_equity: float
    cumulative_return: float
    annualized_return: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    max_drawdown_start: datetime | None
    max_drawdown_end: datetime | None
    win_rate: float | None
    trades: int
    turnover: float
    annualized_turnover: float | None
    total_costs: float
    fees_paid: float
    slippage_paid: float
    funding_paid: float
    gross_pnl: float
    net_pnl: float
    cost_ratio: float | None
    periods_per_year: float

    def report(self) -> str:
        """A compact text block. Deliberately shows costs next to returns."""

        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x * 100:,.2f}%"

        def num(x: float | None) -> str:
            return "n/a" if x is None else f"{x:,.2f}"

        span = ""
        if self.start and self.end:
            span = f"{self.start.date()} -> {self.end.date()} ({self.bars} bars)"
        lines = [
            f"period            {span}",
            f"equity            {self.initial_equity:,.2f} -> {self.final_equity:,.2f}",
            f"cumulative return {pct(self.cumulative_return)}",
            f"annualized return {pct(self.annualized_return)}",
            f"sharpe            {num(self.sharpe)}",
            f"sortino           {num(self.sortino)}",
            f"max drawdown      {pct(self.max_drawdown)}",
            f"trades            {self.trades}",
            f"win rate          {pct(self.win_rate)}",
            f"turnover          {num(self.turnover)}x equity",
            f"net pnl           {self.net_pnl:,.2f}",
            f"gross pnl         {self.gross_pnl:,.2f}",
            f"  fees            {self.fees_paid:,.2f}",
            f"  slippage        {self.slippage_paid:,.2f}",
            f"  funding         {self.funding_paid:,.2f}",
            f"  total costs     {self.total_costs:,.2f}",
            f"costs / gross pnl {pct(self.cost_ratio)}",
        ]
        return "\n".join(lines)


def periodic_returns(equity: Sequence[float]) -> list[float]:
    """Simple period-over-period returns. Raises on a non-positive equity point.

    A backtest that passes through zero or negative equity has blown up, and
    continuing to compute a return series through that point produces a number
    that means nothing.
    """
    out: list[float] = []
    for prev, cur in zip(equity, equity[1:]):
        if prev <= 0:
            raise ValueError(
                "equity reached zero or below; the run is ruined and return "
                "statistics past that point are meaningless"
            )
        out.append(cur / prev - 1.0)
    return out


def max_drawdown(points: Sequence[EquityPoint]) -> tuple[float, datetime | None, datetime | None]:
    """Largest peak-to-trough decline as a positive fraction, with its dates."""
    peak = -math.inf
    peak_ts: datetime | None = None
    worst = 0.0
    worst_start: datetime | None = None
    worst_end: datetime | None = None
    for p in points:
        if p.equity > peak:
            peak = p.equity
            peak_ts = p.ts
        if peak > 0:
            dd = (peak - p.equity) / peak
            if dd > worst:
                worst = dd
                worst_start = peak_ts
                worst_end = p.ts
    return worst, worst_start, worst_end


def sharpe_ratio(
    returns: Sequence[float], periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Annualised Sharpe on periodic returns, sample stdev (ddof=1).

    `risk_free_rate` is an annual rate, converted to per-period simply. Returns
    None for fewer than two observations or zero dispersion, because a Sharpe
    with no variance in the denominator is a division artefact, not a result.
    """
    if len(returns) < 2:
        return None
    rf = risk_free_rate / periods_per_year
    excess = [r - rf for r in returns]
    sd = statistics.stdev(excess)
    if sd == 0:
        return None
    return statistics.fmean(excess) / sd * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: Sequence[float],
    periods_per_year: float,
    risk_free_rate: float = 0.0,
    target: float = 0.0,
) -> float | None:
    """Annualised Sortino: downside deviation below `target` in the denominator.

    Downside deviation is the root mean square of shortfalls, computed over the
    full sample (not only the losing periods) - the standard definition. Returns
    None if nothing fell below target, since there is no measured downside risk
    to divide by.
    """
    if len(returns) < 2:
        return None
    rf = risk_free_rate / periods_per_year
    excess = [r - rf for r in returns]
    shortfalls = [min(0.0, r - target) for r in excess]
    dd = math.sqrt(sum(s * s for s in shortfalls) / len(excess))
    if dd == 0:
        return None
    return statistics.fmean(excess) / dd * math.sqrt(periods_per_year)


def annualized_return(
    cumulative_return: float, elapsed: timedelta
) -> float | None:
    """CAGR from actual elapsed wall-clock time.

    Note how badly this behaves on short windows, which is the point: the
    TradingAgents 3-month backtest in RESEARCH.md section 2a annualises a
    favourable quarter into a number no one should rely on.
    """
    seconds = elapsed.total_seconds()
    if seconds <= 0:
        return None
    growth = 1.0 + cumulative_return
    if growth <= 0:
        return -1.0
    years = seconds / SECONDS_PER_YEAR
    return growth ** (1.0 / years) - 1.0


def win_rate(trades: Sequence[ClosedTrade]) -> float | None:
    """Share of completed round trips that made money net of costs."""
    if not trades:
        return None
    return sum(1 for t in trades if t.is_win) / len(trades)


def infer_periods_per_year(points: Sequence[EquityPoint]) -> float:
    """Periods per year from the median gap between equity points."""
    if len(points) < 2:
        return 365.0
    gaps = [
        (b.ts - a.ts).total_seconds() for a, b in zip(points, points[1:])
    ]
    median = statistics.median(g for g in gaps if g > 0)
    if median <= 0:
        return 365.0
    return SECONDS_PER_YEAR / median


def compute_metrics(
    points: Sequence[EquityPoint],
    portfolio: Portfolio,
    *,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> Metrics:
    if not points:
        raise ValueError("cannot compute metrics from an empty equity curve")
    ppy = periods_per_year or infer_periods_per_year(points)
    equity = [p.equity for p in points]
    initial = equity[0]
    final = equity[-1]
    cum = final / initial - 1.0 if initial > 0 else 0.0
    rets = periodic_returns(equity)
    dd, dd_start, dd_end = max_drawdown(points)

    elapsed = points[-1].ts - points[0].ts
    avg_equity = statistics.fmean(equity)
    turnover = portfolio.traded_notional / avg_equity if avg_equity > 0 else 0.0
    ann_turnover: float | None = None
    if elapsed.total_seconds() > 0:
        ann_turnover = turnover * SECONDS_PER_YEAR / elapsed.total_seconds()

    final_mark = points[-1].mark_price
    net = portfolio.net_pnl(final_mark)
    gross = portfolio.gross_pnl(final_mark)
    costs = portfolio.total_costs
    # Only meaningful when there was a gross profit to eat into. If gross PnL is
    # zero or negative the strategy had no edge to begin with and the ratio is
    # arithmetic noise, so report None and let the caller read gross_pnl.
    cost_ratio = costs / gross if gross > 0 else None

    return Metrics(
        bars=len(points),
        start=points[0].ts,
        end=points[-1].ts,
        initial_equity=initial,
        final_equity=final,
        cumulative_return=cum,
        annualized_return=annualized_return(cum, elapsed),
        sharpe=sharpe_ratio(rets, ppy, risk_free_rate),
        sortino=sortino_ratio(rets, ppy, risk_free_rate),
        max_drawdown=dd,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        win_rate=win_rate(portfolio.closed_trades),
        trades=len(portfolio.closed_trades),
        turnover=turnover,
        annualized_turnover=ann_turnover,
        total_costs=costs,
        fees_paid=portfolio.fees_paid,
        slippage_paid=portfolio.slippage_paid,
        funding_paid=portfolio.funding_paid,
        gross_pnl=gross,
        net_pnl=net,
        cost_ratio=cost_ratio,
        periods_per_year=ppy,
    )
