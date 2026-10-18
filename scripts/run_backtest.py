"""Run a backtest from a CSV of bars.

    python scripts/run_backtest.py --data data/BTC-USD-1d-2023.csv
    python scripts/run_backtest.py --data data/BTC-USD-1d-2023.csv --zero-costs
    python scripts/run_backtest.py --data data/BTC-USD-1d-2023.csv \
        --instrument perp --venue bybit --funding-rate 0.0001

Defaults are pessimistic. `--zero-costs` exists to verify harness arithmetic
against a hand calculation and is labelled as such in the output, because a
number produced with costs switched off is not an evaluation of anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.costs import ZERO_COSTS, pessimistic_costs
from harness.data import BarSeries
from harness.engine import BacktestConfig, run_backtest
from harness.enums import InstrumentType
from harness.strategies import BuyAndHold

STRATEGIES = {"buy_and_hold": BuyAndHold}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="CSV from fetch_ohlcv.py")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="buy_and_hold")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--cash", type=float, default=10_000.0)
    ap.add_argument(
        "--instrument", choices=[i.value for i in InstrumentType], default="spot"
    )
    ap.add_argument("--venue", default=None, help="binance or bybit; default = worst")
    ap.add_argument("--half-spread-bps", type=float, default=5.0)
    ap.add_argument("--depth-quote", type=float, default=250_000.0)
    ap.add_argument("--depth-bps", type=float, default=10.0)
    ap.add_argument(
        "--funding-rate", type=float, default=1e-4, help="per 8h, perp only"
    )
    ap.add_argument(
        "--zero-costs",
        action="store_true",
        help="control run: no fees, no slippage, no funding",
    )
    ap.add_argument("--equity-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found; fetch it with scripts/fetch_ohlcv.py")

    instrument = InstrumentType(args.instrument)
    series = BarSeries.from_csv(args.data, symbol=args.symbol or args.data.stem)

    costs = (
        ZERO_COSTS
        if args.zero_costs
        else pessimistic_costs(
            args.venue,
            instrument,
            half_spread_bps=args.half_spread_bps,
            depth_quote=args.depth_quote,
            depth_bps=args.depth_bps,
            funding_rate=args.funding_rate,
        )
    )
    config = BacktestConfig(initial_cash=args.cash, instrument=instrument, costs=costs)
    result = run_backtest(series, STRATEGIES[args.strategy](), config)

    print(f"strategy          {args.strategy}")
    print(result.report())
    if result.fills:
        first = result.fills[0]
        print(
            f"\nfirst fill        {first.ts.isoformat()} "
            f"{first.side.value} {first.quantity:.8f} @ {first.price:,.2f} "
            f"(ref {first.reference_price:,.2f}, {first.slippage_bps:.2f}bps slip)"
        )
    if args.zero_costs:
        print(
            "\nNOTE: --zero-costs is a control for checking the harness arithmetic. "
            "It is not an evaluation of the strategy."
        )
    if args.equity_out:
        print(f"equity curve      {result.write_equity_csv(args.equity_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
