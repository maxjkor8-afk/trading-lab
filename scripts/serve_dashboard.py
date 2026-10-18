"""Serve an interactive dashboard for the harness on localhost.

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --port 8765 --no-browser

Standard library only, exactly like the harness: `http.server` and `json`, no
web framework. The charting library is fetched from a CDN by the page itself,
so the Python side of this repository still has zero third-party dependencies.

The dashboard is a viewer. Every number it draws comes back from the same
`run_backtest` call that `scripts/run_backtest.py` makes, so there is no second
implementation of the arithmetic that could drift from the tested one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.costs import ZERO_COSTS, pessimistic_costs
from harness.data import BarSeries
from harness.engine import BacktestConfig, run_backtest
from harness.enums import InstrumentType
from harness.strategies import BuyAndHold

DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "dashboard"
STRATEGIES = {"buy_and_hold": BuyAndHold}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


# -- serialisation ---------------------------------------------------------


def _epoch(ts: datetime) -> int:
    """Lightweight Charts wants whole UTC seconds."""
    return int(ts.astimezone(timezone.utc).timestamp())


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _metrics_dict(metrics) -> dict:
    return {
        f.name: _jsonable(getattr(metrics, f.name))
        for f in dataclasses.fields(metrics)
    }


def _drawdown_curve(equity_curve) -> list[dict]:
    """Percent below the running peak, which is what max drawdown measures."""
    out: list[dict] = []
    peak = float("-inf")
    for point in equity_curve:
        peak = max(peak, point.equity)
        out.append(
            {"time": _epoch(point.ts), "value": (point.equity / peak - 1.0) * 100.0}
        )
    return out


def _month_ends(equity_curve) -> list[tuple[int, int, float]]:
    """(year, month, closing equity) for each calendar month in the run."""
    out: list[tuple[int, int, float]] = []
    for point in equity_curve:
        key = (point.ts.year, point.ts.month)
        if out and (out[-1][0], out[-1][1]) == key:
            out[-1] = (key[0], key[1], point.equity)
        else:
            out.append((key[0], key[1], point.equity))
    return out


def _monthly_returns(equity_curve, initial_equity: float) -> list[dict]:
    rows = []
    previous = initial_equity
    for year, month, equity in _month_ends(equity_curve):
        rows.append({"year": year, "month": month, "value": equity / previous - 1.0})
        previous = equity
    return rows


def _eoy_returns(equity_curve, initial_equity: float) -> list[dict]:
    rows = []
    previous = initial_equity
    by_year: dict[int, float] = {}
    for year, _month, equity in _month_ends(equity_curve):
        by_year[year] = equity
    for year in sorted(by_year):
        rows.append({"year": year, "value": by_year[year] / previous - 1.0})
        previous = by_year[year]
    return rows


def _rolling_sharpe(equity_curve, periods_per_year: float, window: int = 60) -> list[dict]:
    """Annualised Sharpe over a trailing window, so a single good year cannot
    hide a long flat stretch inside it."""
    if len(equity_curve) <= window:
        return []
    equities = [p.equity for p in equity_curve]
    returns = [
        equities[i] / equities[i - 1] - 1.0
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    out = []
    root = periods_per_year**0.5
    for i in range(window, len(returns) + 1):
        chunk = returns[i - window : i]
        mean = sum(chunk) / window
        variance = sum((r - mean) ** 2 for r in chunk) / (window - 1)
        deviation = variance**0.5
        value = 0.0 if deviation == 0 else (mean / deviation) * root
        out.append({"time": _epoch(equity_curve[i].ts), "value": value})
    return out


def _worst_drawdowns(equity_curve, limit: int = 5) -> list[dict]:
    """Each peak-to-recovery episode, worst first. An episode still underwater
    at the end of the run is reported with no recovery date."""
    episodes: list[dict] = []
    peak = float("-inf")
    peak_ts = None
    current: dict | None = None
    for point in equity_curve:
        if point.equity >= peak:
            if current is not None:
                current["end"] = point.ts.isoformat()
                current["days"] = (point.ts - datetime.fromisoformat(current["start"])).days
                episodes.append(current)
                current = None
            peak = point.equity
            peak_ts = point.ts
            continue
        depth = point.equity / peak - 1.0
        if current is None:
            current = {
                "start": peak_ts.isoformat(),
                "valley": point.ts.isoformat(),
                "end": None,
                "depth": depth,
                "days": 0,
            }
        elif depth < current["depth"]:
            current["depth"] = depth
            current["valley"] = point.ts.isoformat()
    if current is not None:
        last = equity_curve[-1].ts
        current["days"] = (last - datetime.fromisoformat(current["start"])).days
        episodes.append(current)
    episodes.sort(key=lambda e: e["depth"])
    return episodes[:limit]


def _serialise(result, series: BarSeries, control_equity) -> dict:
    return {
        "symbol": result.symbol,
        "cost_description": result.cost_description,
        "report": result.report(),
        "metrics": _metrics_dict(result.metrics),
        "price": [
            {
                "time": _epoch(bar.close_time),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
            }
            for bar in series
        ],
        "equity": [
            {"time": _epoch(p.ts), "value": p.equity} for p in result.equity_curve
        ],
        "drawdown": _drawdown_curve(result.equity_curve),
        "monthly_returns": _monthly_returns(
            result.equity_curve, result.metrics.initial_equity
        ),
        "eoy_returns": _eoy_returns(result.equity_curve, result.metrics.initial_equity),
        "rolling_sharpe": _rolling_sharpe(
            result.equity_curve, result.metrics.periods_per_year
        ),
        "worst_drawdowns": _worst_drawdowns(result.equity_curve),
        "control_equity": control_equity,
        "fills": [
            {
                "time": _epoch(fill.ts),
                "iso": fill.ts.isoformat(),
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "reference_price": fill.reference_price,
                "slippage_bps": fill.slippage_bps,
                "fee": fill.fee,
                "slippage_cost": fill.slippage_cost,
                "tag": fill.tag,
            }
            for fill in result.fills
        ],
        "rejected": [
            {"iso": r.ts.isoformat(), "reason": r.reason} for r in result.rejected
        ],
        "unfilled_at_end": len(result.unfilled_at_end),
    }


# -- datasets --------------------------------------------------------------


def list_datasets() -> list[dict]:
    datasets = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        meta_path = csv_path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        datasets.append({"file": csv_path.name, "label": csv_path.stem, "meta": meta})
    return datasets


def _dataset_path(name: str) -> Path:
    """Resolve a dataset name to a CSV inside data/, and nowhere else."""
    candidate = (DATA_DIR / name).resolve()
    if (
        candidate.parent != DATA_DIR.resolve()
        or candidate.suffix != ".csv"
        or not candidate.exists()
    ):
        raise ValueError(f"unknown dataset {name!r}")
    return candidate


# -- query parameters ------------------------------------------------------


def _one(params: dict, key: str, default: str | None = None) -> str | None:
    values = params.get(key)
    return values[0] if values else default


def _number(params: dict, key: str, default: float) -> float:
    raw = _one(params, key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None


def _flag(params: dict, key: str) -> bool:
    return _one(params, key, "0") in ("1", "true", "on", "yes")


def run_payload(params: dict) -> dict:
    dataset = _one(params, "dataset")
    if not dataset:
        raise ValueError("dataset is required")
    path = _dataset_path(dataset)

    strategy_name = _one(params, "strategy", "buy_and_hold")
    if strategy_name not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy_name!r}")

    instrument = InstrumentType(_one(params, "instrument", "spot"))
    venue = _one(params, "venue") or None
    zero_costs = _flag(params, "zero_costs")
    cash = _number(params, "cash", 10_000.0)

    series = BarSeries.from_csv(path, symbol=path.stem)

    costs = (
        ZERO_COSTS
        if zero_costs
        else pessimistic_costs(
            venue,
            instrument,
            half_spread_bps=_number(params, "half_spread_bps", 5.0),
            depth_quote=_number(params, "depth_quote", 250_000.0),
            depth_bps=_number(params, "depth_bps", 10.0),
            funding_rate=_number(params, "funding_rate", 1e-4),
        )
    )
    config = BacktestConfig(initial_cash=cash, instrument=instrument, costs=costs)
    result = run_backtest(series, STRATEGIES[strategy_name](), config)

    # The control run is this repository's own argument made visible: the same
    # strategy with costs switched off, so the gap between the two lines IS the
    # cost of trading, drawn rather than asserted.
    control_equity = None
    control_metrics = None
    if _flag(params, "compare") and not zero_costs:
        control = run_backtest(
            series,
            STRATEGIES[strategy_name](),
            BacktestConfig(initial_cash=cash, instrument=instrument, costs=ZERO_COSTS),
        )
        control_equity = [
            {"time": _epoch(p.ts), "value": p.equity} for p in control.equity_curve
        ]
        control_metrics = _metrics_dict(control.metrics)

    payload = _serialise(result, series, control_equity)
    payload["control_metrics"] = control_metrics
    payload["zero_costs"] = zero_costs
    payload["dataset"] = dataset
    payload["strategy"] = strategy_name
    return payload


# -- http ------------------------------------------------------------------


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "trading-lab-dashboard"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                self._send_static(STATIC_DIR / "index.html")
            elif route in ("/app.js", "/styles.css"):
                self._send_static(STATIC_DIR / route.lstrip("/"))
            elif route == "/api/datasets":
                self._send_json({"datasets": list_datasets()})
            elif route == "/api/run":
                self._send_json(run_payload(parse_qs(parsed.query)))
            else:
                self._send_json({"error": f"no route {route}"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # surfaced in the UI rather than swallowed
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_static(self, path: Path) -> None:
        if not path.exists():
            self._send_json({"error": f"missing {path.name}"}, status=404)
            return
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), content_type)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  " + (fmt % args) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    args = ap.parse_args(argv)

    if not list_datasets():
        raise SystemExit(
            f"no CSVs in {DATA_DIR}; fetch one with scripts/fetch_ohlcv.py first"
        )

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"trading-lab dashboard on {url}")
    print("Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
