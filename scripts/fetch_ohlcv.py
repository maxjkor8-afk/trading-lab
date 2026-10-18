"""Fetch historical daily/hourly OHLCV from public endpoints. No keys, no orders.

Read-only HTTP GETs against public market-data endpoints. Nothing here
authenticates, and nothing here can place an order.

    python scripts/fetch_ohlcv.py --source coinbase --symbol BTC-USD \
        --start 2022-12-31 --end 2024-01-01 --out data/BTC-USD-1d-2023.csv

Two sources are supported on purpose. An acceptance gate that only ever sees one
venue's candles cannot tell a harness bug from a data bug, so `--source kraken`
exists to cross-check the same window from an unrelated venue:

    python scripts/fetch_ohlcv.py --source kraken --symbol XBTUSD \
        --start 2022-12-31 --end 2024-01-01 --out data/XBTUSD-1d-2023-kraken.csv

Note for this machine: api.binance.com returns HTTP 451 from US IP addresses, so
Binance is not usable as a source here. Its *fee schedule* is still modelled in
harness/costs.py - that is a published number, not an API call.

Output columns are `open_time,close_time,open,high,low,close,volume`, both
timestamps ISO-8601 UTC. `close_time` is what the harness uses for visibility:
a daily bar is not knowable until the day is over.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

USER_AGENT = "trading-lab-harness/0.1 (historical research; read-only)"
GRANULARITY = {"1d": 86400, "1h": 3600, "6h": 21600, "15m": 900, "1m": 60}


def _get(url: str, retries: int = 4) -> object:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 451:
                raise SystemExit(
                    f"HTTP 451 from {url.split('/')[2]}: this endpoint is "
                    "geo-blocked from your IP. Try --source coinbase or bitstamp."
                ) from exc
            last = exc
        except Exception as exc:  # pragma: no cover - network path
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"giving up on {url}: {last}")


def fetch_coinbase(
    symbol: str, start: datetime, end: datetime, interval: str
) -> list[dict]:
    """Coinbase Exchange public candles. Returns [time, low, high, open, close, volume]."""
    gran = GRANULARITY[interval]
    step = timedelta(seconds=gran * 280)  # endpoint caps at 300 candles per call
    rows: dict[int, dict] = {}
    cursor = start
    while cursor < end:
        window_end = min(cursor + step, end)
        url = (
            f"https://api.exchange.coinbase.com/products/{symbol}/candles"
            f"?granularity={gran}"
            f"&start={cursor.isoformat().replace('+00:00', 'Z')}"
            f"&end={window_end.isoformat().replace('+00:00', 'Z')}"
        )
        payload = _get(url)
        if not isinstance(payload, list):
            raise SystemExit(f"unexpected coinbase payload: {payload!r}")
        for item in payload:
            ts, low, high, open_, close, volume = item[:6]
            # `end` is exclusive here. Coinbase will hand back the bucket that
            # starts exactly at `end`, which would put an extra bar on the series
            # and silently shift any date-bounded acceptance test by a day.
            if not (start.timestamp() <= int(ts) < end.timestamp()):
                continue
            rows[int(ts)] = {
                "open_time": int(ts),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        cursor = window_end
        time.sleep(0.35)  # be polite to a free public endpoint
    return [rows[k] for k in sorted(rows)]


def fetch_bitstamp(
    symbol: str, start: datetime, end: datetime, interval: str
) -> list[dict]:
    """Bitstamp public OHLC. Oldest-first, `limit` caps at 1000 per call.

    Kraken is deliberately not offered as a source: its public OHLC endpoint
    ignores `since` for deep history and returns only the most recent ~720
    intervals, so asking it for 2023 hands back 2024-2026 data without an error.
    A source that silently returns the wrong window is worse than no source.
    """
    step = GRANULARITY[interval]
    rows: dict[int, dict] = {}
    cursor = int(start.timestamp())
    end_ts = int(end.timestamp())
    while cursor < end_ts:
        window_end = min(cursor + step * 1000, end_ts)
        url = (
            f"https://www.bitstamp.net/api/v2/ohlc/{symbol}/"
            f"?step={step}&limit=1000&start={cursor}&end={window_end}"
        )
        payload = _get(url)
        if not isinstance(payload, dict) or "data" not in payload:
            raise SystemExit(f"unexpected bitstamp payload: {payload!r}")
        batch = payload["data"].get("ohlc") or []
        if not batch:
            break
        for item in batch:
            ts = int(item["timestamp"])
            if not (int(start.timestamp()) <= ts < end_ts):
                continue
            rows[ts] = {
                "open_time": ts,
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": float(item["volume"]),
            }
        newest = max(int(i["timestamp"]) for i in batch)
        if newest + step <= cursor:
            break
        cursor = newest + step
        time.sleep(0.35)
    return [rows[k] for k in sorted(rows)]


SOURCES = {"coinbase": fetch_coinbase, "bitstamp": fetch_bitstamp}


def write_csv(rows: list[dict], path: Path, interval: str, source: str, symbol: str) -> Path:
    gran = GRANULARITY[interval]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["open_time", "close_time", "open", "high", "low", "close", "volume"]
        )
        for row in rows:
            open_dt = datetime.fromtimestamp(row["open_time"], tz=timezone.utc)
            close_dt = open_dt + timedelta(seconds=gran)
            writer.writerow(
                [
                    open_dt.isoformat(),
                    close_dt.isoformat(),
                    f"{row['open']:.8f}",
                    f"{row['high']:.8f}",
                    f"{row['low']:.8f}",
                    f"{row['close']:.8f}",
                    f"{row['volume']:.8f}",
                ]
            )
    meta = path.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "source": source,
                "symbol": symbol,
                "interval": interval,
                "bars": len(rows),
                "first_open": datetime.fromtimestamp(
                    rows[0]["open_time"], tz=timezone.utc
                ).isoformat(),
                "last_open": datetime.fromtimestamp(
                    rows[-1]["open_time"], tz=timezone.utc
                ).isoformat(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "Public historical endpoint, read-only. close_time = "
                    "open_time + interval; the harness treats close_time as the "
                    "moment the bar becomes knowable."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def parse_day(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), default="coinbase")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", choices=sorted(GRANULARITY), default="1d")
    ap.add_argument("--start", required=True, help="inclusive, UTC date e.g. 2022-12-31")
    ap.add_argument("--end", required=True, help="exclusive, UTC date e.g. 2024-01-01")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    start, end = parse_day(args.start), parse_day(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    rows = SOURCES[args.source](args.symbol, start, end, args.interval)
    if not rows:
        raise SystemExit("no candles returned; check the symbol and date range")

    gran = GRANULARITY[args.interval]
    expected = int((end - start).total_seconds() // gran)
    path = write_csv(rows, args.out, args.interval, args.source, args.symbol)

    first = datetime.fromtimestamp(rows[0]["open_time"], tz=timezone.utc)
    last = datetime.fromtimestamp(rows[-1]["open_time"], tz=timezone.utc)
    print(f"wrote {len(rows)} bars to {path}")
    print(f"  {first.isoformat()} .. {last.isoformat()}")
    print(f"  first open={rows[0]['open']:,.2f}  last close={rows[-1]['close']:,.2f}")
    if len(rows) != expected:
        print(
            f"  WARNING: expected {expected} bars for the requested window, got "
            f"{len(rows)} - there is a gap. Do not run an acceptance test on a "
            "series with holes in it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
