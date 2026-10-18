# Working on the harness

## Requirements

Python 3.11 or newer and nothing else. The harness is standard library only;
`pytest` is the single development dependency.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

There is no build step and no install step — `scripts/` and `tests/` both put the
repository root on `sys.path` themselves, so the package runs from a bare
checkout.

## Ground rules

These are the rules the design rests on. A change that breaks one of them is a
change that makes the harness lie, so it needs a much better argument than
convenience.

1. **No public API may return data from the future.** Every data access is
   bounds-checked against the clock at access time. If you add an accessor to
   `PointInTimeView`, it checks bounds too, and it raises `LookAheadError`
   rather than `IndexError`.
2. **No option to fill at the current bar's close.** Orders queue on the bar
   that decides them and fill at the next bar's open. A flag to do otherwise
   would be a flag for producing a wrong number quietly.
3. **Costs default to pessimistic.** A forgotten venue argument picks the worst
   fee in the table, not the best. `--zero-costs` exists for checking
   arithmetic and labels its own output as not an evaluation.
4. **Uncomputable metrics return `None`, never `0.0` and never `inf`.** A
   Sharpe of `0.0` reads as "flat"; `None` reads as "not enough data".
5. **Expected values in tests are written as explicit arithmetic** on a sample
   you can work out on paper — never read back from the implementation. A test
   that echoes the code it tests proves nothing.
6. **Tests do not touch the network.** The bundled CSVs exist so the acceptance
   gate runs offline and deterministically. `scripts/fetch_ohlcv.py` is the only
   thing here that makes an HTTP request, and no test calls it.

## Regenerating the bundled data

The CSVs in `data/` are committed deliberately. To rebuild them:

```bash
python scripts/fetch_ohlcv.py --source coinbase --symbol BTC-USD \
  --start 2022-12-31 --end 2024-01-01 --out data/BTC-USD-1d-2023.csv
python scripts/fetch_ohlcv.py --source bitstamp --symbol btcusd \
  --start 2022-12-31 --end 2024-01-01 --out data/BTCUSD-1d-2023-bitstamp.csv
```

Each fetch writes a `.meta.json` sidecar recording source, symbol, bar count,
window and fetch time.

The window starts on 2022-12-31 on purpose. That bar is the warm-up: the
strategy decides on its close and fills at the open of 2023-01-01, so the
measured window is exactly the 2023 calendar year and the result is directly
comparable to published annual figures. If you regenerate with a different start
date, the acceptance test's hand calculation changes and it should fail.

Two notes on sources, learned the hard way:

- `api.binance.com` returns **HTTP 451 from US IP addresses**, so Binance is not
  usable for data here. Its *fee schedule* is still modelled, because that is a
  published number rather than an API call.
- Kraken is deliberately not offered. Its public OHLC endpoint ignores `since`
  for deep history and returns only the most recent ~720 intervals, so asking it
  for 2023 hands back recent data with no error at all. A source that silently
  returns the wrong window is worse than no source.

`.gitattributes` marks `*.csv` as binary so no platform rewrites the line
endings. The acceptance gate reads those bytes directly, so they are part of the
fixture.

## Appendix: the original development machine

Kept only because it explains the `.venv` path that appears in the git history.
Nothing here is needed to run the project.

That machine had no Python on `PATH`. The venv was built from a CPython 3.14.7
vendored under another application's toolchain:

```bash
"$LOCALAPPDATA/hermes/tools/uv-0.12.3-win32-x64/uv.exe" venv \
  --python "$LOCALAPPDATA/hermes/tools/python-3.14.7+20260901-win32-x64/python.exe" .venv
VIRTUAL_ENV="$PWD/.venv" "$LOCALAPPDATA/hermes/tools/uv-0.12.3-win32-x64/uv.exe" pip install pytest
```

Commands were then run as `./.venv/Scripts/python.exe …`. If that vendored
interpreter moves or is removed by an update of the app that owns it, the venv
breaks; delete `.venv` and rebuild against any Python 3.11+.
