# Changelog

All notable changes to this project. Newest entries at the top.

## 2026-06-20

- add backtest engine: bar-by-bar runner, closed-trade metrics, B&H and DCA comparison, candlestick plot with PNG auto-save, `print_trades`, `save_run`
- fix max drawdown to use closed-trade equity curve, matching TradingView Key Stats methodology
- plot: replace tick-mark bars with filled candlestick bodies; chart auto-saved as PNG alongside .md results
- fix Gaussian Channel alpha coefficient (was using 1.414 constant; now uses `math.sqrt(2)`)
- always write CSV alongside parquet on every data fetch (no flag required)
- add EMA Cross strategy: Python backtest + Pine Script mirror (`ema-cross.pine`)
- add Gaussian Channel strategy: Python backtest with warmup-then-slice pattern, StochRSI entry filter
- update `gaussian_channel.pine`: add strategy header, realistic cost defaults, `process_orders_on_close=false`
- replace `input.time()` date fields with `input.int()` year/month/day — reliably triggers TV recalculation
- document TV backtest discrepancies: margin-call micro-trade artifact, EMA initialization divergence, mark-to-market vs closed-trade drawdown
- add asset inception date table and safe `--since` values per asset to backtesting-playbook

## 2026-06-13

- restructure Python docs: move `python/src/README.md` → `python/README.md`, covering all planned modules
- add placeholder stubs: `indicators.py`, `backtest.py`, `strategies/`, `tests/`
- expand root `README.md` with full project layout and links to sub-READMEs
- fix gitignore patterns to cover data subdirectories (`data/**/*` instead of `data/*`)
- set binance as default exchange in `data.py`; make CSV output on by default (`--no-csv` to suppress)
- fix `data.py` incremental update bug — up-to-date cache triggered spurious 50s retry loop and misleading error
- add `python/src/data.py` — OHLCV fetcher and parquet cache for crypto (ccxt) and equities (yfinance)
- incremental updates, backfill, `--until` cutoff, progress bar, rate-limit retry logic
- fix `python/pyproject.toml` — remove invalid `readme` path that broke `pip install -e .`

## Unreleased

- Initial project scaffolding.
