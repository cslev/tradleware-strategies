# Changelog

All notable changes to this project. Newest entries at the top.

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
