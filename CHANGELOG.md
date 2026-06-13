# Changelog

All notable changes to this project. Newest entries at the top.

## 2026-06-13

- add `python/src/data.py` — OHLCV fetcher and parquet cache for crypto (ccxt) and equities (yfinance)
- incremental updates, backfill, `--until` cutoff, progress bar, CSV export, rate-limit retry logic
- add `python/src/README.md` — usage docs for data.py
- fix `python/pyproject.toml` — remove invalid `readme` path that broke `pip install -e .`

## Unreleased

- Initial project scaffolding.
