# Data cache

Local OHLCV cache. Gitignored — regenerate from sources as needed.

## Layout

```
data/
├── crypto/                          # ccxt-sourced, one folder per exchange
│   ├── binance/
│   │   └── BTC_USDT/
│   │       ├── 1d.parquet           ← source of truth
│   │       └── 1d.csv              ← only present if --csv was passed
│   ├── coinbaseadvanced/
│   │   └── DOGE_USDC/
│   │       └── 1d.parquet
│   └── okx/
│       └── ETH_USDT/
│           └── 4h.parquet
└── equities/                        # yfinance-sourced
    ├── SPY/
    │   └── 1d.parquet
    └── PLTR/
        └── 1d.parquet
```

Symbol `/` is replaced with `_` in directory names (`BTC/USDT` → `BTC_USDT`).

## Conventions

- **Parquet is the source of truth.** CSV files are exports — written only when `--csv` is passed, overwritten on every run, never read back by the fetcher.
- **Columns**: `open`, `high`, `low`, `close`, `volume` (all float64). Index is `DatetimeIndex` in UTC.
- **Incremental updates**: the fetcher reads the existing parquet, fetches only candles newer than the last row, and appends them. Don't edit parquet files manually — use `--refresh` to rebuild from scratch.

## Regeneration

```bash
# Fetch / update a crypto pair
cd python && python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01

# Fetch / update an equity
cd python && python -m src.data --equity PLTR --timeframe 1d --since 2022-01-01

# Force full re-download (discard cache)
cd python && python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --refresh
```

See `python/src/README.md` for the full CLI reference.

## Why parquet over CSV

- ~5–10× smaller on disk
- ~10× faster to load with pandas
- Preserves types — CSV reads everything back as strings until you re-cast
- Native pandas support: `df.to_parquet(path)` / `pd.read_parquet(path)`

## Inspecting a file

```python
import pandas as pd
df = pd.read_parquet("data/crypto/binance/BTC_USDT/1d.parquet")
df.tail()
```
