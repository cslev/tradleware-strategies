# Data cache

Local OHLCV cache. Gitignored — regenerate from sources as needed.

## Layout

```
data/
├── crypto/                  # ccxt-sourced
│   ├── BTCUSDT_1d.parquet
│   └── ETHUSDT_4h.parquet
└── equities/                # yfinance-sourced
    ├── SPY_1d.parquet
    └── QQQ_1d.parquet
```

## Conventions

- **One file per symbol/timeframe**, containing full available history.
- **Naming**: `{SYMBOL}_{TIMEFRAME}.parquet`. Timeframe in lowercase short form (`1d`, `4h`, `1h`, `15m`).
- **Format**: parquet. Smaller and faster than CSV; preserves types (`pd.DatetimeIndex` in UTC, float columns for OHLCV).
- **Columns**: `open`, `high`, `low`, `close`, `volume`. Index is `DatetimeIndex` in UTC.
- **Incremental updates**: the fetcher reads the existing file, fetches only the missing tail, and rewrites the file. Don't append manually.

## Regeneration

```bash
# Refresh everything
cd python && python -m src.data --refresh-all

# Refresh specific symbols
cd python && python -m src.data --crypto BTCUSDT ETHUSDT --tf 1d
cd python && python -m src.data --equities SPY QQQ --tf 1d
```

## Why parquet over CSV

- ~5–10× smaller on disk
- ~10× faster to load with pandas
- Preserves types — CSV reads back everything as strings until you re-cast
- Native pandas support: `df.to_parquet(path)` / `pd.read_parquet(path)`

Requires `pyarrow` (already in `pyproject.toml`).

## Inspecting a file manually

```python
import pandas as pd
df = pd.read_parquet("data/crypto/BTCUSDT_1d.parquet")
df.tail()
```