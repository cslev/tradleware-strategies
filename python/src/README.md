# src — Python source modules

## data.py — OHLCV fetcher and cache

Fetches candlestick (OHLCV) data for crypto pairs and equities, stores it locally as
[Parquet](https://parquet.apache.org/) files, and updates incrementally so you never
re-download history you already have.

### Quick start

```bash
cd python
source .venv/bin/activate

# Fetch Bitcoin daily candles from Binance since 2020
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01

# Fetch SPY daily from yfinance
python -m src.data --equity SPY --timeframe 1d --since 2015-01-01

# Fetch PLTR daily with a progress bar and a CSV alongside the parquet
python -m src.data --equity PLTR --timeframe 1d --since 2022-01-01 --verbose --csv

# Show a progress bar and also save a CSV you can open in a spreadsheet
python -m src.data --crypto ETH/USDT --timeframe 4h --since 2022-01-01 --verbose --csv
```

### Where data is stored

```
data/
  crypto/
    <exchange>/
      <SYMBOL>/
        <timeframe>.parquet     ← source of truth
        <timeframe>.csv         ← only written when --csv is passed
  equities/
    <SYMBOL>/
      <interval>.parquet
      <interval>.csv
```

**The parquet file is always the source of truth.** The CSV is just an export — it gets
overwritten from the parquet every time you run with `--csv`. Editing the CSV has no
effect; editing the parquet is what matters.

### How incremental updates work

On the first run, the full history is downloaded from `--since` to today and saved.
On every subsequent run, the code reads the existing parquet, finds the last cached
timestamp, and fetches only the candles that came after it. Old data is never
re-downloaded.

Example: you fetch BTC/USDT daily up to today. Tomorrow you run the same command
again — only today's candle (or the last few if the market was closed) is fetched
and appended.

### Backfill — extending history further back

If you pass a `--since` date that is **earlier** than the first row already in the
cache, the gap is automatically backfilled (fetched and prepended). If `--since` is
equal to or later than the first cached row, it is silently ignored — the cache
already covers that date.

Example:
```bash
# First run: cache starts at 2022-01-01
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2022-01-01

# Second run: --since is earlier, so the gap 2020-01-01 → 2021-12-31 is fetched
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01
```

### Testing with a date cutoff (--until)

Use `--until` to fetch data only up to a specific date. Useful for testing incremental
updates without waiting for the current day.

```bash
# Build a cache that ends at end of 2024
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --until 2024-12-31

# Now run without --until — only the 2025 candles are downloaded
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01
```

### Force a full re-download (--refresh)

`--refresh` discards the cache entirely and re-fetches everything from `--since`.
Use this if the parquet file is corrupt, or if you suspect a gap inside the history
(incremental updates can only fill gaps at the edges, not in the middle).

```bash
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --refresh
```

### Switching exchanges

Crypto data is fetched via [ccxt](https://github.com/ccxt/ccxt). The default exchange
is OKX. Pass `--exchange <id>` to use any ccxt-supported exchange.

```bash
python -m src.data --crypto BTC/USDT  --exchange binance           --since 2020-01-01
python -m src.data --crypto BTC/USDC  --exchange coinbaseadvanced  --since 2021-01-01
python -m src.data --crypto BTC/USDT  --exchange bybit             --since 2020-01-01
```

Exchange-specific notes:
- **Coinbase**: use `coinbaseadvanced`, not `coinbase`. The `coinbase` id points to a
  newer API with very limited historical data.
- Symbol names are exchange-specific. Binance uses `BTC/USDT`; Coinbase uses `BTC/USDC`
  or `BTC/USD`. If a symbol is not found, the error message lists similar ones.

### Rate limiting

Exchanges enforce request limits (usually a few hundred requests per minute). The code
has `enableRateLimit=True` which makes ccxt automatically sleep between requests so
you don't hit the limit under normal use.

If you do get rate-limited (the exchange temporarily blocks your IP), the code will:
1. Retry the failed request up to 3 times with increasing wait times (5s, 15s, 30s).
2. If still blocked after retries, print an error suggesting you wait and try again.

Rate limiting is more likely when fetching very long histories with short timeframes
(e.g. 1-minute data going back years) because that requires hundreds of API calls in
quick succession.

### Available CLI flags

| Flag | Default | Description |
|---|---|---|
| `--crypto SYMBOL [...]` | — | One or more crypto symbols, e.g. `BTC/USDT ETH/USDT` |
| `--equity SYMBOL [...]` | — | One or more equity tickers, e.g. `SPY QQQ AAPL` |
| `--timeframe TF` | `1d` | Candle interval. Crypto: `1m 5m 15m 1h 4h 1d 1w`. Equities: `1m 5m 15m 1h 1d 1wk 1mo` |
| `--since YYYY-MM-DD` | `2020-01-01` | History start date |
| `--until YYYY-MM-DD` | today | Stop fetching at this date |
| `--exchange ID` | `okx` | ccxt exchange id for crypto |
| `--refresh` | off | Discard cache, re-fetch from scratch |
| `--csv` | off | Also write a CSV alongside the parquet |
| `--verbose` / `-v` | off | Show live progress bar |
| `--update` | off | Refresh all default symbols defined in the script |

### Using data.py as a library

```python
from src.data import fetch_crypto, fetch_equity

# Returns a pandas DataFrame with DatetimeIndex (UTC) and columns:
# open, high, low, close, volume
df = fetch_crypto("BTC/USDT", timeframe="1d", since="2020-01-01")
df = fetch_equity("SPY", interval="1d", start="2015-01-01")
```

Both functions use the same cache logic as the CLI — first call downloads, subsequent
calls return cached data with only new candles appended.
