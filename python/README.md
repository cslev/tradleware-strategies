# python — local research and backtesting

All Python tooling for data fetching, indicator math, backtesting, and strategy research.

## Layout

```
python/
  src/
    data.py          ← OHLCV fetcher and parquet cache (crypto + equities)
    indicators.py    ← custom indicator math not covered by pandas-ta
    backtest.py      ← metrics, walk-forward runner, parameter sweeps
    strategies/      ← one .py per strategy
  notebooks/         ← dated research notebooks (YYYY-MM-DD-topic.ipynb)
  tests/             ← unit tests for indicator math and metric correctness
```

## Setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## How the modules connect

```
data.py          →  fetches and caches OHLCV parquet files
indicators.py    →  custom indicator functions that take/return DataFrames
backtest.py      →  wraps vectorbt / backtrader; produces the standard metric set
strategies/      →  each strategy imports from the above three and runs a backtest
notebooks/       →  exploration; logic gets promoted to src/ once reused
```

---

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
python -m src.data --equity PLTR --timeframe 1d --since 2022-01-01 --verbose

# Fetch ETH 4h candles
python -m src.data --crypto ETH/USDT --timeframe 4h --since 2022-01-01 --verbose
```

### Cache layout

```
data/
  crypto/
    <exchange>/
      <SYMBOL>/
        <timeframe>.parquet     ← source of truth
        <timeframe>.csv         ← written by default alongside parquet
  equities/
    <SYMBOL>/
      <interval>.parquet
      <interval>.csv
```

**The parquet file is always the source of truth.** The CSV is just an export — it gets
overwritten from the parquet on every run. Editing the CSV has no effect.

See [../data/README.md](../data/README.md) for full cache layout and conventions.

### How incremental updates work

On the first run, the full history is downloaded from `--since` to today and saved.
On every subsequent run, the code reads the existing parquet, finds the last cached
timestamp, and fetches only the candles that came after it. Old data is never
re-downloaded.

### Backfill — extending history further back

If you pass a `--since` date earlier than the first row in the cache, the gap is
automatically backfilled and prepended. If `--since` is already covered by the cache,
it is silently ignored.

### Testing with a date cutoff (--until)

```bash
# Build a cache ending at 2024
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --until 2024-12-31

# Run again without --until — only 2025 candles are fetched
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01
```

### Force a full re-download (--refresh)

```bash
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --refresh
```

### Switching exchanges

```bash
python -m src.data --crypto BTC/USDT  --exchange binance           --since 2020-01-01
python -m src.data --crypto BTC/USDC  --exchange coinbaseadvanced  --since 2021-01-01
python -m src.data --crypto BTC/USDT  --exchange bybit             --since 2020-01-01
```

**Coinbase note**: use `coinbaseadvanced`, not `coinbase`. The `coinbase` id has very
limited historical data.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--crypto SYMBOL [...]` | — | One or more crypto symbols, e.g. `BTC/USDT ETH/USDT` |
| `--equity SYMBOL [...]` | — | One or more equity tickers, e.g. `SPY QQQ AAPL` |
| `--timeframe TF` | `1d` | Candle interval. Crypto: `1m 5m 15m 1h 4h 1d 1w`. Equities: `1m 5m 15m 1h 1d 1wk 1mo` |
| `--since YYYY-MM-DD` | `2020-01-01` | History start date |
| `--until YYYY-MM-DD` | today | Stop fetching at this date |
| `--exchange ID` | `binance` | ccxt exchange id for crypto |
| `--refresh` | off | Discard cache, re-fetch from scratch |
| `--no-csv` | off | Skip writing CSV alongside the parquet |
| `--verbose` / `-v` | off | Show live progress bar |
| `--update` | off | Refresh all default symbols defined in the script |

### Library usage

```python
from src.data import fetch_crypto, fetch_equity

df = fetch_crypto("BTC/USDT", timeframe="1d", since="2020-01-01")
df = fetch_equity("SPY", interval="1d", start="2015-01-01")
```

---

## indicators.py — custom indicator math

Functions that aren't in [pandas-ta](https://github.com/twopirllc/pandas-ta) or where
the pandas-ta version has a known issue. All functions take a `pd.DataFrame` or
`pd.Series` and return the same shape.

*(Not yet implemented — add functions here as needed.)*

---

## backtest.py — metrics and walk-forward runner

Wraps [vectorbt](https://vectorbt.pro/) (fast parameter sweeps) and
[backtrader](https://www.backtrader.com/) (realistic event-driven simulation) with the
standard metric set from `claude/knowledge/backtesting-playbook.md`:
total return, CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor,
avg win/loss, trade count, and exposure.

*(Not yet implemented.)*

---

## strategies/ — strategy implementations

One `.py` file per strategy. Each file:
- States the hypothesis in the module docstring.
- Imports OHLCV data via `fetch_crypto` / `fetch_equity`.
- Runs a backtest and prints the full metric set.
- Can be run directly: `python -m src.strategies.<name>`.

Mirror filenames with Pine Script counterparts: `spy_daily_trend_follow.py` ↔
`spy-daily-trend-follow.pine`.

*(No strategies yet.)*

---

## notebooks/ — research journal

Dated exploration notebooks: `YYYY-MM-DD-topic.ipynb`. These are a research journal,
not reusable code. Once logic is used in a third notebook, extract it to `src/`.

```bash
cd python && jupyter lab
```

---

## tests/ — unit tests

Narrow scope: indicator math and metric correctness only. Not strategy backtests — the
backtest itself is the strategy's test.

```bash
cd python && pytest
```
