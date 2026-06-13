# tradleware-strategies

Research repo for developing and backtesting trading strategies on **crypto spot** and **stocks / ETFs**, focused on **swing (1H–1D)** and **position / trend-following (1D+)** timeframes.

Strategies are written in **Pine Script** (for TradingView execution) and **Python** (for local backtesting and research).

This is a personal research repo. Strategies are hypotheses to be tested, not advice.

## Layout

```
pinescript/                   TradingView indicators and strategies (Pine Script v6)
  _template.pine              Skeleton — copy this when starting a new strategy

python/                       Local research and backtesting
  src/
    data.py                   OHLCV fetcher and parquet cache (crypto + equities)
    indicators.py             Custom indicator math not covered by pandas-ta
    backtest.py               Metrics, walk-forward runner, parameter sweeps
    strategies/               One .py per strategy
  notebooks/                  Dated research notebooks (YYYY-MM-DD-topic.ipynb)
  tests/                      Unit tests for indicator math and metric correctness

data/                         Local OHLCV cache — gitignored, regenerate as needed
results/                      Backtest outputs, organized per strategy
```

See [python/README.md](python/README.md) for the Python tooling guide and [data/README.md](data/README.md) for the cache layout.

## Setup

```bash
git clone <repo-url>
cd trading-strategies/python
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## How to start

Before running any strategy or backtest, you need local candlestick data. All strategies read from the parquet cache under `data/` — nothing works without it. Fetch a symbol with:

```bash
cd python
source .venv/bin/activate
python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01
```

See [data/README.md](data/README.md) for the full cache layout, CLI reference, and how incremental updates and backfills work.

## Conventions

- **Pine Script v6** is the default. Every script has a header stating its hypothesis, regime, timeframe, assets, and known failure modes.
- **Python**: 4-space indent, type hints on all functions, docstrings on every function, `black` for formatting.
- **Backtests** report the full metric set (Sharpe, drawdown, profit factor, trade count, etc.) with realistic costs — never a single headline number.
- **No time-based exits** on trend-following or position strategies.

## Going live

Once a strategy is built and backtested, the signals it produces still need to reach a real broker. For private, self-hosted execution — where API keys stay on your own hardware and signal logic never touches a third-party server — see [Tradleware](https://tradleware.com) ([GitHub](https://github.com/cslev/tradleware)). It's an open-source middleware that routes TradingView webhooks (or custom signals) to crypto exchanges and Interactive Brokers from your own machine, with no cloud custody of keys.

## Status

See `CHANGELOG.md` for recent changes.