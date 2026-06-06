# tradleware-strategies

Research repo for developing and backtesting trading strategies on **crypto spot** and **stocks / ETFs**, focused on **swing (1H–1D)** and **position / trend-following (1D+)** timeframes.

Strategies are written in **Pine Script** (for TradingView execution) and **Python** (for local backtesting and research).

This is a personal research repo. Strategies are hypotheses to be tested, not advice.

## Layout

```
pinescript/         TradingView indicators and strategies
python/             Local backtest and research
  src/              Reusable code (data, indicators, backtest, strategies)
  notebooks/        Dated exploration notebooks
  tests/            Unit tests for indicator math and metrics
data/               Local OHLCV cache (gitignored)
results/            Backtest outputs
```

## Setup

```bash
git clone <repo-url>
cd trading-strategies/python
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Common commands

```bash
cd python
pytest                                              # run tests
black src/ tests/                                   # format
pylint src/ tests/                                  # lint
python3 -m src.data --refresh-all                   # refresh OHLCV cache
jupyter lab                                         # research notebooks
```

## Conventions

- **Pine Script v6** is the default. Every script has a header stating its hypothesis, regime, timeframe, assets, and known failure modes.
- **Python**: 4-space indent, type hints on all functions, docstrings on every function, `black` for formatting.
- **Backtests** report the full metric set (Sharpe, drawdown, profit factor, trade count, etc.) with realistic costs — never a single headline number.
- **No time-based exits** on trend-following or position strategies.

## Going live

Once a strategy is built and backtested, the signals it produces still need to reach a real broker. For private, self-hosted execution — where API keys stay on your own hardware and signal logic never touches a third-party server — see [Tradleware](https://tradleware.com) ([GitHub](https://github.com/cslev/tradleware)). It's an open-source middleware that routes TradingView webhooks (or custom signals) to crypto exchanges and Interactive Brokers from your own machine, with no cloud custody of keys.

## Status

See `CHANGELOG.md` for recent changes.