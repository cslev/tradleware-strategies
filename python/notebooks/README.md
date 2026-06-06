# Notebooks

Dated exploration journal. These are **research artifacts, not reusable code**.

## Naming

`YYYY-MM-DD-topic.ipynb` — for example:

- `2025-05-23-spy-rsi-mean-rev.ipynb`
- `2025-05-30-btc-regime-analysis.ipynb`
- `2025-06-01-atr-stop-sensitivity.ipynb`

The date prefix turns the folder into a chronological research log.

## What goes here

- Quick exploration of a hypothesis or signal
- Plotting equity curves, drawdown, parameter heatmaps
- One-off backtests, sensitivity sweeps
- Comparing strategies, asset behaviors, regime visualizations
- Anything throwaway-but-worth-keeping for the record

## What doesn't

- **Reusable logic.** When the same code appears in a third notebook, extract it into `src/` (`indicators.py`, `backtest.py`, or a new module).
- **The strategy itself.** Strategies live in `src/strategies/`. Notebooks may import and explore them, but the canonical implementation is in `src/`.

## Don't tidy up old notebooks

They're a record of what was tried, including the ideas that didn't work. If a conclusion isn't obvious from a notebook's filename or last cell, add a brief markdown cell at the top noting the takeaway. Then leave it.

## Setup

```bash
cd python && source .venv/bin/activate
jupyter lab
```

Notebooks should `import` from `src/` (e.g. `from src.backtest import metrics`), not duplicate code.