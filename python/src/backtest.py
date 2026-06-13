"""
Backtesting utilities: metric calculation, walk-forward runner, parameter sweeps.

Not a full backtesting engine — use vectorbt (fast, parameter sweeps) or
backtrader (realistic stops, multi-leg logic) for the actual simulation.
This module wraps those tools with the standard metric set and evaluation
patterns defined in claude/knowledge/backtesting-playbook.md.

Metrics reported on every backtest:
  total return, CAGR, Sharpe, Sortino, max drawdown, Calmar,
  win rate, profit factor, avg win/loss (R), trade count, exposure.
"""
