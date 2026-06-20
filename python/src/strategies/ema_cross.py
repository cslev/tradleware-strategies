"""
Strategy: Dual EMA Crossover
Pine Script mirror: pinescript/strategies/ema-cross.pine

Hypothesis:
    When the fast EMA crosses above the slow EMA, the short-term trend has
    shifted bullish — ride it until the cross reverses.

Assumed regime: trending. Whipsaws heavily in sideways markets.
Timeframe: 1D (designed and validated on daily bars).
Assets: crypto spot or large-cap equities.

Known failure modes:
    - Choppy / sideways markets generate many losing round-trips
    - Lagging by definition: enters after the move has started, exits after it ends
    - Does not short, so equity bleeds during bear markets

Usage:
    cd python
    python -m src.strategies.ema_cross --symbol BTC/USDT --since 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta

from src import backtest as bt
from src.data import fetch_crypto, fetch_equity

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Tradleware-EMA Cross"


def signals(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Compute entry and exit signals for a dual EMA crossover strategy.

    Entry: fast EMA crosses above slow EMA (crossover event, not state).
    Exit:  fast EMA crosses below slow EMA (crossunder event).
    Stop:  none — exits are signal-only.

    Fill semantics (matching Pine Script process_orders_on_close=false):
        Signal fires at bar close, fills at the NEXT bar's open.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex, sorted ascending.
    fast : int, default 20
        Fast EMA period.
    slow : int, default 50
        Slow EMA period.

    Returns
    -------
    entries : pd.Series[bool]
    exits   : pd.Series[bool]
    stops   : None  (no stop-loss in this strategy)
    """
    fast_ema = ta.ema(df["close"], length=fast)
    slow_ema = ta.ema(df["close"], length=slow)

    # Crossover: fast was <= slow, now fast > slow
    entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))

    # Crossunder: fast was >= slow, now fast < slow
    exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

    return entries, exits, None


def _results_path(symbol: str, timeframe: str, since: str) -> str:
    """Return the auto-generated results file path for this run."""
    import datetime, os
    today = datetime.date.today().isoformat()
    slug = symbol.replace("/", "-")
    fname = f"{today}_{slug}_{timeframe}_since-{since}.md"
    # __file__ is python/src/strategies/ema_cross.py → go up 4 levels to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(repo_root, "results", "ema-cross", fname)


def _run_strategy(
    df_full: pd.DataFrame,
    symbol: str,
    timeframe: str,
    trade_since: str,
    fast: int = 20,
    slow: int = 50,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_ticks: int = 3,
    show_plot: bool = True,
) -> bt.BacktestResult:
    # Compute signals on the full dataset so EMAs are converged by trade_since.
    # Slice to the trading window before running the backtest.
    entries_full, exits_full, _ = signals(df_full, fast=fast, slow=slow)

    trade_start = pd.Timestamp(trade_since, tz=df_full.index.tz)
    mask = df_full.index >= trade_start
    df = df_full[mask]
    entries = entries_full[mask]
    exits = exits_full[mask]

    logger.info("running backtest on %d bars (%s → %s)",
                len(df), df.index[0].date(), df.index[-1].date())

    result = bt.run(
        df, entries, exits, None,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_ticks=slippage_ticks,
    )
    bt.compute_metrics(result)

    title = f"{STRATEGY_NAME}  |  {symbol} {timeframe}  |  EMA {fast}/{slow}"
    bnh = bt.buy_and_hold(df, initial_capital=initial_capital, commission_pct=commission_pct)

    bt.print_metrics(result, title=title, bnh=bnh)
    bt.print_trades(result)

    out_path = _results_path(symbol, timeframe, trade_since)
    chart_path = out_path.replace(".md", ".png")

    bt.plot(result, title=title, compare=[bnh], save_path=chart_path, show=show_plot)
    saved = bt.save_run(result, out_path, title=title, chart_path=chart_path, bnh=bnh)
    logger.info("results saved → %s", saved)
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the Dual EMA Crossover strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m src.strategies.ema_cross --symbol BTC/USDT --since 2020-01-01
  python -m src.strategies.ema_cross --symbol SPY --equity --since 2015-01-01 --fast 50 --slow 200
        """,
    )

    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC/USDT or SPY.")
    parser.add_argument("--equity", action="store_true",
                        help="Fetch as equity via yfinance. Default: crypto via ccxt.")
    parser.add_argument("--exchange", default="binance",
                        help="ccxt exchange id (crypto only). Default: binance.")
    parser.add_argument("--timeframe", default="1d", help="Bar timeframe. Default: 1d.")
    parser.add_argument("--since", default="2020-01-01", metavar="YYYY-MM-DD",
                        help="History start date.")
    parser.add_argument("--until", default=None, metavar="YYYY-MM-DD",
                        help="History end date. Default: today.")

    parser.add_argument("--fast", type=int, default=20, help="Fast EMA period. Default: 20.")
    parser.add_argument("--slow", type=int, default=50, help="Slow EMA period. Default: 50.")

    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital. Default: 10000.")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission per side as a fraction. Default: 0.001 (0.1%%).")
    parser.add_argument("--slippage", type=int, default=3, help="Slippage in ticks. Default: 3.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the chart output.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.equity:
        df_full = fetch_equity(args.symbol, interval=args.timeframe,
                               start=args.since, end=args.until)
    else:
        df_full = fetch_crypto(args.symbol, timeframe=args.timeframe,
                               since=args.since, until=args.until,
                               exchange_id=args.exchange)

    if args.until:
        until_dt = pd.Timestamp(args.until, tz=df_full.index.tz)
        df_full = df_full[df_full.index <= until_dt]

    _run_strategy(
        df_full=df_full,
        symbol=args.symbol,
        timeframe=args.timeframe,
        trade_since=args.since,
        fast=args.fast,
        slow=args.slow,
        initial_capital=args.capital,
        commission_pct=args.commission,
        slippage_ticks=args.slippage,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    _cli()
