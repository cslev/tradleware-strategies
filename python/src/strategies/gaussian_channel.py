"""
Strategy: Gaussian Channel + Stochastic RSI
Pine Script mirror: pinescript/strategies/gaussian_channel.pine

Hypothesis:
    Price breaking above the Gaussian upper band in an uptrending channel signals
    a momentum continuation. Stochastic RSI filters for timing — either overbought
    (momentum entry) or oversold (dip within the trend). Exit when price drops back
    below the upper band or the channel reverses direction.

Assumed regime: trending uptrend. Underperforms in choppy/ranging markets.
Timeframe: 1D (designed and validated on daily bars).
Assets: crypto spot (BTC, ETH); adaptable to large-cap equities.

Known failure modes:
    - Whipsaws during sideways consolidation
    - Upper-band breakouts that immediately reverse
    - Does not short, so drawdowns during bear markets can be severe
    - High-period filter (144 bars) means the strategy is slow to respond to
      regime changes — it works best on assets with sustained trends

Usage:
    # From the repo root:
    cd python
    python -m src.strategies.gaussian_channel --symbol BTC/USDT --since 2020-01-01

    # Or as a library:
    from src.strategies.gaussian_channel import signals
    entries, exits, stops = signals(df)
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Optional

import pandas as pd

from src import backtest as bt
from src.data import fetch_crypto, fetch_equity
from src.indicators import gaussian_channel, stoch_rsi

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Tradleware-Gaussian Channel + StochRSI"


def _warmup_since(since: str, timeframe: str, warmup_bars: int) -> str:
    """
    Return a start date that is `warmup_bars` bars before `since`.

    Used to fetch pre-backtest data so that the Gaussian filter is fully
    converged by the time the strategy starts trading. Without warmup, the
    recursive IIR filter initialises from zero and produces incorrect channel
    values for roughly the first `period` bars — shifting signals and
    compounding errors through every subsequent position size.
    """
    since_dt = pd.Timestamp(since)
    m = re.match(r"^(\d+)([mhdwMW])", timeframe.lower())
    if not m:
        return since
    n, unit = int(m.group(1)), m.group(2)
    if unit == "m":
        delta = pd.Timedelta(minutes=n * warmup_bars)
    elif unit == "h":
        delta = pd.Timedelta(hours=n * warmup_bars)
    elif unit == "d":
        delta = pd.Timedelta(days=n * warmup_bars)
    elif unit == "w":
        delta = pd.Timedelta(weeks=n * warmup_bars)
    else:
        delta = pd.Timedelta(days=warmup_bars)
    return (since_dt - delta).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------


def signals(
    df: pd.DataFrame,
    poles: int = 4,
    period: int = 144,
    mult: float = 1.414,
    rsi_length: int = 14,
    stoch_length: int = 14,
    stoch_k: int = 3,
    stoch_d: int = 3,
    upper_threshold: float = 80.0,
    lower_threshold: float = 15.0,
    use_stop_loss: bool = True,
    mode_lag: bool = False,
    mode_fast: bool = False,
    src: str = "hlc3",
) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Compute entry, exit, and stop-loss signals from OHLCV data.

    This is the Python translation of the Gaussian Channel + Stochastic RSI
    Pine Script strategy. Signal logic is asset-agnostic — the same function
    works on BTC/USDT daily, SPY daily, or any other instrument.

    Fill semantics (matching the Pine Script):
        - Entries/exits fire at bar close, fill at the NEXT bar's open.
        - Stop-loss fires intrabar when the bar's low touches the lower band.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Must be sorted ascending with a DatetimeIndex.
    poles : int, default 4
        Gaussian filter poles (1–9). More poles → smoother channel, more lag.
    period : int, default 144
        Sampling period for the Gaussian filter. Longer → slower channel.
    mult : float, default 1.414
        Band width multiplier (applied to the filtered true range).
    rsi_length : int, default 14
        RSI lookback period used inside Stochastic RSI.
    stoch_length : int, default 14
        Stochastic lookback applied to the RSI series.
    stoch_k : int, default 3
        %K smoothing (SMA length).
    stoch_d : int, default 3
        %D smoothing (SMA of %K).
    upper_threshold : float, default 80.0
        Stoch RSI %K level above which we consider momentum overbought.
        An entry fires when %K > upper_threshold (breakout strength) OR
        %K < lower_threshold (oversold dip within the trend).
    lower_threshold : float, default 15.0
        Stoch RSI %K level below which we consider momentum oversold.
    use_stop_loss : bool, default True
        When True, returns the lower Gaussian band as a dynamic stop-loss.
        The stop fires if the bar's low touches or crosses below the lower band.
    mode_lag : bool, default False
        Gaussian filter lag-reduction mode.
    mode_fast : bool, default False
        Gaussian filter fast-response mode (averages 1-pole and N-pole outputs).
    src : str, default "hlc3"
        Price series fed into the Gaussian filter.
        Options: "hlc3", "close", "hl2", "ohlc4".

    Returns
    -------
    entries : pd.Series (bool)
        True on the bar where the entry signal fires.
    exits : pd.Series (bool)
        True on the bar where the exit signal fires.
    stops : pd.Series or None
        Dynamic stop-loss price at each bar (the lower Gaussian band).
        None if use_stop_loss=False.
    """
    gc = gaussian_channel(df, poles=poles, period=period, mult=mult,
                          mode_lag=mode_lag, mode_fast=mode_fast, src=src)

    k, _d = stoch_rsi(df["close"], rsi_length=rsi_length, stoch_length=stoch_length,
                      k_smooth=stoch_k, d_smooth=stoch_d)

    filt  = gc["filt"]
    hband = gc["hband"]
    lband = gc["lband"]

    # Channel is "green" (trending up) when the filter is rising
    channel_green = filt > filt.shift(1)

    # --- Entry condition (mirrors Pine Script longCondition) ---
    # Channel trending up AND close broke above upper band AND
    # Stoch RSI is overbought (momentum) OR oversold (dip in uptrend)
    stoch_condition = (k > upper_threshold) | (k < lower_threshold)
    entries = channel_green & (df["close"] > hband) & stoch_condition

    # --- Exit condition (mirrors Pine Script closeCondition) ---
    # Close drops back below upper band OR channel reverses (strictly: filt < filt[1])
    # Pine Script uses `filt < filt[1]` (strictly less than), not `filt != filt[1]`.
    exits = (df["close"] < hband) | (filt < filt.shift(1))

    # Only exit when we actually have a position to close — the backtest engine
    # ignores exit signals when flat, but this keeps the signal semantically clean.
    # (strategy.close in Pine Script is also a no-op when flat)

    stops: Optional[pd.Series] = lband if use_stop_loss else None

    return entries, exits, stops


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _results_path(symbol: str, timeframe: str, since: str) -> str:
    """Return the auto-generated results file path for this run."""
    import datetime, os
    today = datetime.date.today().isoformat()
    slug = symbol.replace("/", "-")
    fname = f"{today}_{slug}_{timeframe}_since-{since}.md"
    # __file__ is python/src/strategies/gaussian_channel.py → go up 4 levels to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(repo_root, "results", "gaussian-channel", fname)


def _run_strategy(
    df_full: pd.DataFrame,
    symbol: str,
    timeframe: str,
    trade_since: str,
    warmup_start: Optional[str] = None,
    poles: int = 4,
    period: int = 144,
    mult: float = 1.414,
    rsi_length: int = 14,
    stoch_length: int = 14,
    stoch_k: int = 3,
    stoch_d: int = 3,
    upper_threshold: float = 80.0,
    lower_threshold: float = 15.0,
    use_stop_loss: bool = True,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_ticks: int = 3,
    show_plot: bool = True,
) -> bt.BacktestResult:
    """
    Compute signals on the full (warmup + trading) DataFrame, then slice to
    the trading period before running the backtest.

    The Gaussian Channel is a recursive IIR filter that needs ~period bars to
    converge from its zero initial state. `df_full` should start well before
    `trade_since` so that signals during the actual trading period are based
    on a fully warmed-up filter. `trade_since` marks where the backtest clock
    starts — capital is deployed only from that date onward.
    """
    logger.info("computing signals for %s %s (warmup from %s, trading from %s)",
                symbol, timeframe, df_full.index[0].date(), trade_since)

    # If caller pinned an exact warmup start (e.g. to match TradingView's chart
    # history), trim df_full to that date before computing the filter so both
    # see the same initial conditions.
    if warmup_start is not None:
        ws = pd.Timestamp(warmup_start, tz=df_full.index.tz)
        df_full = df_full[df_full.index >= ws]
        logger.info("indicator warmup pinned to %s (%d bars before trading start)",
                    warmup_start, (df_full.index < pd.Timestamp(trade_since, tz=df_full.index.tz)).sum())

    entries_full, exits_full, stops_full = signals(
        df_full,
        poles=poles, period=period, mult=mult,
        rsi_length=rsi_length, stoch_length=stoch_length,
        stoch_k=stoch_k, stoch_d=stoch_d,
        upper_threshold=upper_threshold, lower_threshold=lower_threshold,
        use_stop_loss=use_stop_loss,
    )

    # Slice all series to the actual trading window.
    trade_start = pd.Timestamp(trade_since, tz=df_full.index.tz)
    mask = df_full.index >= trade_start
    df = df_full[mask]
    entries = entries_full[mask]
    exits = exits_full[mask]
    stops = stops_full[mask] if stops_full is not None else None

    logger.info("running backtest on %d bars (%s → %s)",
                len(df), df.index[0].date(), df.index[-1].date())
    result = bt.run(
        df, entries, exits, stops,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_ticks=slippage_ticks,
    )
    bt.compute_metrics(result)

    title = f"{STRATEGY_NAME}  |  {symbol} {timeframe}"
    bnh = bt.buy_and_hold(df, initial_capital=initial_capital, commission_pct=commission_pct)
    dca_result = bt.dca(df, initial_capital=initial_capital, commission_pct=commission_pct)
    gc_df = gaussian_channel(df_full, poles=poles, period=period, mult=mult)[mask]

    bt.print_metrics(result, title=title, bnh=bnh)
    bt.print_trades(result)

    out_path = _results_path(symbol, timeframe, trade_since)
    chart_path = out_path.replace(".md", ".png")

    bt.plot(result, gc_df=gc_df, title=title, compare=[bnh, dca_result],
            save_path=chart_path, show=show_plot)
    saved = bt.save_run(result, out_path, title=title, chart_path=chart_path, bnh=bnh)
    logger.info("results saved → %s", saved)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the Gaussian Channel + Stochastic RSI strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Crypto via ccxt (Binance by default)
  python -m src.strategies.gaussian_channel --symbol BTC/USDT --since 2020-01-01

  # Equity via yfinance
  python -m src.strategies.gaussian_channel --symbol SPY --equity --since 2015-01-01

  # Custom parameters
  python -m src.strategies.gaussian_channel --symbol ETH/USDT --since 2021-01-01 \\
      --poles 4 --period 144 --mult 1.414 --no-stop-loss
        """,
    )

    # --- Asset ---
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC/USDT or SPY.")
    parser.add_argument("--equity", action="store_true",
                        help="Fetch as equity via yfinance. Default: crypto via ccxt.")
    parser.add_argument("--exchange", default="binance",
                        help="ccxt exchange id (crypto only). Default: binance.")
    parser.add_argument("--timeframe", default="1d",
                        help="Bar timeframe. Default: 1d.")
    parser.add_argument("--since", default="2020-01-01", metavar="YYYY-MM-DD",
                        help="History start date. Default: 2020-01-01.")
    parser.add_argument("--until", default=None, metavar="YYYY-MM-DD",
                        help="History end date. Default: today.")

    # --- Strategy parameters ---
    parser.add_argument("--poles", type=int, default=4, help="Gaussian filter poles (1–9). Default: 4.")
    parser.add_argument("--period", type=int, default=144, help="Gaussian filter period. Default: 144.")
    parser.add_argument("--mult", type=float, default=1.414, help="Band width multiplier. Default: 1.414.")
    parser.add_argument("--rsi-length", type=int, default=14, help="RSI length. Default: 14.")
    parser.add_argument("--stoch-length", type=int, default=14, help="Stochastic length. Default: 14.")
    parser.add_argument("--stoch-k", type=int, default=3, help="Stoch %%K smoothing. Default: 3.")
    parser.add_argument("--stoch-d", type=int, default=3, help="Stoch %%D smoothing. Default: 3.")
    parser.add_argument("--upper-threshold", type=float, default=80.0,
                        help="Stoch RSI overbought threshold. Default: 80.")
    parser.add_argument("--lower-threshold", type=float, default=15.0,
                        help="Stoch RSI oversold threshold. Default: 15.")
    parser.add_argument("--no-stop-loss", action="store_true",
                        help="Disable the lower-band stop-loss.")

    # --- Backtest parameters ---
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Starting capital. Default: 10000.")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission per side as a fraction. Default: 0.001 (0.1%%).")
    parser.add_argument("--slippage", type=int, default=3,
                        help="Slippage in ticks. Default: 3.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the chart output.")
    parser.add_argument("--warmup-bars", type=int, default=None,
                        help="Extra bars to fetch before --since for filter warmup. "
                             "Default: 3 × period (e.g. 432 bars for period=144 on 1d). "
                             "Set to 0 to disable.")
    parser.add_argument("--warmup-start", default=None, metavar="YYYY-MM-DD",
                        help="Pin the exact date where indicator computation begins. "
                             "Use this to match an external tool's chart history start "
                             "(e.g. --warmup-start 2017-10-01 to match TradingView's "
                             "BINANCE:BTCUSDT chart). Overrides --warmup-bars.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Warmup: fetch extra history before --since so the Gaussian filter
    # converges before the strategy starts trading. Without this, the recursive
    # IIR filter starts from zero and produces incorrect channel values for
    # roughly the first `period` bars, shifting all subsequent signals.
    warmup_bars = args.warmup_bars if args.warmup_bars is not None else 3 * args.period
    data_since = _warmup_since(args.since, args.timeframe, warmup_bars) if warmup_bars > 0 else args.since

    # Fetch data (includes warmup period)
    if args.equity:
        df_full = fetch_equity(args.symbol, interval=args.timeframe,
                               start=data_since, end=args.until)
    else:
        df_full = fetch_crypto(args.symbol, timeframe=args.timeframe,
                               since=data_since, until=args.until,
                               exchange_id=args.exchange)

    # Slice to --until if specified. fetch_crypto returns the full cache
    # regardless of until (which only controls what gets downloaded).
    if args.until:
        until_dt = pd.Timestamp(args.until, tz=df_full.index.tz)
        df_full = df_full[df_full.index <= until_dt]

    _run_strategy(
        df_full=df_full,
        symbol=args.symbol,
        timeframe=args.timeframe,
        trade_since=args.since,
        warmup_start=args.warmup_start,
        poles=args.poles,
        period=args.period,
        mult=args.mult,
        rsi_length=args.rsi_length,
        stoch_length=args.stoch_length,
        stoch_k=args.stoch_k,
        stoch_d=args.stoch_d,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
        use_stop_loss=not args.no_stop_loss,
        initial_capital=args.capital,
        commission_pct=args.commission,
        slippage_ticks=args.slippage,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    _cli()
