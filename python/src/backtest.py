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

from __future__ import annotations

# Standard library
import math
from dataclasses import dataclass, field

# Third-party
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """
    A single completed round-trip trade (entry + exit).

    Attributes
    ----------
    entry_bar : int
        Integer position (iloc index) of the bar where the entry signal fired.
    entry_date : pd.Timestamp
        Timestamp of the entry signal bar.
    entry_price : float
        Actual fill price after slippage (next bar's open ± slippage).
    exit_bar : int
        Integer position of the bar where the fill occurred.
    exit_date : pd.Timestamp
        Timestamp of the exit fill bar.
    exit_price : float
        Actual fill price after slippage.
    qty : float
        Number of units held.
    pnl : float
        Profit or loss in currency terms, net of both commissions.
    pnl_pct : float
        P&L as a percentage of the entry value (entry_price * qty).
    exit_reason : str
        Either "signal" (exit Series triggered) or "stop" (stop-loss hit).
    """

    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_bar: int
    exit_date: pd.Timestamp
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """
    Everything produced by a backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Mark-to-market equity at each bar's close, with DatetimeIndex.
        When flat: equity = cash. When long: equity = cash + qty * close.
    drawdown : pd.Series
        Drawdown as a fraction at each bar, always <= 0.
        Calculated as (equity - running_peak) / running_peak.
    trades : list[Trade]
        All completed trades in chronological order.
    initial_capital : float
        Starting cash.
    metrics : dict
        Populated by compute_metrics(). Empty until that function is called.
    df : pd.DataFrame
        The original OHLCV DataFrame passed to run().
    """

    equity_curve: pd.Series
    drawdown: pd.Series
    trades: list[Trade]
    initial_capital: float
    metrics: dict = field(default_factory=dict)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def run(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    stops: Optional[pd.Series] = None,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_ticks: int = 3,
    tick_size: float = 0.01,
    qty_pct: float = 100.0,
) -> BacktestResult:
    """
    Run a bar-by-bar backtest that mirrors TradingView's strategy tester
    with ``process_orders_on_close=false`` semantics.

    Fill semantics
    --------------
    - Entry / exit signals fire at bar close → fill at the *next* bar's open.
    - A stop-loss is checked intrabar on each open bar:
        * If bar.open <= stop_price  → gap down; fill at bar.open.
        * If bar.low <= stop_price < bar.open  → fill at stop_price exactly.
        * If bar.low > stop_price  → stop not reached; no fill.
    - When both a stop and an exit signal are active on the same bar the stop
      takes priority (it fires intrabar before the signal exit, which would
      fill at the *next* bar's open).
    - Only one long position at a time (no pyramiding).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex and columns: open, high, low, close, volume.
        Must be sorted ascending by time.
    entries : pd.Series
        Boolean (or 0/1) Series aligned with df.index.
        True on the bar where the entry signal fires; fill on next bar's open.
    exits : pd.Series
        Boolean (or 0/1) Series aligned with df.index.
        True on the bar where the exit signal fires; fill on next bar's open.
    stops : Optional[pd.Series]
        Stop-loss price at each bar, aligned with df.index.
        NaN means no stop is active for that bar. The stop is dynamic — the
        engine uses the stop value on the *current* bar to protect the position.
        Pass None to disable stops entirely.
    initial_capital : float
        Starting account equity in dollars (or whatever currency your price data
        uses). Default: 10,000.
    commission_pct : float
        Commission charged per side as a fraction of trade value.
        Default: 0.001 (0.1%, typical for crypto spot taker fees).
    slippage_ticks : int
        Number of ticks of slippage applied to every fill.
        Buys fill slightly above the reference price; sells slightly below.
        Default: 3.
    tick_size : float
        Dollar value of one tick. Default: 0.01 (1 cent, fine for most assets).
    qty_pct : float
        Percentage of current equity to risk on each trade. Default: 100.0
        (all-in per trade — one position at a time).

    Returns
    -------
    BacktestResult
        Contains equity_curve, drawdown series, and completed trades list.
        Call compute_metrics(result) to populate result.metrics.
    """
    # --- Validate inputs ---
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")

    n = len(df)
    slippage_dollars = slippage_ticks * tick_size

    # Normalise signals to bool arrays for speed
    entry_arr = entries.reindex(df.index).fillna(False).astype(bool).to_numpy()
    exit_arr = exits.reindex(df.index).fillna(False).astype(bool).to_numpy()

    # Stop prices — NaN where inactive
    if stops is not None:
        stop_arr = stops.reindex(df.index).to_numpy(dtype=float)
    else:
        stop_arr = np.full(n, np.nan)

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    dates = df.index

    # --- State ---
    cash: float = initial_capital
    position_qty: float = 0.0
    entry_price_fill: float = 0.0
    entry_bar_idx: int = -1
    entry_commission: float = 0.0

    equity_values = np.empty(n, dtype=float)
    completed_trades: list[Trade] = []

    # Track whether we have a pending entry signal (fires at next bar's open)
    pending_entry: bool = False
    pending_exit: bool = False

    for i in range(n):
        in_position = position_qty > 0.0

        # -----------------------------------------------------------------
        # 1. Check stop-loss intrabar (before processing pending signals).
        #    Only relevant when we hold a position.
        #
        #    TradingView semantics with process_orders_on_close=false:
        #      - strategy.exit(stop=lband) is called on bar N's close.
        #      - The stop price registered is lband[N].
        #      - That stop is checked against bar N+1's intrabar prices.
        #    So the correct stop price when processing bar i is stop_arr[i-1],
        #    not stop_arr[i]. Using stop_arr[i] is a one-bar look-ahead on the
        #    stop level.
        # -----------------------------------------------------------------
        stop_triggered = False
        if in_position and i > 0 and not np.isnan(stop_arr[i - 1]):
            stop_price = stop_arr[i - 1]
            if opens[i] <= stop_price:
                # Gap down through the stop — fill at bar open
                stop_fill = opens[i] - slippage_dollars
                stop_fill = max(stop_fill, 1e-10)  # can't fill below zero
                stop_comm = stop_fill * position_qty * commission_pct
                pnl = (stop_fill - entry_price_fill) * position_qty - entry_commission - stop_comm
                entry_value = entry_price_fill * position_qty
                pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
                cash += stop_fill * position_qty - stop_comm
                completed_trades.append(
                    Trade(
                        entry_bar=entry_bar_idx,
                        entry_date=dates[entry_bar_idx],
                        entry_price=entry_price_fill,
                        exit_bar=i,
                        exit_date=dates[i],
                        exit_price=stop_fill,
                        qty=position_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="stop",
                    )
                )
                position_qty = 0.0
                pending_exit = False  # stop already closed us
                stop_triggered = True
            elif lows[i] <= stop_price:
                # Price dipped to the stop level intrabar — fill at stop_price
                stop_fill = stop_price - slippage_dollars
                stop_fill = max(stop_fill, 1e-10)
                stop_comm = stop_fill * position_qty * commission_pct
                pnl = (stop_fill - entry_price_fill) * position_qty - entry_commission - stop_comm
                entry_value = entry_price_fill * position_qty
                pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
                cash += stop_fill * position_qty - stop_comm
                completed_trades.append(
                    Trade(
                        entry_bar=entry_bar_idx,
                        entry_date=dates[entry_bar_idx],
                        entry_price=entry_price_fill,
                        exit_bar=i,
                        exit_date=dates[i],
                        exit_price=stop_fill,
                        qty=position_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="stop",
                    )
                )
                position_qty = 0.0
                pending_exit = False
                stop_triggered = True

        in_position = position_qty > 0.0  # refresh after potential stop fill

        # -----------------------------------------------------------------
        # 2. Process pending exit from the previous bar's close signal.
        #    Fills at this bar's open (if stop didn't already close us).
        # -----------------------------------------------------------------
        if pending_exit and in_position and not stop_triggered:
            exit_fill = opens[i] - slippage_dollars
            exit_fill = max(exit_fill, 1e-10)
            exit_comm = exit_fill * position_qty * commission_pct
            pnl = (exit_fill - entry_price_fill) * position_qty - entry_commission - exit_comm
            entry_value = entry_price_fill * position_qty
            pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
            cash += exit_fill * position_qty - exit_comm
            completed_trades.append(
                Trade(
                    entry_bar=entry_bar_idx,
                    entry_date=dates[entry_bar_idx],
                    entry_price=entry_price_fill,
                    exit_bar=i,
                    exit_date=dates[i],
                    exit_price=exit_fill,
                    qty=position_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason="signal",
                )
            )
            position_qty = 0.0

        pending_exit = False  # consumed
        in_position = position_qty > 0.0

        # -----------------------------------------------------------------
        # 3. Process pending entry from the previous bar's close signal.
        #    Only enter if we are now flat (stop or exit may have just closed).
        # -----------------------------------------------------------------
        if pending_entry and not in_position:
            entry_fill = opens[i] + slippage_dollars
            equity_now = cash  # we're flat, so equity = cash at this point
            invest = equity_now * (qty_pct / 100.0)
            qty = invest / entry_fill if entry_fill > 0 else 0.0
            if qty > 0:
                entry_comm = entry_fill * qty * commission_pct
                cash -= entry_fill * qty + entry_comm
                position_qty = qty
                entry_price_fill = entry_fill
                entry_bar_idx = i
                entry_commission = entry_comm
                in_position = True

        pending_entry = False  # consumed

        # -----------------------------------------------------------------
        # 3a. Check stop on the same bar as entry.
        #     TradingView registers the stop order before the entry bar opens,
        #     so the entry bar's intrabar prices are immediately checked against
        #     lband[i-1] (the stop set at bar i-1's close). Without this check,
        #     we would miss stop-outs that occur on the very bar a trade opens.
        #     In practice this is rare for this strategy (entry requires close
        #     far above lband) but is required for correctness.
        # -----------------------------------------------------------------
        if in_position and entry_bar_idx == i and i > 0 and not np.isnan(stop_arr[i - 1]):
            stop_price = stop_arr[i - 1]
            if opens[i] <= stop_price:
                # Gapped below stop on entry bar — fill at open (same as entry open)
                stop_fill = opens[i] - slippage_dollars
                stop_fill = max(stop_fill, 1e-10)
                stop_comm = stop_fill * position_qty * commission_pct
                pnl = (stop_fill - entry_price_fill) * position_qty - entry_commission - stop_comm
                entry_value = entry_price_fill * position_qty
                pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
                cash += stop_fill * position_qty - stop_comm
                completed_trades.append(
                    Trade(
                        entry_bar=entry_bar_idx,
                        entry_date=dates[entry_bar_idx],
                        entry_price=entry_price_fill,
                        exit_bar=i,
                        exit_date=dates[i],
                        exit_price=stop_fill,
                        qty=position_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="stop",
                    )
                )
                position_qty = 0.0
                in_position = False
            elif lows[i] <= stop_price:
                # Price dipped to stop intrabar on entry bar
                stop_fill = stop_price - slippage_dollars
                stop_fill = max(stop_fill, 1e-10)
                stop_comm = stop_fill * position_qty * commission_pct
                pnl = (stop_fill - entry_price_fill) * position_qty - entry_commission - stop_comm
                entry_value = entry_price_fill * position_qty
                pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
                cash += stop_fill * position_qty - stop_comm
                completed_trades.append(
                    Trade(
                        entry_bar=entry_bar_idx,
                        entry_date=dates[entry_bar_idx],
                        entry_price=entry_price_fill,
                        exit_bar=i,
                        exit_date=dates[i],
                        exit_price=stop_fill,
                        qty=position_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="stop",
                    )
                )
                position_qty = 0.0
                in_position = False

        # -----------------------------------------------------------------
        # 4. Record signals that fired on this bar's CLOSE.
        #    They will be acted on at the NEXT bar's open.
        #
        #    TradingView semantics: if strategy.close and strategy.entry both
        #    fire on the same bar, the close executes at next bar's open and
        #    then the entry also executes at the same next bar's open.
        #    So when an exit fires while in position, we also record any
        #    simultaneous entry signal — the exit step (step 2) will flatten
        #    us first, and then the entry step (step 3) will open the new trade.
        # -----------------------------------------------------------------
        if exit_arr[i] and in_position:
            pending_exit = True
            # Also queue re-entry if the entry condition fires on this same bar.
            # Step 3 only acts on pending_entry when not in_position, which will
            # be satisfied after step 2 fills the exit.
            if entry_arr[i]:
                pending_entry = True
        if entry_arr[i] and not in_position:
            pending_entry = True

        # -----------------------------------------------------------------
        # 5. Mark-to-market equity at bar close.
        # -----------------------------------------------------------------
        equity_values[i] = cash + position_qty * closes[i]

    # If we're still in a position at the end, close at the last close price
    if position_qty > 0.0:
        last_i = n - 1
        final_fill = closes[last_i] - slippage_dollars
        final_fill = max(final_fill, 1e-10)
        final_comm = final_fill * position_qty * commission_pct
        pnl = (final_fill - entry_price_fill) * position_qty - entry_commission - final_comm
        entry_value = entry_price_fill * position_qty
        pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0
        cash += final_fill * position_qty - final_comm
        completed_trades.append(
            Trade(
                entry_bar=entry_bar_idx,
                entry_date=dates[entry_bar_idx],
                entry_price=entry_price_fill,
                exit_bar=last_i,
                exit_date=dates[last_i],
                exit_price=final_fill,
                qty=position_qty,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="signal",
            )
        )
        equity_values[last_i] = cash  # flat now

    # Build equity Series and drawdown Series
    equity_curve = pd.Series(equity_values, index=df.index, name="equity")

    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    drawdown.name = "drawdown"

    return BacktestResult(
        equity_curve=equity_curve,
        drawdown=drawdown,
        trades=completed_trades,
        initial_capital=initial_capital,
        df=df,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(result: BacktestResult) -> dict:
    """
    Compute the full standard metric set and store it in result.metrics.

    All metrics are defined in claude/knowledge/backtesting-playbook.md.
    This function mutates result.metrics in place and also returns the dict
    so you can inspect it immediately after calling.

    Annualisation uses the median bar interval inferred from the DataFrame's
    DatetimeIndex, so it works correctly for daily, hourly, 4h, and other bar
    sizes without any manual setting. For crypto (24/7 markets) this gives:
      daily  → bars_per_year ≈ 365
      hourly → bars_per_year ≈ 8766
      4h     → bars_per_year ≈ 2191

    Parameters
    ----------
    result : BacktestResult
        A result returned by run(), buy_and_hold(), or dca(). Must have
        a populated equity_curve and trades list.

    Returns
    -------
    dict
        The same dict stored in result.metrics.
    """
    ec = result.equity_curve
    initial = result.initial_capital
    trades = result.trades
    df = result.df

    # --- Annualisation factor ---
    # Infer from median bar spacing. Works for any bar size.
    if len(df.index) > 1:
        median_secs = df.index.to_series().diff().median().total_seconds()
        bars_per_year = (365.25 * 24 * 3600) / median_secs if median_secs > 0 else 252.0
    else:
        bars_per_year = 252.0

    # --- Return metrics ---
    final_equity = float(ec.iloc[-1])
    total_return_pct = (final_equity - initial) / initial * 100.0

    years = (df.index[-1] - df.index[0]).days / 365.25 if len(df.index) > 1 else 1.0
    if years > 0 and final_equity > 0:
        cagr_pct = ((final_equity / initial) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    # --- Risk-adjusted returns ---
    # Per-bar returns on the equity curve
    bar_returns = ec.pct_change().dropna()

    ann_factor = math.sqrt(bars_per_year)

    if bar_returns.std() > 0:
        sharpe = float(bar_returns.mean() / bar_returns.std() * ann_factor)
    else:
        sharpe = 0.0

    downside_returns = bar_returns[bar_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = float(bar_returns.mean() / downside_returns.std() * ann_factor)
    else:
        sortino = 0.0

    # --- Drawdown (closed-trade) ---
    # Computed on the equity after each trade closes, not mark-to-market.
    # Matches TradingView Key Stats methodology: only realised P&L counts.
    if trades:
        running = float(initial)
        closed_equity_points = [running]
        for t in trades:
            running += t.pnl
            closed_equity_points.append(running)
        closed_eq = pd.Series(closed_equity_points, dtype=float)
        closed_peak = closed_eq.cummax()
        max_drawdown_pct = float(((closed_eq - closed_peak) / closed_peak).min() * 100.0)
    else:
        max_drawdown_pct = 0.0
    calmar = (
        abs(cagr_pct / max_drawdown_pct) if max_drawdown_pct != 0 else 0.0
    )

    # --- Trade-level metrics ---
    num_trades = len(trades)

    if num_trades == 0:
        win_rate_pct = 0.0
        profit_factor = 0.0
        avg_win_pct = 0.0
        avg_loss_pct = 0.0
        r_multiple = 0.0
    else:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate_pct = len(wins) / num_trades * 100.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win_pct = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
        avg_loss_pct = float(abs(np.mean([t.pnl_pct for t in losses]))) if losses else 0.0
        r_multiple = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else float("inf")

    # --- Exposure ---
    # Count bars where we held a position.
    # We infer this from the equity curve vs cash: when equity > cash-equivalent
    # we're in a position. Simpler: reconstruct from trade entry/exit bars.
    exposed_bars = set()
    for t in trades:
        for b in range(t.entry_bar, t.exit_bar + 1):
            exposed_bars.add(b)
    exposure_pct = len(exposed_bars) / len(ec) * 100.0 if len(ec) > 0 else 0.0

    metrics = {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "r_multiple": r_multiple,
        "num_trades": num_trades,
        "exposure_pct": exposure_pct,
    }

    result.metrics = metrics
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_metrics(result: BacktestResult, title: str = "", file=None, bnh: Optional["BacktestResult"] = None) -> None:
    """
    Print a formatted table of all backtest metrics.

    Assumes compute_metrics() has already been called on the result.
    If metrics is empty, compute_metrics() is called automatically.

    Parameters
    ----------
    result : BacktestResult
        A fully-run backtest result.
    title : str
        Optional label printed in the header bar.
    file : file-like, optional
        Where to write output. Defaults to sys.stdout.
    """
    import sys
    out = file if file is not None else sys.stdout

    if not result.metrics:
        compute_metrics(result)

    m = result.metrics

    def _pct(val: float, sign: bool = True) -> str:
        prefix = "+" if sign and val >= 0 else ""
        return f"{prefix}{val:.1f}%"

    def _float2(val: float) -> str:
        return "∞" if math.isinf(val) else f"{val:.2f}"

    def _int(val: float) -> str:
        return f"{int(val)}"

    if title:
        print(f"## {title}\n", file=out)

    rows = [
        ("Total return",  _pct(m["total_return_pct"]),         "compare to buy-and-hold over the same period"),
        ("CAGR",          _pct(m["cagr_pct"]),                 "> 15% solid, > 30% strong"),
        ("Sharpe",        _float2(m["sharpe"]),                "> 1 decent, > 2 strong, > 3 suspicious"),
        ("Sortino",       _float2(m["sortino"]),               "like Sharpe but penalises downside vol only"),
        ("Max drawdown",  _pct(m["max_drawdown_pct"]),         "on closed-trade equity; lower is better; > 50% is brutal"),
        ("Calmar",        _float2(m["calmar"]),                "CAGR / max DD; > 0.5 decent, > 1 good"),
        ("Win rate",      _pct(m["win_rate_pct"], sign=False), "30–45% normal for trend, 60–75% for mean-rev"),
        ("Profit factor", _float2(m["profit_factor"]),         "> 1.5 decent, > 2 good"),
        ("Avg win",       _pct(m["avg_win_pct"]),              "compare to avg loss below"),
        ("Avg loss",      f"-{m['avg_loss_pct']:.1f}%",        "keep smaller than avg win"),
        ("R-multiple",    _float2(m["r_multiple"]),            "avg win / avg loss; > 1.5 sustains a low win rate"),
        ("Trades",        _int(m["num_trades"]),               "< 30 = stats unreliable; aim for 100+"),
        ("Exposure",      _pct(m["exposure_pct"], sign=False), "% of bars in market; higher = more capital utilised"),
    ]

    print("| Metric | Value | Note |", file=out)
    print("|:---|---:|:---|", file=out)
    for label, value, note in rows:
        print(f"| {label} | {value} | {note} |", file=out)

    if bnh is not None:
        if not bnh.metrics:
            compute_metrics(bnh)
        bm = bnh.metrics
        bnh_dd = float(bnh.drawdown.min() * 100)  # mark-to-market: B&H never closes intraperiod
        bnh_rows = [
            ("B&H return",  _pct(bm["total_return_pct"]), "buy-and-hold over the same period"),
            ("B&H CAGR",    _pct(bm["cagr_pct"]),         "annualised buy-and-hold return"),
            ("B&H Sharpe",  _float2(bm["sharpe"]),        "buy-and-hold risk-adjusted return"),
            ("B&H max DD",  _pct(bnh_dd),                 "buy-and-hold worst intra-period drawdown"),
        ]
        print("| *Buy & Hold* | | |", file=out)
        for label, value, note in bnh_rows:
            print(f"| {label} | {value} | {note} |", file=out)

    print(file=out)


def print_trades(result: BacktestResult, file=None) -> None:
    """
    Print a trade-by-trade table.

    Parameters
    ----------
    result : BacktestResult
        A fully-run backtest result.
    file : file-like, optional
        Where to write output. Defaults to sys.stdout.
    """
    import sys
    out = file if file is not None else sys.stdout

    trades = result.trades
    if not trades:
        print("No trades.", file=out)
        return

    print(f"## Trades ({len(trades)} total)\n", file=out)
    print("| # | Entry date | Entry $ | Exit date | Exit $ | PnL% | W/L | Reason |", file=out)
    print("|--:|:----------|-------:|:---------|------:|-----:|:---:|:-------|", file=out)
    for i, t in enumerate(trades, 1):
        pnl_pct = (t.exit_price / t.entry_price - 1) * 100
        wl = "W" if t.pnl > 0 else "L"
        print(
            f"| {i} | {t.entry_date.date()} | {t.entry_price:,.2f} | "
            f"{t.exit_date.date()} | {t.exit_price:,.2f} | {pnl_pct:+.2f}% | {wl} | {t.exit_reason} |",
            file=out,
        )
    print(file=out)


def save_run(result: BacktestResult, path: str, title: str = "", chart_path: Optional[str] = None, bnh: Optional[BacktestResult] = None) -> str:
    """
    Write the full metrics table and trade list to a text file.

    Creates the directory if it does not exist. Returns the path written.

    Parameters
    ----------
    result : BacktestResult
    path : str
        Destination file path (e.g. "results/ema-cross/2026-06-20_BTC-USDT_1d.txt").
    title : str
        Header line written at the top of the file.
    """
    import os, datetime
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if title:
            print(f"# {title}\n", file=f)
        print(f"*Run: {datetime.date.today()}*\n", file=f)
        if chart_path:
            chart_filename = os.path.basename(chart_path)
            print(f"![Chart]({chart_filename})\n", file=f)
        print_metrics(result, file=f, bnh=bnh)
        print_trades(result, file=f)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(
    result: BacktestResult,
    gc_df: Optional[pd.DataFrame] = None,
    title: str = "",
    compare: Optional[list[BacktestResult]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Produce a 3-panel chart summarising the backtest.

    Panel 1 (price): OHLC bars, optional Gaussian Channel overlay, and
    entry/exit fill markers.

    Panel 2 (equity): equity curve for the main result, plus any comparison
    strategies passed via `compare` (drawn in grey dashed lines).

    Panel 3 (drawdown): drawdown as a red filled area below zero.

    Parameters
    ----------
    result : BacktestResult
        The primary strategy result to plot.
    gc_df : Optional[pd.DataFrame]
        Optional Gaussian Channel DataFrame with columns: filt, hband, lband.
        If provided these three series are drawn over the price panel.
    title : str
        Chart title (shown at the top of the figure).
    compare : Optional[list[BacktestResult]]
        Additional backtest results to overlay on the equity panel only.
        Each is drawn as a dashed grey line, labelled by its index in the list.
    """
    plt.style.use("dark_background")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 2, 1]},
        sharex=False,
    )
    ax_price, ax_equity, ax_dd = axes

    df = result.df
    ec = result.equity_curve
    dd = result.drawdown

    # Use integer x-axis for price so bars are evenly spaced,
    # then replace x-tick labels with dates.
    n = len(df)
    x = np.arange(n)
    date_labels = df.index

    # --- Panel 1: Price OHLC bars ---
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    for i in range(n):
        is_up = closes[i] >= opens[i]
        color = "#26a69a" if is_up else "#ef5350"  # teal up, red down
        # High-low wick
        ax_price.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.8, zorder=2)
        # Candle body — filled rectangle from open to close
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i]) or 1e-6  # doji guard: avoid zero-height patch
        rect = mpatches.Rectangle(
            (x[i] - 0.4, body_bottom), 0.8, body_height,
            facecolor=color, edgecolor=color, linewidth=0, zorder=3,
        )
        ax_price.add_patch(rect)

    # Gaussian channel overlay
    if gc_df is not None:
        gc_x = [df.index.get_loc(ts) for ts in gc_df.index if ts in df.index]
        if "filt" in gc_df.columns:
            ax_price.plot(gc_x, gc_df["filt"].reindex(df.index).dropna(), color="#2196F3",
                          linewidth=1.2, label="GC filt")
        if "hband" in gc_df.columns:
            ax_price.plot(gc_x, gc_df["hband"].reindex(df.index).dropna(), color="#66BB6A",
                          linewidth=0.8, linestyle="--", label="GC hband")
        if "lband" in gc_df.columns:
            ax_price.plot(gc_x, gc_df["lband"].reindex(df.index).dropna(), color="#EF5350",
                          linewidth=0.8, linestyle="--", label="GC lband")

    # Entry/exit markers at fill prices
    for t in result.trades:
        if t.entry_bar < n:
            ax_price.scatter(
                t.entry_bar,
                t.entry_price,
                marker="^",
                color="#00E676",
                s=60,
                zorder=5,
            )
        if t.exit_bar < n:
            marker_color = "#FF1744" if t.exit_reason == "stop" else "#FF6D00"
            ax_price.scatter(
                t.exit_bar,
                t.exit_price,
                marker="v",
                color=marker_color,
                s=60,
                zorder=5,
            )

    # Legend patches for markers
    entry_patch = mpatches.Patch(color="#00E676", label="Entry fill")
    exit_patch = mpatches.Patch(color="#FF6D00", label="Exit (signal)")
    stop_patch = mpatches.Patch(color="#FF1744", label="Exit (stop)")
    ax_price.legend(handles=[entry_patch, exit_patch, stop_patch], fontsize=7, loc="upper left")

    # X-tick labels: show ~8 evenly spaced dates
    tick_positions = np.linspace(0, n - 1, min(8, n), dtype=int)
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(
        [date_labels[i].strftime("%Y-%m-%d") for i in tick_positions],
        fontsize=7,
        rotation=30,
    )
    ax_price.set_xlim(-1, n)
    ax_price.set_ylabel("Price", fontsize=9)
    ax_price.grid(True, alpha=0.15)
    if title:
        ax_price.set_title(title, fontsize=11, pad=6)

    # --- Panel 2: Equity curves ---
    # Use the DatetimeIndex directly on the equity panel
    ax_equity.plot(ec.index, ec.values, color="#2196F3", linewidth=1.4, label="Strategy")

    compare_colors = ["#9E9E9E", "#BDBDBD", "#757575", "#616161"]
    if compare:
        for idx, cmp in enumerate(compare):
            color = compare_colors[idx % len(compare_colors)]
            label = getattr(cmp, "_label", f"Compare {idx + 1}")
            ax_equity.plot(
                cmp.equity_curve.index,
                cmp.equity_curve.values,
                color=color,
                linewidth=1.0,
                linestyle="--",
                label=label,
            )

    ax_equity.axhline(result.initial_capital, color="#FFFFFF", linewidth=0.5, linestyle=":")
    ax_equity.set_ylabel("Equity ($)", fontsize=9)
    ax_equity.legend(fontsize=8, loc="upper left")
    ax_equity.grid(True, alpha=0.15)

    # --- Panel 3: Drawdown ---
    ax_dd.fill_between(dd.index, dd.values * 100, 0, color="#EF5350", alpha=0.7)
    ax_dd.set_ylabel("Drawdown %", fontsize=9)
    ax_dd.grid(True, alpha=0.15)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reference strategies
# ---------------------------------------------------------------------------


def buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
) -> BacktestResult:
    """
    Simple buy-and-hold reference strategy.

    Buys at the open of bar 1 (index position 1) using 100% of capital,
    holds through all bars, and sells at the last bar's close.

    This gives a fair comparison baseline: same capital, same start date,
    same asset. If your strategy doesn't beat this risk-adjusted, "made money"
    is the wrong frame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex.
    initial_capital : float
        Starting capital. Default: 10,000.
    commission_pct : float
        Commission per side as a fraction. Default: 0.001.

    Returns
    -------
    BacktestResult
        Fully populated result with metrics already computed.
    """
    if len(df) < 2:
        raise ValueError("df must have at least 2 bars for buy_and_hold")

    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # Buy at bar 1's open
    entry_price = opens[1]
    entry_comm = entry_price * commission_pct
    # How many units can we afford?
    qty = (initial_capital - initial_capital * commission_pct) / entry_price

    equity_values = np.empty(n, dtype=float)
    cash_after_entry = initial_capital - entry_price * qty - entry_price * qty * commission_pct

    # Bar 0: we haven't bought yet — equity = cash = initial_capital
    equity_values[0] = initial_capital

    for i in range(1, n):
        equity_values[i] = cash_after_entry + qty * closes[i]

    # Final exit at last bar's close (already marked at close; record the trade)
    last_i = n - 1
    exit_price = closes[last_i]
    exit_comm = exit_price * qty * commission_pct
    pnl = (exit_price - entry_price) * qty - entry_price * qty * commission_pct - exit_comm
    entry_value = entry_price * qty
    pnl_pct = pnl / entry_value * 100.0 if entry_value > 0 else 0.0

    trade = Trade(
        entry_bar=1,
        entry_date=df.index[1],
        entry_price=entry_price,
        exit_bar=last_i,
        exit_date=df.index[last_i],
        exit_price=exit_price,
        qty=qty,
        pnl=pnl,
        pnl_pct=pnl_pct,
        exit_reason="signal",
    )

    equity_curve = pd.Series(equity_values, index=df.index, name="equity")
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    drawdown.name = "drawdown"

    result = BacktestResult(
        equity_curve=equity_curve,
        drawdown=drawdown,
        trades=[trade],
        initial_capital=initial_capital,
        df=df,
    )
    result._label = "Buy & Hold"  # type: ignore[attr-defined]
    compute_metrics(result)
    return result


def dca(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    freq_bars: int = 30,
    commission_pct: float = 0.001,
) -> BacktestResult:
    """
    Dollar-cost averaging (DCA) reference strategy.

    Divides the total capital into equal tranches and invests one tranche at
    the open of every ``freq_bars``-th bar, starting from bar 0. Never sells
    (holds to the end). This mirrors how many retail investors build positions
    over time and is a useful benchmark for any systematic strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex.
    initial_capital : float
        Total capital to deploy across all buy events. Default: 10,000.
    freq_bars : int
        Number of bars between each buy. Default: 30 (roughly monthly on
        daily bars). Adjust to match your bar size.
    commission_pct : float
        Commission per side as a fraction. Default: 0.001.

    Returns
    -------
    BacktestResult
        Fully populated result with metrics already computed.
    """
    if len(df) < 1:
        raise ValueError("df must have at least 1 bar for dca")

    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # How many buy events fit?
    n_buys = max(1, n // freq_bars)
    tranche = initial_capital / n_buys  # dollars per buy event

    buy_indices = [i * freq_bars for i in range(n_buys) if i * freq_bars < n]

    # Simulate: track cash and total units held
    cash = initial_capital
    total_qty = 0.0
    completed_trades: list[Trade] = []
    equity_values = np.empty(n, dtype=float)

    buy_set = set(buy_indices)

    for i in range(n):
        if i in buy_set and cash >= tranche:
            buy_price = opens[i]
            invest = min(tranche, cash)  # guard against rounding
            comm = invest * commission_pct
            qty_bought = (invest - comm) / buy_price if buy_price > 0 else 0.0
            cash -= invest
            total_qty += qty_bought
            # Record as an open trade — we'll close all at the end
            completed_trades.append(
                Trade(
                    entry_bar=i,
                    entry_date=df.index[i],
                    entry_price=buy_price,
                    exit_bar=n - 1,         # placeholder — sold at end
                    exit_date=df.index[-1],
                    exit_price=closes[-1],
                    qty=qty_bought,
                    pnl=0.0,                # filled below
                    pnl_pct=0.0,
                    exit_reason="signal",
                )
            )

        equity_values[i] = cash + total_qty * closes[i]

    # Finalise trade P&Ls using the last close
    last_price = closes[-1]
    for t in completed_trades:
        exit_comm = last_price * t.qty * commission_pct
        entry_comm = t.entry_price * t.qty * commission_pct
        t.pnl = (last_price - t.entry_price) * t.qty - entry_comm - exit_comm
        entry_value = t.entry_price * t.qty
        t.pnl_pct = t.pnl / entry_value * 100.0 if entry_value > 0 else 0.0

    equity_curve = pd.Series(equity_values, index=df.index, name="equity")
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    drawdown.name = "drawdown"

    result = BacktestResult(
        equity_curve=equity_curve,
        drawdown=drawdown,
        trades=completed_trades,
        initial_capital=initial_capital,
        df=df,
    )
    result._label = f"DCA (every {freq_bars} bars)"  # type: ignore[attr-defined]
    compute_metrics(result)
    return result
