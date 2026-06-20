"""
Custom indicator math — functions not available in pandas-ta.

pandas-ta covers most common indicators (RSI, MACD, ATR, Bollinger, etc.).
Add functions here only when the indicator doesn't exist in pandas-ta or
the pandas-ta version has a known accuracy issue.

Convention: functions take a pd.DataFrame or pd.Series and return the same shape.
"""

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gaussian Channel
# ---------------------------------------------------------------------------

# Binomial coefficients for the f_filt9x recurrence.
# Index: _BINOM[i] gives the list [m2, m3, ..., mi] for pole i.
# Pole 1 needs no m-coefficients; poles 2–9 need the Pascal's-triangle row.
_BINOM: dict[int, list[int]] = {
    1: [],
    2: [1],
    3: [3, 1],
    4: [6, 4, 1],
    5: [10, 10, 5, 1],
    6: [15, 20, 15, 6, 1],
    7: [21, 35, 35, 21, 7, 1],
    8: [28, 56, 70, 56, 28, 8, 1],
    9: [36, 84, 126, 126, 84, 36, 9, 1],
}


def _filt9x(alpha: float, src: np.ndarray, pole: int) -> np.ndarray:
    """
    Apply a single-pole recursive IIR Gaussian filter to a 1-D source array.

    This matches the Pine Script f_filt9x() function from DonovanWall's
    Gaussian Channel indicator.

    The recurrence for bar n is:

        f[n] = alpha^pole * src[n]
             + pole*(1-alpha) * f[n-1]
             - m2*(1-alpha)^2 * f[n-2]   (if pole >= 2)
             + m3*(1-alpha)^3 * f[n-3]   (if pole >= 3)
             - ...

    Signs alternate starting from the m2 term (the f[n-1] term is always +).

    Parameters
    ----------
    alpha : float
        Filter coefficient computed from period and number of poles.
    src : np.ndarray
        1-D float array of source values (already lag-adjusted if requested).
    pole : int
        Which pole level to compute (1–9).

    Returns
    -------
    np.ndarray
        Filtered values, same length as src.
    """
    n = len(src)
    f = np.zeros(n)
    a_i = alpha ** pole          # alpha^pole, constant for all bars
    one_minus_a = 1.0 - alpha
    m_coeffs = _BINOM[pole]      # [m2, m3, ..., mpole] — may be empty for pole=1

    for bar in range(n):
        val = a_i * src[bar]

        # Always add the pole*(1-alpha)*f[bar-1] term when a previous bar exists
        if bar >= 1:
            val += pole * one_minus_a * f[bar - 1]

        # Higher-lag terms: m_{k} * (1-alpha)^k * f[bar-k], alternating sign
        # m_coeffs[j] corresponds to k = j+2 (first entry is m2)
        for j, m_k in enumerate(m_coeffs):
            k = j + 2          # lag index (2, 3, 4, ...)
            if bar < k:
                break          # not enough history yet
            # Signs alternate: k=2 → -, k=3 → +, k=4 → -, ...
            sign = -1.0 if (k % 2 == 0) else 1.0
            val += sign * m_k * (one_minus_a ** k) * f[bar - k]

        f[bar] = val

    return f


def gaussian_channel(
    df: pd.DataFrame,
    poles: int = 4,
    period: int = 144,
    mult: float = 1.414,
    mode_lag: bool = False,
    mode_fast: bool = False,
    src: str = "hlc3",
) -> pd.DataFrame:
    """
    Compute the Gaussian Channel indicator (DonovanWall) on OHLC price data.

    The Gaussian Channel applies a recursive IIR filter (approximating a Gaussian
    blur) to both price and true range, then draws a band around the filtered
    price using filtered ATR * mult.

    Think of it as a super-smooth moving average with self-adjusting bands
    based on volatility — similar in spirit to Bollinger Bands, but the center
    line reacts much more smoothly to price.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: high, low, close. open is optional.
    poles : int, default 4
        Number of filter poles (1–9). More poles → smoother, more lag.
        4 is the original default.
    period : int, default 144
        Lookback period for the filter. Longer → slower-moving channel.
    mult : float, default 1.414
        Band width multiplier applied to the filtered true range.
        1.414 ≈ sqrt(2), the original default.
    mode_lag : bool, default False
        When True, applies a lag-reduction adjustment to the source before
        filtering. Reduces lag but may introduce some noise.
    mode_fast : bool, default False
        When True, the output is the average of the 1-pole and N-pole filters.
        Produces a faster-reacting channel.
    src : str, default "hlc3"
        Which price series to use as the filter input.
        Options: "hlc3" (typical price), "close", "hl2", "ohlc4".

    Returns
    -------
    pd.DataFrame
        Columns:
          - ``filt``  : filtered center line
          - ``hband`` : upper band (filt + filttr * mult)
          - ``lband`` : lower band (filt - filttr * mult)
        Same index as df. Rows before enough history is available will be 0.0.

    Notes
    -----
    The filter is recursive: each output value depends on up to 9 previous
    output values of the same series. Vectorization is not possible — the
    computation uses a Python loop over bars.
    """
    if poles < 1 or poles > 9:
        raise ValueError(f"poles must be between 1 and 9, got {poles}")
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # --- Compute requested source series ---
    if src == "hlc3":
        src_arr = (high + low + close) / 3.0
    elif src == "hl2":
        src_arr = (high + low) / 2.0
    elif src == "ohlc4":
        open_ = df["open"].to_numpy(dtype=float)
        src_arr = (open_ + high + low + close) / 4.0
    elif src == "close":
        src_arr = close.copy()
    else:
        raise ValueError(f"Unknown src '{src}'. Use 'hlc3', 'close', 'hl2', or 'ohlc4'.")

    # --- True range: max(H-L, |H-prevC|, |L-prevC|) ---
    # First bar has no previous close, so TR = H - L.
    tr_arr = np.empty(len(close))
    tr_arr[0] = high[0] - low[0]
    tr_arr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )

    # --- Alpha computation (matches Pine Script formula) ---
    # beta = (1 - cos(4*asin(1)/per)) / (1.414^(2/N) - 1)
    # alpha = -beta + sqrt(beta^2 + 2*beta)
    # Note: Pine Script uses the literal 1.414 (not the full sqrt(2) = 1.41421356...)
    # to match DonovanWall's original formula exactly. Using the full value produces
    # a slightly different alpha that drifts from TradingView over thousands of bars.
    beta = (1.0 - math.cos(4.0 * math.asin(1.0) / period)) / (
        math.pow(1.414, 2.0 / poles) - 1.0
    )
    alpha = -beta + math.sqrt(beta * beta + 2.0 * beta)

    # --- Lag reduction (mode_lag) ---
    # lag = (period - 1) / (2 * poles)
    # adjusted_src[n] = src[n] + (src[n] - src[n - lag])
    # We use integer lag (Pine Script uses integer indexing too).
    lag = int((period - 1) / (2 * poles))

    if mode_lag:
        # Shift the source by `lag` bars; pad the first `lag` bars with the
        # earliest available value to avoid NaN.
        src_lagged = np.empty_like(src_arr)
        src_lagged[:lag] = src_arr[0]
        src_lagged[lag:] = src_arr[:-lag] if lag > 0 else src_arr

        tr_lagged = np.empty_like(tr_arr)
        tr_lagged[:lag] = tr_arr[0]
        tr_lagged[lag:] = tr_arr[:-lag] if lag > 0 else tr_arr

        src_in = src_arr + (src_arr - src_lagged)
        tr_in = tr_arr + (tr_arr - tr_lagged)
    else:
        src_in = src_arr
        tr_in = tr_arr

    # --- Apply the recursive filter at each pole level ---
    # We always need pole-1 (for fast mode) and pole-N (the main output).
    filt1 = _filt9x(alpha, src_in, 1)
    filt1_tr = _filt9x(alpha, tr_in, 1)

    if poles == 1:
        filtn = filt1
        filtn_tr = filt1_tr
    else:
        # Compute intermediate poles; only the final one is kept.
        # Each pole's filter feeds into the next level's source.
        # In DonovanWall's Pine implementation, each f_filt9x call always
        # uses the *original* src — it is not chained. The pole index i in
        # the recurrence already encodes the cascade mathematically.
        filtn = _filt9x(alpha, src_in, poles)
        filtn_tr = _filt9x(alpha, tr_in, poles)

    # --- Fast mode: average 1-pole and N-pole results ---
    if mode_fast:
        filt_out = (filtn + filt1) / 2.0
        filttr_out = (filtn_tr + filt1_tr) / 2.0
    else:
        filt_out = filtn
        filttr_out = filtn_tr

    hband = filt_out + filttr_out * mult
    lband = filt_out - filttr_out * mult

    return pd.DataFrame(
        {"filt": filt_out, "hband": hband, "lband": lband},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------


def stoch_rsi(
    close: pd.Series,
    rsi_length: int = 14,
    stoch_length: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute the Stochastic RSI indicator, matching TradingView's implementation.

    Stochastic RSI combines two indicators:
    1. RSI tells you whether the asset is overbought or oversold.
    2. Stochastic of the RSI tells you where RSI sits within its own recent
       range — giving an earlier signal than raw RSI alone.

    The result is %K (a smoothed stochastic of RSI) and %D (a smoothed %K).
    Values range from 0 to 100; 80+ is overbought, 20- is oversold.

    Parameters
    ----------
    close : pd.Series
        Closing prices. Any index works; it is preserved in the output.
    rsi_length : int, default 14
        Number of bars for the RSI calculation.
    stoch_length : int, default 14
        Lookback window for the stochastic calculation applied to the RSI series.
    k_smooth : int, default 3
        Simple moving average length applied to the raw stochastic to get %K.
    d_smooth : int, default 3
        Simple moving average length applied to %K to get %D.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (k, d) where:
          - ``k`` is the %K line (faster, more reactive)
          - ``d`` is the %D line (slower, signal line)
        Both have the same index as ``close``.

    Notes
    -----
    RSI is computed using Wilder's smoothing method (equivalent to an EMA
    with alpha = 1 / rsi_length), which matches TradingView's built-in RSI.
    """
    close_arr = close.to_numpy(dtype=float)
    n = len(close_arr)

    # --- Step 1: Wilder's RSI ---
    # Wilder's smoothing: EMA with alpha = 1/rsi_length.
    # We seed the average gain/loss with the simple mean of the first rsi_length
    # differences (exactly how TradingView initializes it).
    delta = np.diff(close_arr, prepend=np.nan)  # length n, first value is nan

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi_arr = np.full(n, np.nan)

    if n > rsi_length:
        # Seed: simple average of the first rsi_length gain/loss values
        avg_gain = np.mean(gain[1 : rsi_length + 1])
        avg_loss = np.mean(loss[1 : rsi_length + 1])

        # First RSI value
        if avg_loss == 0.0:
            rsi_arr[rsi_length] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_arr[rsi_length] = 100.0 - 100.0 / (1.0 + rs)

        # Wilder's smoothing for the rest
        alpha_w = 1.0 / rsi_length
        for i in range(rsi_length + 1, n):
            avg_gain = avg_gain * (1.0 - alpha_w) + gain[i] * alpha_w
            avg_loss = avg_loss * (1.0 - alpha_w) + loss[i] * alpha_w
            if avg_loss == 0.0:
                rsi_arr[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_arr[i] = 100.0 - 100.0 / (1.0 + rs)

    # --- Step 2: Stochastic of RSI ---
    # stoch[i] = 100 * (rsi[i] - min(rsi, stoch_length)[i])
    #                / (max(rsi, stoch_length)[i] - min(rsi, stoch_length)[i])
    # When max == min (flat RSI), result is 0 by convention.
    rsi_series = pd.Series(rsi_arr, index=close.index)
    rsi_min = rsi_series.rolling(stoch_length).min()
    rsi_max = rsi_series.rolling(stoch_length).max()
    rsi_range = rsi_max - rsi_min

    # Avoid division by zero: where range is 0, stoch = 0.
    # Clip to [0, 100] to guard against floating-point values marginally outside
    # the valid range (e.g., 100.000000000001 from finite-precision arithmetic).
    stoch_raw = pd.Series(
        np.clip(
            np.where(rsi_range != 0, 100.0 * (rsi_series - rsi_min) / rsi_range, 0.0),
            0.0,
            100.0,
        ),
        index=close.index,
    )

    # --- Step 3: %K = SMA of raw stochastic ---
    k = stoch_raw.rolling(k_smooth).mean()

    # --- Step 4: %D = SMA of %K ---
    d = k.rolling(d_smooth).mean()

    k.name = "stochrsi_k"
    d.name = "stochrsi_d"

    return k, d
