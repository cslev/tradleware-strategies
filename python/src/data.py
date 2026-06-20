"""
OHLCV data fetching and local parquet caching.

Crypto:   ccxt (default exchange: OKX, swap via EXCHANGE env var or fetch_crypto() param)
Equities: yfinance

Cache layout:
  data/crypto/<exchange>/<symbol_safe>/<timeframe>.parquet
  data/equities/<symbol>/<interval>.parquet

Where <symbol_safe> replaces "/" with "_"  (e.g. BTC/USDT → BTC_USDT).

Typical usage:
  from src.data import fetch_crypto, fetch_equity

  df = fetch_crypto("BTC/USDT", "1d", since="2020-01-01")
  df = fetch_equity("SPY", interval="1d", start="2020-01-01")

Run as a script to refresh all cached symbols:
  cd python && python -m src.data --refresh-all
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import time

import ccxt
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # python/src/data.py → repo root
_DATA_ROOT = _REPO_ROOT / "data"
_CRYPTO_ROOT = _DATA_ROOT / "crypto"
_EQUITY_ROOT = _DATA_ROOT / "equities"


# ---------------------------------------------------------------------------
# Exchange factory — swap OKX for any other ccxt exchange by name
# ---------------------------------------------------------------------------

def _make_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate a ccxt exchange by id (e.g. 'okx', 'binance', 'bybit').

    Reads API credentials from env vars <EXCHANGE_ID>_API_KEY and
    <EXCHANGE_ID>_API_SECRET if present (needed only for private endpoints;
    public OHLCV data works without credentials).
    """
    prefix = exchange_id.upper()
    options: dict = {}
    api_key = os.getenv(f"{prefix}_API_KEY")
    api_secret = os.getenv(f"{prefix}_API_SECRET")
    if api_key:
        options["apiKey"] = api_key
    if api_secret:
        options["secret"] = api_secret

    if not hasattr(ccxt, exchange_id):
        raise ValueError(
            f"'{exchange_id}' is not a supported ccxt exchange. "
            f"Run: python -c \"import ccxt; print(ccxt.exchanges)\" to list all supported exchanges."
        )
    exchange_class = getattr(ccxt, exchange_id)
    # enableRateLimit makes ccxt automatically sleep between requests to avoid 429 errors.
    options["enableRateLimit"] = True
    exchange = exchange_class(options)
    logger.info("connected to %s (ccxt %s)", exchange_id, ccxt.__version__)
    return exchange


def _check_exchange_capabilities(exchange: ccxt.Exchange, timeframe: str) -> None:
    """Raise a clear error if the exchange can't do what we need before making any API calls."""
    if not exchange.has.get("fetchOHLCV"):
        raise ValueError(
            f"{exchange.id} does not support OHLCV candle data (fetchOHLCV=False). "
            f"Try a different exchange — OKX and Binance both support it."
        )
    # Load timeframes only if the exchange publishes them (not all do).
    supported = getattr(exchange, "timeframes", None)
    if supported and timeframe not in supported:
        raise ValueError(
            f"{exchange.id} does not support the '{timeframe}' timeframe. "
            f"Supported timeframes: {sorted(supported.keys())}"
        )
    logger.info("%s supports fetchOHLCV ✓  timeframe '%s' ✓", exchange.id, timeframe)


# ---------------------------------------------------------------------------
# Crypto — ccxt
# ---------------------------------------------------------------------------

_DEFAULT_EXCHANGE = os.getenv("EXCHANGE", "binance")

# How many candles ccxt returns per request varies by exchange.
# OKX allows up to 300; use a conservative default that works broadly.
_CANDLES_PER_REQUEST = 300

# Millisecond duration of each ccxt timeframe string.
# Used to calculate the exact limit to pass when an `until` cutoff is set,
# so the exchange never sends candles past the cutoff in the first place.
_TIMEFRAME_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
    "3d":  259_200_000,
    "1w":  604_800_000,
    "1M":  2_592_000_000,  # approximate (30 days)
}


def fetch_crypto(
    symbol: str,
    timeframe: str = "1d",
    since: Optional[str] = None,
    until: Optional[str] = None,
    exchange_id: Optional[str] = None,
    refresh: bool = False,
    csv: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data for a crypto pair, using the local parquet cache when possible.

    Handles three cases automatically:
    - No cache: full fetch from `since` to `until` (or now).
    - Cache exists, `since` is before the earliest cached bar: backfills the gap,
      then also appends any new candles after the latest cached bar.
    - Cache exists, `since` >= earliest cached bar: appends new candles only.

    Pass refresh=True to discard the cache and re-fetch everything from scratch.

    Args:
        symbol:      ccxt market symbol, e.g. "BTC/USDT".
        timeframe:   ccxt timeframe string: "1m", "5m", "1h", "4h", "1d", etc.
        since:       ISO-8601 start date string, e.g. "2020-01-01". On the first
                     fetch this is the start of history. On later calls, if this is
                     earlier than what's already cached, the gap is backfilled.
        until:       ISO-8601 end date string, e.g. "2024-12-31". Candles after
                     this date are excluded. Defaults to now.
        exchange_id: ccxt exchange id to use. Defaults to the EXCHANGE env var
                     ("okx" if not set).
        refresh:     If True, ignore the cache and re-fetch everything from `since`.
        csv:         If True, also write a human-readable CSV alongside the parquet file.
        verbose:     If True, show a live progress bar while fetching.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume  — all float64.
    """
    xid = (exchange_id or _DEFAULT_EXCHANGE).lower()
    cache_path = _crypto_cache_path(xid, symbol, timeframe)

    tf_ms = _TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Add it to _TIMEFRAME_MS if needed.")

    since_ms: Optional[int] = None
    if since:
        since_ms = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000)

    until_ms: Optional[int] = None
    if until:
        until_ms = int(datetime.fromisoformat(until).replace(tzinfo=timezone.utc).timestamp() * 1000)

    if refresh or not cache_path.exists():
        logger.info("full fetch: %s %s %s→%s on %s",
                    symbol, timeframe, since or "start", until or "now", xid)
        exchange = _make_exchange(xid)
        _check_exchange_capabilities(exchange, timeframe)
        candles = _paginate_candles(exchange, symbol, timeframe, tf_ms, since_ms, until_ms, verbose)
        df = _candles_to_dataframe(candles)
        _write_parquet(df, cache_path, csv=csv)
        logger.info("saved %d rows → %s", len(df), cache_path)
        return df

    cached = _read_parquet(cache_path)
    cached_start_ms = int(cached.index.min().timestamp() * 1000)
    cached_end_ms   = int(cached.index.max().timestamp() * 1000)
    exchange = _make_exchange(xid)
    _check_exchange_capabilities(exchange, timeframe)
    pieces: list[pd.DataFrame] = []

    # --- backfill: requested start is before what we have ---
    if since_ms is not None and since_ms < cached_start_ms:
        logger.info("backfilling %s %s from %s to cached start %s",
                    symbol, timeframe, since, cached.index.min().date())
        back_candles = _paginate_candles(
            exchange, symbol, timeframe, tf_ms,
            since_ms=since_ms,
            until_ms=cached_start_ms - 1,
            verbose=verbose,
            allow_empty=True,
        )
        if back_candles:
            pieces.append(_candles_to_dataframe(back_candles))

    pieces.append(cached)

    # --- forward update: append candles newer than what we have ---
    forward_until_ms = until_ms  # None means "up to now"
    if forward_until_ms is None or forward_until_ms > cached_end_ms:
        logger.info("forward update: %s %s from %s",
                    symbol, timeframe, cached.index.max().date())
        fwd_candles = _paginate_candles(
            exchange, symbol, timeframe, tf_ms,
            since_ms=cached_end_ms + 1,
            until_ms=forward_until_ms,
            verbose=verbose,
            allow_empty=True,
        )
        if fwd_candles:
            fwd_df = _candles_to_dataframe(fwd_candles)
            # Drop the last cached bar — it may have been a still-forming candle
            fwd_df = fwd_df[fwd_df.index > cached.index.max()]
            pieces.append(fwd_df)

    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _write_parquet(df, cache_path, csv=csv)
    logger.info("saved %d rows → %s", len(df), cache_path)
    return df


def _paginate_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    tf_ms: int,
    since_ms: Optional[int],
    until_ms: Optional[int],
    verbose: bool = False,
    allow_empty: bool = False,
) -> list[list]:
    """Paginate ccxt fetch_ohlcv between since_ms and until_ms, returning raw candle lists.

    allow_empty=True skips retry/probe logic when an empty first batch is expected
    (e.g. backfill or forward update — the symbol is known to exist, so empty just
    means there are no candles in the requested range yet).
    allow_empty=False (default) treats an empty first batch as suspicious and retries,
    then probes to produce a useful error message (used for initial full fetches).
    """
    candles: list[list] = []
    cursor = since_ms

    total = None
    if verbose and since_ms is not None and until_ms is not None:
        total = max(1, (until_ms - since_ms) // tf_ms + 1)

    bar = tqdm(
        total=total,
        unit="candle",
        desc=f"{symbol} {timeframe}",
        disable=not verbose,
    )

    with bar:
        while True:
            if until_ms is not None and cursor is not None:
                remaining = max(1, (until_ms - cursor) // tf_ms + 1)
                limit = min(_CANDLES_PER_REQUEST, remaining)
            else:
                limit = _CANDLES_PER_REQUEST

            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            except ccxt.BadSymbol as e:
                raise ValueError(
                    f"'{symbol}' is not a valid symbol on {exchange.id}.\n"
                    f"Exchange says: {e}\n"
                    f"Tip: symbol names are exchange-specific (e.g. Coinbase uses BTC/USD, not BTC/USDT)."
                ) from e
            except ccxt.RateLimitExceeded as e:
                raise ValueError(f"rate limit hit on {exchange.id}: {e}") from e
            except ccxt.NetworkError as e:
                raise ValueError(f"network error talking to {exchange.id}: {e}") from e
            except ccxt.ExchangeError as e:
                raise ValueError(f"{exchange.id} returned an error: {e}") from e

            if not batch:
                if not candles and not allow_empty:
                    # First batch is empty on an initial fetch — could be a transient
                    # rate-limit that ccxt didn't surface as an exception.
                    # Retry up to 3 times with backoff before concluding there's no data.
                    _RETRY_DELAYS = [5, 15, 30]
                    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
                        logger.warning("\n"
                            "empty response on first request (possible rate limit) — "
                            "retrying in %ds (attempt %d/%d)…", delay, attempt, len(_RETRY_DELAYS)
                        )
                        time.sleep(delay)
                        try:
                            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
                        except (ccxt.NetworkError, ccxt.ExchangeError):
                            batch = []
                        if batch:
                            break

                if not batch:
                    if not candles and not allow_empty:
                        # Still empty after retries — probe to produce a useful error.
                        try:
                            markets = exchange.load_markets()
                            if symbol not in markets:
                                base = symbol.split("/")[0]
                                similar = [s for s in sorted(markets) if base in s][:10]
                                raise ValueError(
                                    f"'{symbol}' does not exist on {exchange.id}.\n"
                                    f"Similar symbols: {similar or list(sorted(markets))[:10]}"
                                )
                            # Symbol exists — fetch from Unix epoch to find the earliest candle
                            probe = exchange.fetch_ohlcv(symbol, timeframe, since=0, limit=1)
                            if probe:
                                earliest = datetime.fromtimestamp(
                                    probe[0][0] / 1000, tz=timezone.utc
                                ).strftime("%Y-%m-%d")
                                raise ValueError(
                                    f"'{symbol}' on {exchange.id} was listed on {earliest} — "
                                    f"no data exists before that date.\n"
                                    f"Fix: use --since {earliest}"
                                )
                        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                            logger.warning("could not probe symbol availability: %s", e)
                        raise ValueError(
                            f"'{symbol}' returned no candles from {exchange.id} after retries. "
                            f"Possible causes: rate limit still active, symbol wrong, or no data "
                            f"for this date range. Wait a minute and try again."
                        )
                    break

                # Batch was recovered after retry — continue normally
                candles.extend(batch)
                bar.update(len(batch))
                last_ts = batch[-1][0]
                if verbose:
                    date_str = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    bar.set_postfix({"up to": date_str})
                if until_ms is not None and last_ts >= until_ms:
                    break
                cursor = last_ts + 1
                continue

            candles.extend(batch)
            bar.update(len(batch))
            last_ts = batch[-1][0]
            if verbose:
                date_str = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                bar.set_postfix({"up to": date_str})
            if until_ms is not None and last_ts >= until_ms:
                break
            cursor = last_ts + 1

    return candles


def _crypto_cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    symbol_safe = symbol.replace("/", "_")
    return _CRYPTO_ROOT / exchange_id / symbol_safe / f"{timeframe}.parquet"


def _candles_to_dataframe(candles: list[list]) -> pd.DataFrame:
    """Convert raw ccxt OHLCV list-of-lists to a tidy DataFrame."""
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


# ---------------------------------------------------------------------------
# Equities — yfinance
# ---------------------------------------------------------------------------

def fetch_equity(
    symbol: str,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
    csv: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for a stock/ETF via yfinance, caching to parquet.

    On the first call, fetches from `start` to now and saves to parquet.
    On subsequent calls, reads the cache and fetches only the candles newer
    than the last cached timestamp — then appends and saves.
    Pass refresh=True to discard the cache and re-fetch from scratch.

    Args:
        symbol:   Ticker symbol, e.g. "SPY", "AAPL".
        interval: yfinance interval string: "1d", "1wk", "1mo", "1h", "5m", etc.
                  Note: intraday intervals (<= "1h") only go back ~60–730 days
                  depending on the interval.
        start:    Start date string "YYYY-MM-DD". Only used on the first fetch
                  (no cache). Ignored once a cache file exists.
        end:      End date string "YYYY-MM-DD". Defaults to today. Always respected,
                  even on incremental updates — rows after this date are excluded.
        refresh:  If True, ignore the cache and re-fetch everything.
        csv:      If True, also write a human-readable CSV alongside the parquet file.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume  — all float64.
    """
    cache_path = _equity_cache_path(symbol, interval)

    if refresh or not cache_path.exists():
        logger.info("full fetch: %s %s %s→%s from yfinance",
                    symbol, interval, start or "start", end or "today")
        df = _yfinance_fetch(symbol, interval, start=start, end=end)
        _write_parquet(df, cache_path, csv=csv)
        logger.info("saved %d rows → %s", len(df), cache_path)
        return df

    cached = _read_parquet(cache_path)
    pieces: list[pd.DataFrame] = []

    # --- backfill: requested start is before what we have ---
    if start is not None and start < cached.index.min().strftime("%Y-%m-%d"):
        logger.info("backfilling %s %s from %s to cached start %s",
                    symbol, interval, start, cached.index.min().date())
        back_end = cached.index.min().strftime("%Y-%m-%d")
        back_df = _yfinance_fetch(symbol, interval, start=start, end=back_end)
        back_df = back_df[back_df.index < cached.index.min()]
        if not back_df.empty:
            pieces.append(back_df)

    pieces.append(cached)

    # --- forward update: append rows newer than what we have ---
    if end is None or end > cached.index.max().strftime("%Y-%m-%d"):
        logger.info("forward update: %s %s from %s", symbol, interval, cached.index.max().date())
        fwd_start = cached.index.max().strftime("%Y-%m-%d")
        fwd_df = _yfinance_fetch(symbol, interval, start=fwd_start, end=end)
        fwd_df = fwd_df[fwd_df.index > cached.index.max()]
        if not fwd_df.empty:
            pieces.append(fwd_df)

    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _write_parquet(df, cache_path, csv=csv)
    logger.info("saved %d rows → %s", len(df), cache_path)
    return df


def _yfinance_fetch(
    symbol: str,
    interval: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """Single yfinance history call, normalised to the standard DataFrame shape."""
    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        interval=interval,
        start=start,
        end=end,
        auto_adjust=True,   # folds in splits and dividends — correct for backtesting
        actions=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol} ({interval}) {start}→{end}")
    return _normalize_equity_df(raw)


def _equity_cache_path(symbol: str, interval: str) -> Path:
    return _EQUITY_ROOT / symbol.upper() / f"{interval}.parquet"


def _normalize_equity_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise yfinance output to match the crypto DataFrame shape."""
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    # Make the index tz-aware UTC so it's consistent with crypto data
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path, csv: bool = False) -> None:
    if df.empty:
        logger.warning("no data to write — skipping %s (directory not created)", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    if csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)
        logger.info("wrote CSV → %s", csv_path)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# CLI — python -m src.data --refresh-all
# ---------------------------------------------------------------------------

# Edit these lists to add/remove default symbols for bulk refresh.
_DEFAULT_CRYPTO_SYMBOLS = [
    ("BTC/USDT", "1d"),
    ("ETH/USDT", "1d"),
]

_DEFAULT_EQUITY_SYMBOLS = [
    ("SPY", "1d"),
    ("QQQ", "1d"),
]


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache OHLCV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # incremental update for all default symbols
  python -m src.data --update

  # fetch a specific crypto symbol at a specific timeframe
  python -m src.data --crypto BTC/USDT --timeframe 4h --since 2023-01-01

  # fetch a specific equity
  python -m src.data --equity SPY --timeframe 1d --since 2020-01-01

  # fetch multiple symbols in one call
  python -m src.data --crypto BTC/USDT ETH/USDT --timeframe 1h

  # fetch only up to a cutoff (useful for testing incremental updates)
  python -m src.data --crypto BTC/USDT --timeframe 1d --until 2024-12-31

  # full re-fetch from scratch
  python -m src.data --crypto BTC/USDT --timeframe 1d --since 2020-01-01 --refresh

  # use a different exchange
  python -m src.data --crypto BTC/USDT --timeframe 1d --exchange binance
        """,
    )

    # --- what to fetch ---
    parser.add_argument(
        "--update", action="store_true",
        help="Fetch/update all default symbols defined in _DEFAULT_CRYPTO_SYMBOLS "
             "and _DEFAULT_EQUITY_SYMBOLS (incremental by default).",
    )
    parser.add_argument(
        "--crypto", nargs="+", metavar="SYMBOL",
        help="One or more crypto symbols to fetch, e.g. BTC/USDT ETH/USDT.",
    )
    parser.add_argument(
        "--equity", nargs="+", metavar="SYMBOL",
        help="One or more equity/ETF tickers to fetch, e.g. SPY QQQ AAPL.",
    )
    parser.add_argument(
        "--timeframe", default="1d", metavar="TF",
        help="Candle timeframe for crypto (ccxt format: 1m 5m 15m 1h 4h 1d 1w …) "
             "and equities (yfinance format: 1m 5m 15m 1h 1d 1wk 1mo). (default: 1d)",
    )

    # --- date range ---
    parser.add_argument(
        "--since", default="2020-01-01", metavar="YYYY-MM-DD",
        help="Earliest date to fetch. On first run this is the history start. "
             "On later runs, a date earlier than the cache triggers a backfill. "
             "(default: 2020-01-01)",
    )
    parser.add_argument(
        "--until", default=None, metavar="YYYY-MM-DD",
        help="Latest date to fetch (inclusive). Omit to fetch up to today.",
    )

    # --- options ---
    parser.add_argument(
        "--exchange", default=_DEFAULT_EXCHANGE, metavar="ID",
        help=f"ccxt exchange id for crypto symbols (default: {_DEFAULT_EXCHANGE}). "
             f"Common ids: okx, binance, bybit, coinbaseadvanced (use this for Coinbase — "
             f"'coinbase' has limited history), kraken.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Discard the cache and re-fetch everything from --since.",
    )
    parser.add_argument(
        "--no-csv", dest="csv", action="store_false",
        help="Skip writing a CSV file alongside the parquet.",
    )
    parser.set_defaults(csv=True)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show a live progress bar while downloading candles.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Validate dates early so the user gets a clear message, not a traceback.
    for flag, value in [("--since", args.since), ("--until", args.until)]:
        if value is not None:
            try:
                datetime.fromisoformat(value)
            except ValueError:
                parser.error(f"{flag} '{value}' is not a valid date — use YYYY-MM-DD (e.g. 2024-01-31)")

    if args.since and args.until and args.since > args.until:
        parser.error(f"--since ({args.since}) must be before --until ({args.until})")

    if not any([args.update, args.crypto, args.equity]):
        parser.print_help()
        return

    try:
        if args.update:
            for symbol, tf in _DEFAULT_CRYPTO_SYMBOLS:
                fetch_crypto(symbol, tf, since=args.since, until=args.until,
                             exchange_id=args.exchange, refresh=args.refresh,
                             csv=args.csv, verbose=args.verbose)
            for symbol, interval in _DEFAULT_EQUITY_SYMBOLS:
                fetch_equity(symbol, interval, start=args.since, end=args.until,
                             refresh=args.refresh, csv=args.csv)

        for symbol in (args.crypto or []):
            fetch_crypto(symbol, args.timeframe, since=args.since, until=args.until,
                         exchange_id=args.exchange, refresh=args.refresh,
                         csv=args.csv, verbose=args.verbose)

        for symbol in (args.equity or []):
            fetch_equity(symbol, args.timeframe, start=args.since, end=args.until,
                         refresh=args.refresh, csv=args.csv)

    except ccxt.BadSymbol as e:
        parser.error(f"unknown symbol: {e}")
    except ccxt.NetworkError as e:
        parser.error(f"network error — check your connection: {e}")
    except ccxt.ExchangeError as e:
        parser.error(f"exchange error: {e}")
    except ValueError as e:
        parser.error(str(e))

    print("Done.")


if __name__ == "__main__":
    _cli()
