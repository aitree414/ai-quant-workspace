"""
Market data loader using yfinance.

Supports Taiwan stocks (auto-append .TW) and US stocks.
Provides local disk caching to reduce redundant API calls.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Regular market suffixes recognised by yfinance
KNOWN_SUFFIXES = {".TW", ".TWO", ".HK", ".SS", ".SZ", ".TO", ".L", ".DE", ".PA"}

CACHE_ROOT = Path(os.getenv("DATA_CACHE_DIR", "data/market"))


def _has_suffix(symbol: str) -> bool:
    """Check if symbol already carries a recognised exchange suffix."""
    return any(symbol.endswith(sfx) for sfx in KNOWN_SUFFIXES)


def resolve_ticker(symbol: str) -> str:
    """Normalise a ticker symbol.

    - 4-digit Taiwan stocks (e.g. "2330") → "2330.TW"
    - Already-suffixed symbols (e.g. "AAPL", "2330.TW") → unchanged
    """
    s = symbol.strip().upper()
    if _has_suffix(s):
        return s
    # Taiwan stocks listed on TWSE are numeric (4 digits) or have trailing
    # letters like "2330", "2603", "2888"
    if s.isdigit() or (s[:-1].isdigit() and len(s) <= 5):
        return s + ".TW"
    return s


def _cache_path(symbol: str, interval: str) -> Path:
    """Return the local cache file path for a given symbol and interval."""
    safe_name = symbol.replace(".", "_")
    return CACHE_ROOT / f"{safe_name}_{interval}.parquet"


def load_data(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for one or more symbols.

    Each DataFrame contains columns:
        Open, High, Low, Close, Volume, (Dividends, Stock Splits when available)

    Args:
        symbols: List of ticker symbols (e.g. ["2330", "AAPL", "TSLA"]).
        start: Start date string in ``YYYY-MM-DD`` format.
        end: End date string in ``YYYY-MM-DD`` format.
        interval: Data interval (``1d``, ``1h``, ``1m``, …).
        cache_dir: Override the default cache directory (``data/market``).
        use_cache: When ``True``, read from / write to local parquet cache.

    Returns:
        Dict mapping each symbol to its OHLCV DataFrame.
    """
    _cache_root = Path(cache_dir) if cache_dir else CACHE_ROOT
    if use_cache:
        _cache_root.mkdir(parents=True, exist_ok=True)

    resolved = [resolve_ticker(s) for s in symbols]
    logger.info("Fetching data for %s  [%s → %s, interval=%s]", resolved, start, end, interval)

    result: dict[str, pd.DataFrame] = {}

    for sym in resolved:
        # ---- cache hit ----
        if use_cache:
            cache_file = _cache_path(sym, interval)
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    # Filter to requested date range
                    df = df.loc[start:end]
                    if not df.empty:
                        logger.info("Cache HIT for %s  (%d rows)", sym, len(df))
                        result[sym] = df
                        continue
                except Exception as exc:
                    logger.warning("Cache read failed for %s, re-fetching: %s", sym, exc)

        # ---- fetch from yfinance ----
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning("No data returned for %s", sym)
                result[sym] = pd.DataFrame()
                continue

            # yfinance returns a DatetimeIndex; keep it
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            result[sym] = df

            # ---- write cache ----
            if use_cache:
                try:
                    df.to_parquet(cache_file)
                    logger.info("Cached %s → %s  (%d rows)", sym, cache_file, len(df))
                except Exception as exc:
                    logger.warning("Failed to write cache for %s: %s", sym, exc)

        except Exception as exc:
            logger.exception("Failed to fetch data for %s: %s", sym, exc)
            result[sym] = pd.DataFrame()

    return result


def load_single(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`load_data` for a single symbol."""
    return load_data(
        [symbol], start, end, interval, cache_dir=cache_dir, use_cache=use_cache
    ).get(resolve_ticker(symbol), pd.DataFrame())
