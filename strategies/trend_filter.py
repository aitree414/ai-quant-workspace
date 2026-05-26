"""
Trend filter — weekly MA50 gate for multi-timeframe analysis.

Filters out buy signals when the weekly trend is bearish,
preventing counter-trend entries in downtrends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from strategies.agents.base_agent import Signal

logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    """Result of a trend evaluation."""
    trend_allowed: bool = True       # True if weekly MA50 is flat or up
    weekly_ma50: float = 0.0         # Current weekly MA50 value
    weekly_close: float = 0.0        # Current weekly close price
    weekly_ma50_slope: float = 0.0   # % change of MA50 over last 2 weeks
    trend_direction: str = "unknown" # "up" | "down" | "sideways"
    weekly_bars: int = 0             # Number of weekly bars available


class TrendFilter:
    """Multi-timeframe trend filter using weekly MA50.

    Evaluates the weekly trend and gates buy signals when bearish.
    """

    def __init__(
        self,
        ma_period: int = 50,
        weekly_min_bars: int = 30,
        slope_threshold: float = 0.001,
    ) -> None:
        """
        Args:
            ma_period: Period for weekly moving average.
            weekly_min_bars: Minimum weekly bars for a valid reading.
            slope_threshold: Minimum MA50 slope (as fraction) to qualify as "up".
        """
        self.ma_period = ma_period
        self.weekly_min_bars = weekly_min_bars
        self.slope_threshold = slope_threshold

    def evaluate(self, data: pd.DataFrame) -> TrendResult:
        """Evaluate weekly trend from daily OHLCV data.

        Resamples daily data to weekly, computes MA50, checks slope.
        Returns TrendResult with safe defaults when data is insufficient.
        """
        result = TrendResult()

        if data.empty:
            logger.warning("TrendFilter: empty data, defaulting to allowed")
            return result

        # Resample to weekly (use last close of each week)
        if "Close" not in data.columns:
            logger.warning("TrendFilter: no Close column, defaulting to allowed")
            return result

        weekly = data["Close"].resample("W").last().dropna()
        result.weekly_bars = len(weekly)
        result.weekly_close = float(weekly.iloc[-1]) if not weekly.empty else 0.0

        if len(weekly) < self.weekly_min_bars:
            logger.info(
                "TrendFilter: insufficient weekly bars (%d < %d), defaulting to allowed",
                len(weekly), self.weekly_min_bars,
            )
            return result

        # Compute weekly MA50
        ma50 = weekly.rolling(self.ma_period).mean()
        if ma50.isna().all():
            logger.warning("TrendFilter: MA50 all NaN, defaulting to allowed")
            return result

        result.weekly_ma50 = float(ma50.iloc[-1])

        # Compute slope: compare current MA50 to value 2 weeks ago
        if len(ma50) >= 2 and not pd.isna(ma50.iloc[-2]):
            prev_ma50 = ma50.iloc[-2]
            if prev_ma50 != 0:
                result.weekly_ma50_slope = (ma50.iloc[-1] - prev_ma50) / abs(prev_ma50)

        # Determine direction
        if result.weekly_ma50_slope > self.slope_threshold:
            result.trend_direction = "up"
            result.trend_allowed = True
        elif result.weekly_ma50_slope < -self.slope_threshold:
            result.trend_direction = "down"
            result.trend_allowed = False
        else:
            result.trend_direction = "sideways"
            result.trend_allowed = True

        logger.debug(
            "TrendFilter: %s (MA50=%.2f, slope=%.4f, bars=%d)",
            result.trend_direction, result.weekly_ma50,
            result.weekly_ma50_slope, result.weekly_bars,
        )
        return result

    def apply_filter(
        self,
        signals: list[Signal],
        trend: TrendResult,
    ) -> list[Signal]:
        """Gate buy signals when trend is down.

        In a downtrend, buy signals are converted to hold.
        Sell and hold signals pass through unchanged.
        If insufficient data, pass through unchanged.
        """
        if trend.trend_allowed or trend.weekly_bars < self.weekly_min_bars:
            return signals

        filtered: list[Signal] = []
        overridden_count = 0
        for s in signals:
            if s.action == "buy":
                filtered.append(Signal(
                    timestamp=s.timestamp,
                    symbol=s.symbol,
                    action="hold",
                    confidence=s.confidence * 0.5,
                    price=s.price,
                    metadata={
                        **s.metadata,
                        "trend_override": "bearish_weekly",
                        "original_action": "buy",
                    },
                ))
                overridden_count += 1
            else:
                filtered.append(s)

        if overridden_count:
            logger.info(
                "TrendFilter: overridden %d buy signals (weekly trend: %s)",
                overridden_count, trend.trend_direction,
            )

        return filtered
