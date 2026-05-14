"""
Momentum strategy agent using RSI, MACD, and SMA technical indicators.

Rule-based signal generation:
  - RSI(14) < 30 + MACD golden cross           → BUY  (0.90)
  - RSI(14) > 70 + MACD death cross            → SELL (0.90)
  - Close > SMA(50)                             → BUY  (0.55)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base_agent import BaseAgent, Signal

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """Momentum trading agent driven by technical indicators.

    Parameters
    ----------
    name:
        Agent name (default ``"momentum-agent"``).
    rsi_period:
        Look-back period for RSI (default 14).
    macd_fast / macd_slow / macd_signal:
        MACD parameters (default 12 / 26 / 9).
    sma_period:
        Simple moving average period (default 50).
    """

    def __init__(
        self,
        name: str = "momentum-agent",
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_period: int = 50,
    ) -> None:
        super().__init__(name)
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.sma_period = sma_period

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def _macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        line = fast - slow
        sig = line.ewm(span=self.macd_signal, adjust=False).mean()
        hist = line - sig
        return line, sig, hist

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate momentum-based trading signals.

        Args:
            data: OHLCV DataFrame with a ``DatetimeIndex``.

        Returns:
            List of ``Signal`` instances, one per trigger bar.
        """
        if data.empty or "Close" not in data.columns:
            return []

        close = data["Close"]

        # --- compute indicators ---
        rsi = self._rsi(close)
        macd_line, macd_signal, _ = self._macd(close)
        sma = close.rolling(self.sma_period).mean()

        # --- crossover detection ---
        macd_golden = (macd_line > macd_signal) & (
            macd_line.shift(1) <= macd_signal.shift(1)
        )
        macd_death = (macd_line < macd_signal) & (
            macd_line.shift(1) >= macd_signal.shift(1)
        )

        symbol = data.attrs.get("symbol", "")
        signals: list[Signal] = []

        for i, ts in enumerate(data.index):
            if pd.isna(rsi.iloc[i]) or pd.isna(sma.iloc[i]):
                continue

            price = float(close.iloc[i])

            # --- BUY: RSI oversold + MACD golden cross ---
            if rsi.iloc[i] < 30 and bool(macd_golden.iloc[i]):
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=0.90,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "rule": "rsi_oversold_macd_golden",
                            "rsi": round(float(rsi.iloc[i]), 2),
                        },
                    )
                )

            # --- BUY: price above SMA50 ---
            if close.iloc[i] > sma.iloc[i]:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=0.55,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "rule": "above_sma50",
                            "sma50": round(float(sma.iloc[i]), 2),
                        },
                    )
                )

            # --- SELL: RSI overbought + MACD death cross ---
            if rsi.iloc[i] > 70 and bool(macd_death.iloc[i]):
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="sell",
                        confidence=0.90,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "rule": "rsi_overbought_macd_death",
                            "rsi": round(float(rsi.iloc[i]), 2),
                        },
                    )
                )

        logger.info("%s: %d signal(s) generated", self.name, len(signals))
        return signals
