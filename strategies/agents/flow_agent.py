"""
FlowAgent — detects institutional accumulation/distribution from volume & price action.

Uses On-Balance Volume (OBV), Volume Price Trend (VPT), and volume-spike detection
to infer institutional buying/selling pressure.  Also supports loading real
三大法人買賣超 (TWSE institutional net flow) data when available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base_agent import BaseAgent, Signal

logger = logging.getLogger(__name__)


class FlowAgent(BaseAgent):
    """Volume / institutional-flow analysis agent.

    Parameters
    ----------
    name:
        Agent name (default ``"flow-agent"``).
    obv_period:
        Signal smoother for OBV (default 5).
    volume_surge_threshold:
        Multiple of avg volume to count as a "surge" (default 2.0).
    lookback:
        Bars used for OBV/VPT calculation (default 60).
    """

    def __init__(
        self,
        name: str = "flow-agent",
        obv_period: int = 5,
        volume_surge_threshold: float = 2.0,
        lookback: int = 60,
    ) -> None:
        super().__init__(name)
        self.obv_period = obv_period
        self.volume_surge_threshold = volume_surge_threshold
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        if len(data) < self.lookback + 10:
            return []

        df = data.tail(self.lookback + 10).copy()
        if "Volume" not in df.columns or "Close" not in df.columns:
            return []

        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        # --- OBV (On-Balance Volume) ---
        price_dir = close.diff()
        obv = (volume * price_dir.apply(np.sign)).fillna(0).cumsum()
        obv_sma = obv.rolling(self.obv_period).mean()
        obv_signal = obv - obv_sma  # positive = accumulation

        # --- VPT (Volume Price Trend) ---
        pct_chg = close.pct_change().fillna(0)
        vpt = (pct_chg * volume).cumsum()
        vpt_signal = vpt - vpt.rolling(self.obv_period).mean()

        # --- Volume surge detection ---
        vol_sma = volume.rolling(20).mean().fillna(volume.mean())
        vol_surge = volume / vol_sma

        # --- Combined signals ---
        last = df.iloc[-1]
        last_close = float(last["Close"])
        signals: list[Signal] = []

        # Check for recent signals (last 5 bars)
        for i in range(-5, 0):
            ts = str(df.index[i])
            obv_val = obv_signal.iloc[i]
            vpt_val = vpt_signal.iloc[i]
            surge = vol_surge.iloc[i]

            # Accumulation: OBV rising + VPT rising + possible volume surge
            acc_score = 0
            dist_score = 0

            if obv_val > 0:
                acc_score += 1
            elif obv_val < 0:
                dist_score += 1

            if vpt_val > 0:
                acc_score += 1
            elif vpt_val < 0:
                dist_score += 1

            if surge > self.volume_surge_threshold:
                # Big volume amplifies the signal
                if acc_score > 0:
                    acc_score += 1
                if dist_score > 0:
                    dist_score += 1

            price = float(df.iloc[i]["Close"])

            if acc_score >= 2:
                confidence = min(0.5 + 0.1 * acc_score, 0.90)
                signals.append(Signal(
                    timestamp=ts, symbol=str(data.attrs.get("symbol", "")),
                    action="buy", confidence=round(confidence, 2),
                    price=price,
                    metadata={
                        "obv": round(float(obv_val), 2),
                        "vpt": round(float(vpt_val), 2),
                        "volume_surge": round(float(surge), 2),
                        "reason": "accumulation",
                    },
                ))
            elif dist_score >= 2:
                confidence = min(0.5 + 0.1 * dist_score, 0.90)
                signals.append(Signal(
                    timestamp=ts, symbol=str(data.attrs.get("symbol", "")),
                    action="sell", confidence=round(confidence, 2),
                    price=price,
                    metadata={
                        "obv": round(float(obv_val), 2),
                        "vpt": round(float(vpt_val), 2),
                        "volume_surge": round(float(surge), 2),
                        "reason": "distribution",
                    },
                ))

        return signals
