"""
Sentiment analysis agent — analyses news headlines and market fear/greed.

Modes:
  *api*: Sends recent news headlines to DeepSeek for sentiment scoring.
  *fallback*: Uses price-based volatility proxy as a fear/greed indicator.

The agent emits signals when sentiment is extreme:
  - Very bearish sentiment (fear) on a stock that hasn't crashed → contrarian BUY
  - Very bullish sentiment (greed) on a stock that has rallied → contrarian SELL
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from .base_agent import BaseAgent, Signal

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_SENTIMENT_SYSTEM_PROMPT = """\
You are a financial sentiment analyst. Given a list of recent news headlines
about a stock, score the overall market sentiment.

Return ONLY a JSON object:
{
  "sentiment_score": -1.0 to 1.0,
  "action": "buy" | "sell" | "hold",
  "confidence": 0.0 to 1.0,
  "summary": "one-sentence summary in Traditional Chinese"
}

Rules:
- sentiment_score: -1.0 = extremely bearish, 0 = neutral, +1.0 = extremely bullish
- If sentiment is very negative (fear) but price hasn't crashed, consider contrarian "buy"
- If sentiment is very positive (greed) and price has rallied significantly, consider "sell"
- If sentiment is mixed or neutral, choose "hold"
"""


class SentimentAgent(BaseAgent):
    """News sentiment analysis agent.

    Parameters
    ----------
    name:
        Agent name.
    api_key:
        DeepSeek API key for LLM sentiment analysis.
    fallback_mode:
        When True, uses volatility-based proxy instead of news + LLM.
    analysis_interval:
        How often (in bars) to re-evaluate sentiment (default 10).
    model:
        LLM model name (default ``"deepseek-chat"``).
    """

    def __init__(
        self,
        name: str = "sentiment-agent",
        api_key: Optional[str] = None,
        fallback_mode: bool = False,
        analysis_interval: int = 10,
        model: str = "deepseek-chat",
    ) -> None:
        super().__init__(name)
        self.fallback_mode = fallback_mode
        self.analysis_interval = analysis_interval
        self.model = model

        if not fallback_mode:
            key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not key:
                logger.warning(
                    "%s: DEEPSEEK_API_KEY not set — falling back to rule mode",
                    self.name,
                )
                self.fallback_mode = True
            else:
                if OpenAI is None:
                    raise ImportError("openai package required")
                self._client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate sentiment-based trading signals."""
        if data.empty or "Close" not in data.columns:
            return []

        symbol = data.attrs.get("symbol", "")

        if self.fallback_mode:
            return self._fallback_signals(data, symbol)

        return self._api_signals(data, symbol)

    # ------------------------------------------------------------------
    # Fallback: volatility-based fear/greed proxy
    # ------------------------------------------------------------------

    def _fallback_signals(
        self, data: pd.DataFrame, symbol: str
    ) -> list[Signal]:
        """Use price volatility as a sentiment proxy.

        High volatility + down trend → fear → contrarian BUY
        Low volatility + up trend → complacency → caution SELL
        """
        close = data["Close"]
        signals: list[Signal] = []

        # 10-day realised volatility (annualised)
        returns = close.pct_change()
        vol_10d = returns.rolling(10).std() * np.sqrt(252)

        # 20-day return for trend
        ret_20d = close.pct_change(20)

        # Bollinger Band width as volatility measure
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = (std20 * 2) / sma20  # normalised band width

        for i, ts in enumerate(data.index):
            if i < 20 or pd.isna(vol_10d.iloc[i]) or pd.isna(ret_20d.iloc[i]):
                continue

            # Only check at intervals
            if i % self.analysis_interval != 0:
                continue

            price = float(close.iloc[i])
            v = float(vol_10d.iloc[i])
            r = float(ret_20d.iloc[i])
            bw = float(bb_width.iloc[i]) if not pd.isna(bb_width.iloc[i]) else 0.0

            # Fear: high vol + negative return → contrarian BUY
            if v > 0.35 and r < -0.05:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=min(0.50 + v, 0.85),
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "sentiment": "fear",
                            "vol_10d": round(v, 4),
                            "ret_20d_pct": round(r * 100, 2),
                        },
                    )
                )

            # Greed: low vol + strong positive return → SELL
            elif v < 0.15 and r > 0.10:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="sell",
                        confidence=0.65,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "sentiment": "greed",
                            "vol_10d": round(v, 4),
                            "ret_20d_pct": round(r * 100, 2),
                        },
                    )
                )

        logger.info("%s [fallback]: %d signal(s)", self.name, len(signals))
        return signals

    # ------------------------------------------------------------------
    # API mode: fetch news + LLM sentiment
    # ------------------------------------------------------------------

    def _api_signals(
        self, data: pd.DataFrame, symbol: str
    ) -> list[Signal]:
        """Fetch news via yfinance and score sentiment with LLM."""
        signals: list[Signal] = []
        close = data["Close"]

        # Fetch news headlines
        headlines = self._fetch_news(symbol)
        if not headlines:
            logger.warning("%s: no news found, falling back", self.name)
            return self._fallback_signals(data, symbol)

        # Call LLM once with all headlines
        for i, ts in enumerate(data.index):
            if i % self.analysis_interval != 0 or i < 20:
                continue

            price = float(close.iloc[i])
            ret_20d = float(close.pct_change(20).iloc[i]) if i >= 20 else 0.0

            prompt = (
                f"Stock: {symbol}\n"
                f"Current price: {price:.2f}\n"
                f"20-day return: {ret_20d * 100:+.2f}%\n\n"
                f"Recent news headlines:\n"
            )
            for j, h in enumerate(headlines[:15], 1):
                prompt += f"{j}. {h}\n"

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=256,
                )
                content = resp.choices[0].message.content or ""
                decision = self._parse_response(content)
                if decision and decision["action"] != "hold":
                    signals.append(
                        Signal(
                            timestamp=str(ts),
                            symbol=symbol,
                            action=decision["action"],
                            confidence=decision["confidence"],
                            price=price,
                            metadata={
                                "agent": self.name,
                                "mode": "deepseek",
                                "sentiment_score": decision.get("sentiment_score", 0),
                                "summary": decision.get("summary", ""),
                            },
                        )
                    )
            except Exception as exc:
                logger.error("%s: LLM error: %s", self.name, exc)
                break  # Don't keep retrying

        logger.info("%s [api]: %d signal(s)", self.name, len(signals))
        return signals

    @staticmethod
    def _fetch_news(symbol: str) -> list[str]:
        """Fetch recent news headlines via yfinance."""
        if yf is None:
            return []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            headlines = []
            for item in news:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
            return headlines
        except Exception as exc:
            logger.warning("Failed to fetch news for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _parse_response(text: str) -> Optional[dict]:
        """Parse LLM JSON response."""
        m = re.search(r"```(?:json)?\s*\n?(\{.*?\})\n?```", text, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r'(\{.*"action".*\})', text, re.DOTALL)
            if m:
                json_str = m.group(1)
            else:
                return None
        try:
            obj = json.loads(json_str)
            return {
                "action": obj.get("action", "hold"),
                "confidence": float(obj.get("confidence", 0.5)),
                "sentiment_score": float(obj.get("sentiment_score", 0)),
                "summary": obj.get("summary", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return None
