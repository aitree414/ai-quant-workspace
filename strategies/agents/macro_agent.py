"""
Macro-economic agent — analyses broad market conditions to adjust risk.

Uses market-wide indicators as proxies for macro conditions:
  - VIX (CBOE Volatility Index) for market fear
  - S&P 500 trend for overall market health
  - US Treasury yields (via TLT ETF) for interest rate environment
  - Gold (GLD) for safe-haven demand

Modes:
  *api*: Sends macro summary to DeepSeek for analysis.
  *fallback*: Rule-based macro regime detection.

Macro regimes:
  - RISK_ON:  VIX low, SPY trending up → BUY signal
  - RISK_OFF: VIX high, SPY trending down → SELL signal
  - NEUTRAL:  Mixed signals → HOLD
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

# Macro proxy tickers
MACRO_TICKERS = {
    "vix": "^VIX",       # CBOE Volatility Index
    "spy": "SPY",        # S&P 500 ETF
    "tlt": "TLT",        # 20+ Year Treasury Bond ETF
    "gld": "GLD",        # Gold ETF
    "dxy": "DX-Y.NYB",   # US Dollar Index
}

_MACRO_SYSTEM_PROMPT = """\
You are a macro-economic analyst for an investment committee.
Given current macro indicators, assess the overall market regime.

Return ONLY a JSON object:
{
  "regime": "risk_on" | "risk_off" | "neutral",
  "action": "buy" | "sell" | "hold",
  "confidence": 0.0 to 1.0,
  "analysis": "brief analysis in Traditional Chinese"
}

Rules:
- risk_on: Low VIX, SPY uptrend, stable bonds → favour equities → "buy"
- risk_off: High VIX, SPY downtrend, flight to safety → reduce exposure → "sell"
- neutral: Mixed signals → "hold"
"""


class MacroAgent(BaseAgent):
    """Macro-economic regime detection agent.

    Parameters
    ----------
    name:
        Agent name.
    api_key:
        DeepSeek API key for LLM analysis.
    fallback_mode:
        When True, uses rule-based regime detection.
    analysis_interval:
        How often (in bars) to re-evaluate macro conditions (default 20).
    model:
        LLM model name.
    """

    def __init__(
        self,
        name: str = "macro-agent",
        api_key: Optional[str] = None,
        fallback_mode: bool = False,
        analysis_interval: int = 20,
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

        self._macro_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate macro-regime-based signals."""
        if data.empty or "Close" not in data.columns:
            return []

        symbol = data.attrs.get("symbol", "")
        start = str(data.index[0].date())
        end = str(data.index[-1].date())

        # Load macro data for the same period
        macro_data = self._load_macro_data(start, end)

        if self.fallback_mode:
            return self._fallback_signals(data, macro_data, symbol)
        return self._api_signals(data, macro_data, symbol)

    # ------------------------------------------------------------------
    # Macro data loading
    # ------------------------------------------------------------------

    def _load_macro_data(self, start: str, end: str) -> dict[str, pd.DataFrame]:
        """Fetch macro proxy data via yfinance."""
        if self._macro_cache is not None:
            return self._macro_cache

        result: dict[str, pd.DataFrame] = {}
        if yf is None:
            logger.warning("yfinance not available for macro data")
            return result

        for name, ticker in MACRO_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, interval="1d")
                if not df.empty:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    result[name] = df
                    logger.info("Loaded macro data: %s (%d bars)", name, len(df))
            except Exception as exc:
                logger.warning("Failed to load %s (%s): %s", name, ticker, exc)

        self._macro_cache = result
        return result

    # ------------------------------------------------------------------
    # Fallback: rule-based regime detection
    # ------------------------------------------------------------------

    def _fallback_signals(
        self,
        data: pd.DataFrame,
        macro_data: dict[str, pd.DataFrame],
        symbol: str,
    ) -> list[Signal]:
        """Rule-based macro regime detection."""
        close = data["Close"]
        signals: list[Signal] = []

        # Get VIX and SPY data
        vix_data = macro_data.get("vix")
        spy_data = macro_data.get("spy")

        for i, ts in enumerate(data.index):
            if i < 50 or i % self.analysis_interval != 0:
                continue

            price = float(close.iloc[i])

            # Determine macro regime
            regime = "neutral"
            confidence = 0.50

            # VIX analysis
            vix_level = None
            if vix_data is not None and "Close" in vix_data.columns:
                # Find closest VIX date
                vix_close = vix_data["Close"]
                mask = vix_data.index <= ts
                if mask.any():
                    vix_level = float(vix_close[mask].iloc[-1])

            # SPY trend
            spy_trend = None
            if spy_data is not None and "Close" in spy_data.columns:
                spy_close = spy_data["Close"]
                mask = spy_data.index <= ts
                if mask.any() and mask.sum() >= 50:
                    recent_spy = spy_close[mask]
                    spy_sma50 = recent_spy.rolling(50).mean().iloc[-1]
                    spy_current = recent_spy.iloc[-1]
                    spy_trend = "up" if spy_current > spy_sma50 else "down"

            # Regime classification
            if vix_level is not None and spy_trend is not None:
                if vix_level < 18 and spy_trend == "up":
                    regime = "risk_on"
                    confidence = 0.70
                elif vix_level > 25 and spy_trend == "down":
                    regime = "risk_off"
                    confidence = 0.75
                elif vix_level > 30:
                    regime = "risk_off"
                    confidence = 0.80
                elif vix_level < 15 and spy_trend == "up":
                    regime = "risk_on"
                    confidence = 0.65
            elif vix_level is not None:
                if vix_level > 28:
                    regime = "risk_off"
                    confidence = 0.65
                elif vix_level < 16:
                    regime = "risk_on"
                    confidence = 0.60

            # Emit signal based on regime
            if regime == "risk_on":
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=confidence,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "regime": regime,
                            "vix": vix_level,
                            "spy_trend": spy_trend,
                        },
                    )
                )
            elif regime == "risk_off":
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="sell",
                        confidence=confidence,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "regime": regime,
                            "vix": vix_level,
                            "spy_trend": spy_trend,
                        },
                    )
                )

        logger.info("%s [fallback]: %d signal(s)", self.name, len(signals))
        return signals

    # ------------------------------------------------------------------
    # API mode: LLM macro analysis
    # ------------------------------------------------------------------

    def _api_signals(
        self,
        data: pd.DataFrame,
        macro_data: dict[str, pd.DataFrame],
        symbol: str,
    ) -> list[Signal]:
        """Use LLM to analyse macro conditions."""
        close = data["Close"]
        signals: list[Signal] = []

        for i, ts in enumerate(data.index):
            if i < 50 or i % self.analysis_interval != 0:
                continue

            price = float(close.iloc[i])

            # Build macro summary
            summary = self._build_macro_summary(ts, macro_data)
            if not summary:
                continue

            prompt = (
                f"Target stock: {symbol} (price: {price:.2f})\n\n"
                f"Macro indicators as of {ts.date()}:\n{summary}\n\n"
                f"Based on these macro conditions, should we be risk-on or risk-off?"
            )

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _MACRO_SYSTEM_PROMPT},
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
                                "regime": decision.get("regime", "unknown"),
                                "analysis": decision.get("analysis", ""),
                            },
                        )
                    )
            except Exception as exc:
                logger.error("%s: LLM error: %s", self.name, exc)
                break

        logger.info("%s [api]: %d signal(s)", self.name, len(signals))
        return signals

    def _build_macro_summary(
        self, ts: pd.Timestamp, macro_data: dict[str, pd.DataFrame]
    ) -> str:
        """Build a text summary of macro indicators at a given timestamp."""
        lines = []
        for name, df in macro_data.items():
            if "Close" not in df.columns:
                continue
            mask = df.index <= ts
            if not mask.any():
                continue
            current = float(df["Close"][mask].iloc[-1])
            if mask.sum() >= 20:
                sma20 = float(df["Close"][mask].rolling(20).mean().iloc[-1])
                ret_20d = float(df["Close"][mask].pct_change(20).iloc[-1] * 100)
                lines.append(
                    f"- {name.upper()}: {current:.2f} "
                    f"(SMA20: {sma20:.2f}, 20d change: {ret_20d:+.1f}%)"
                )
            else:
                lines.append(f"- {name.upper()}: {current:.2f}")

        return "\n".join(lines)

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
                "regime": obj.get("regime", "neutral"),
                "action": obj.get("action", "hold"),
                "confidence": float(obj.get("confidence", 0.5)),
                "analysis": obj.get("analysis", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return None
