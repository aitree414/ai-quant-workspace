"""
Value investing agent with DeepSeek API support or fallback rule mode.

Behaviour modes
---------------
*api* (default):
    Every ``analysis_interval`` bars, sends a market summary to the DeepSeek
    API and lets the LLM decide the next action.
*fallback*:
    Rule-based mode — no API call needed.

Fallback rules
--------------
- RSI(14) < 30  AND  10-day return < -5 %   → BUY  (0.80)
- RSI(14) > 70  AND  10-day return > +5 %   → SELL (0.80)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import pandas as pd

from .base_agent import BaseAgent, Signal

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_DEFAULT_VALUE_PROMPT = """\
You are a value investing analyst applying a Graham-inspired framework.
Analyse the price action below and decide: buy, sell, or hold.

Consider:
- Is the stock oversold relative to its recent range?
- Are there signs of capitulation or accumulation?
- Is the risk/reward favourable for a value entry?

Return a JSON object:
  {"action": "buy"|"sell"|"hold", "confidence": 0.0-1.0, "rationale": "..."}
"""


class ValueAgent(BaseAgent):
    """Value investing agent.

    Parameters
    ----------
    name:
        Agent name.
    api_key:
        DeepSeek API key.  Falls back to ``DEEPSEEK_API_KEY`` env var.
    fallback_mode:
        When ``True``, uses rule-based logic instead of the DeepSeek API.
    analysis_interval:
        How many bars between DeepSeek API calls (default 20).
    model:
        DeepSeek model name (default ``"deepseek-chat"``).
    system_prompt:
        Custom system prompt for the LLM.
    """

    def __init__(
        self,
        name: str = "value-agent",
        api_key: Optional[str] = None,
        fallback_mode: bool = False,
        analysis_interval: int = 20,
        model: str = "deepseek-chat",
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.fallback_mode = fallback_mode
        self.analysis_interval = analysis_interval
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_VALUE_PROMPT

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
                    raise ImportError(
                        "openai package required for DeepSeek API — "
                        "run: pip install openai"
                    )
                self._client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

        self._call_count = 0  # tracks bars seen for interval throttling

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate value-oriented trading signals.

        In API mode, calls DeepSeek every ``analysis_interval`` bars.
        In fallback mode, applies rule-based logic on every bar.

        Args:
            data: OHLCV DataFrame with a ``DatetimeIndex``.

        Returns:
            List of ``Signal`` instances.
        """
        if data.empty or "Close" not in data.columns:
            return []

        close = data["Close"]
        symbol = data.attrs.get("symbol", "")

        if self.fallback_mode:
            return self._fallback_signals(close, symbol, data.index)

        return self._api_signals(close, symbol, data.index, data)

    # ------------------------------------------------------------------
    # Fallback (rule-based)
    # ------------------------------------------------------------------

    def _fallback_signals(
        self,
        close: pd.Series,
        symbol: str,
        index: pd.Index,
    ) -> list[Signal]:
        """Rule-based value signals (no API call)."""
        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # 10-day return
        ret_10d = close.pct_change(10)

        signals: list[Signal] = []
        for i, ts in enumerate(index):
            if pd.isna(rsi.iloc[i]) or pd.isna(ret_10d.iloc[i]):
                continue

            price = float(close.iloc[i])

            # BUY: oversold + sharp pullback
            if rsi.iloc[i] < 30 and ret_10d.iloc[i] < -0.05:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=0.80,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "rule": "oversold_pullback",
                            "rsi": round(float(rsi.iloc[i]), 2),
                            "ret_10d_pct": round(float(ret_10d.iloc[i] * 100), 2),
                        },
                    )
                )

            # SELL: overbought + sharp rally
            if rsi.iloc[i] > 70 and ret_10d.iloc[i] > 0.05:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="sell",
                        confidence=0.80,
                        price=price,
                        metadata={
                            "agent": self.name,
                            "mode": "fallback",
                            "rule": "overbought_rally",
                            "rsi": round(float(rsi.iloc[i]), 2),
                            "ret_10d_pct": round(float(ret_10d.iloc[i] * 100), 2),
                        },
                    )
                )

        logger.info(
            "%s [fallback]: %d signal(s) generated", self.name, len(signals)
        )
        return signals

    # ------------------------------------------------------------------
    # DeepSeek API mode
    # ------------------------------------------------------------------

    def _api_signals(
        self,
        close: pd.Series,
        symbol: str,
        index: pd.Index,
        data: pd.DataFrame,
    ) -> list[Signal]:
        """Generate signals via DeepSeek API at a throttled interval."""
        signals: list[Signal] = []

        # Iterate through bars; call API every `analysis_interval` bars
        for i, ts in enumerate(index):
            self._call_count += 1
            if self._call_count % self.analysis_interval != 0:
                continue

            # --- build context from the most recent N bars ---
            lookback = min(self.analysis_interval * 2, i + 1)
            chunk = data.iloc[i - lookback + 1 : i + 1]
            price = float(close.iloc[i])

            context = self._build_api_context(chunk, symbol)

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception as exc:
                logger.error(
                    "%s: DeepSeek API error at %s: %s",
                    self.name, ts, exc,
                )
                continue

            decision = self._parse_api_response(resp.choices[0].message.content or "")
            if decision is None:
                continue

            action, confidence, rationale = decision
            if action == "hold":
                continue

            signals.append(
                Signal(
                    timestamp=str(ts),
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=price,
                    metadata={
                        "agent": self.name,
                        "mode": "deepseek",
                        "rationale": rationale,
                    },
                )
            )
            logger.info(
                "%s [DeepSeek] %s → %s (%.2f) — %s",
                self.name, ts.date(), action.upper(), confidence, rationale,
            )

        logger.info(
            "%s [DeepSeek]: %d signal(s) generated", self.name, len(signals)
        )
        return signals

    def _build_api_context(self, chunk: pd.DataFrame, symbol: str) -> str:
        """Format a recent data slice for the DeepSeek prompt."""
        close = chunk["Close"]
        summary = (
            f"Symbol: {symbol}\n"
            f"Period: {chunk.index[0].strftime('%Y-%m-%d')} → "
            f"{chunk.index[-1].strftime('%Y-%m-%d')}\n"
            f"Bars: {len(chunk)}\n"
            f"Price range: {chunk['Low'].min():.2f} – {chunk['High'].max():.2f}\n"
            f"Current close: {close.iloc[-1]:.2f}\n"
            f"Change over period: {((close.iloc[-1] / close.iloc[0]) - 1) * 100:+.2f}%\n"
            f"Volatility (std): {close.pct_change().std() * 100:.2f}%\n\n"
        )
        # compact OHLC table
        rows = ["Date,Open,High,Low,Close,Volume"]
        for dt, row in chunk.iterrows():
            rows.append(
                f"{dt.strftime('%Y-%m-%d')},"
                f"{row['Open']:.2f},{row['High']:.2f},"
                f"{row['Low']:.2f},{row['Close']:.2f},"
                f"{row.get('Volume', 0):.0f}"
            )
        return summary + "\n".join(rows)

    @staticmethod
    def _parse_api_response(
        text: str,
    ) -> Optional[tuple[str, float, str]]:
        """Extract ``(action, confidence, rationale)`` from DeepSeek output."""
        # Try JSON block first
        m = re.search(
            r"```(?:json)?\s*\n?(\{.*?\})\n?```", text, re.DOTALL | re.IGNORECASE
        )
        if m:
            json_str = m.group(1)
        else:
            # Bare JSON object
            m = re.search(r'(\{.*"action".*\})', text, re.DOTALL)
            if m:
                json_str = m.group(1)
            else:
                return None

        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        action = obj.get("action", "hold")
        confidence = float(obj.get("confidence", 0.5))
        rationale = str(obj.get("rationale", ""))
        return action, confidence, rationale
