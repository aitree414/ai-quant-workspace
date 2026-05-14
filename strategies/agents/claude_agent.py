"""
Claude API-powered trading strategy agent.

Uses Anthropic's Claude to analyse OHLCV market data and generate
trading signals based on natural-language reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import anthropic
import pandas as pd

from .base_agent import BaseAgent, Signal

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert quantitative trading analyst.  Your task is to analyse \
OHLCV market data and output trading signals.

**Analysis guidelines**
- Identify clear trends, support/resistance levels, and chart patterns.
- Consider momentum, volatility, and volume as confirming factors.
- Avoid overtrading — only signal when the risk/reward is clearly favourable.
- Prefer trading in the direction of the prevailing trend.
- Manage risk: don't buy at resistance or sell at support without confirmation.

**Output format**
Return a JSON array of signal objects.  Only include signals when you want to \
*change* your position (entries or exits).  Omit "hold" — absence of a signal \
means hold the current position.

Each signal object:
```json
{
  "timestamp": "YYYY-MM-DD",
  "action": "buy" | "sell",
  "confidence": 0.0-1.0,
  "rationale": "Brief reason for this signal"
}
```

- `timestamp` must match one of the dates in the data provided below.
- `confidence` reflects how strongly you feel about this signal (0.0 = unsure, \
1.0 = very confident).
- `rationale` is a short explanation (one sentence).
"""


class ClaudeAgent(BaseAgent):
    """Strategy agent that uses Claude to generate trading signals.

    Parameters
    ----------
    name:
        Agent name (default ``"claude-agent"``).
    model:
        Claude model ID (default ``"claude-opus-4-7"``).
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not provided.
    system_prompt:
        Custom system prompt.  Uses a sensible default when ``None``.
    max_data_rows:
        Maximum number of recent OHLCV rows to include verbatim in the prompt.
        Older data is summarised statistically instead.
    confidence_threshold:
        Minimum confidence to accept a signal.  Signals below this threshold
        are dropped (default ``0.0`` = accept everything).
    max_retries:
        How many times to retry the API call on transient errors.
    """

    def __init__(
        self,
        name: str = "claude-agent",
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_data_rows: int = 100,
        confidence_threshold: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.max_data_rows = max_data_rows
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required — set ANTHROPIC_API_KEY "
                "environment variable or pass api_key=..."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Analyse OHLCV data via Claude and return trading signals.

        Args:
            data: OHLCV DataFrame with a ``DatetimeIndex``.

        Returns:
            A (possibly empty) list of ``Signal`` instances.
        """
        if data.empty:
            logger.warning("%s: empty DataFrame, skipping", self.name)
            return []

        market_context = self._prepare_market_context(data)
        user_prompt = (
            "Analyse the following market data and output trading signals "
            "in the specified JSON format.\n\n"
            f"{market_context}"
        )

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=[{"type": "text", "text": self.system_prompt}],
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except anthropic.APIStatusError as exc:
                logger.error(
                    "%s: API error (attempt %d/%d): %s",
                    self.name, attempt + 1, 1 + self.max_retries, exc,
                )
                last_error = exc
                continue
            except anthropic.APIConnectionError as exc:
                logger.warning(
                    "%s: connection error (attempt %d/%d), retrying…",
                    self.name, attempt + 1, 1 + self.max_retries,
                )
                last_error = exc
                continue

            # Extract text from response (skip thinking blocks)
            text = self._extract_text(response)

            signals = self._parse_response(text, data)
            if signals is not None:
                # Filter by confidence threshold
                filtered = [
                    s for s in signals
                    if s.confidence >= self.confidence_threshold
                ]
                logger.info(
                    "%s: generated %d signals (%d after threshold %.2f)",
                    self.name, len(signals), len(filtered),
                    self.confidence_threshold,
                )
                return filtered

            last_error = RuntimeError("Failed to parse Claude response")
            logger.warning(
                "%s: parse failure (attempt %d/%d), retrying…",
                self.name, attempt + 1, 1 + self.max_retries,
            )

        logger.error(
            "%s: all %d attempts failed — returning empty signals",
            self.name, 1 + self.max_retries,
        )
        raise RuntimeError(
            f"ClaudeAgent failed after {1 + self.max_retries} attempts"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt preparation
    # ------------------------------------------------------------------

    def _prepare_market_context(self, data: pd.DataFrame) -> str:
        """Build a market-context string for the Claude prompt.

        Includes summary statistics for the full period and the most recent
        ``max_data_rows`` rows verbatim.
        """
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data.get("Volume", pd.Series(index=data.index, dtype=float))

        # --- summary statistics ---
        total_return = (close.iloc[-1] / close.iloc[0]) - 1.0
        peak = close.expanding().max()
        drawdown = ((close - peak) / peak).min()

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else None

        summary = (
            f"## Market Summary\n"
            f"- Symbol: {data.attrs.get('symbol', 'N/A')}\n"
            f"- Period: {data.index[0].strftime('%Y-%m-%d')} → "
            f"{data.index[-1].strftime('%Y-%m-%d')}\n"
            f"- Bars: {len(data)}\n"
            f"- Start: {close.iloc[0]:.2f} | End: {close.iloc[-1]:.2f}\n"
            f"- Return: {total_return * 100:+.2f}%\n"
            f"- Max Drawdown: {drawdown * 100:.2f}%\n"
            f"- High: {high.max():.2f} | Low: {low.min():.2f}\n"
            f"- Avg Volume: {volume.mean():.0f}\n"
        )

        if sma50 is not None:
            trend = "bullish" if sma20.iloc[-1] > sma50.iloc[-1] else "bearish"
            summary += (
                f"- SMA20: {sma20.iloc[-1]:.2f} | SMA50: {sma50.iloc[-1]:.2f} "
                f"({trend})\n"
            )

        # --- recent data table ---
        recent = data.tail(self.max_data_rows)
        table_lines = ["\n## Recent OHLCV Data\n", "| Date | Open | High | Low | Close | Volume |"]
        table_lines.append("|------|------|------|------|-------|--------|")
        for dt, row in recent.iterrows():
            table_lines.append(
                f"| {dt.strftime('%Y-%m-%d')} "
                f"| {row['Open']:.2f} "
                f"| {row['High']:.2f} "
                f"| {row['Low']:.2f} "
                f"| {row['Close']:.2f} "
                f"| {row.get('Volume', 0):.0f} |"
            )

        return summary + "\n".join(table_lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        """Pull the text content out of a Claude response, ignoring thinking blocks."""
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _parse_response(
        self,
        text: str,
        data: pd.DataFrame,
    ) -> Optional[list[Signal]]:
        """Try to extract a JSON signal array from Claude's text response.

        Returns ``None`` when parsing fails (triggers a retry).
        """
        # Try to find a JSON array in a code block first
        json_str = self._extract_json_block(text)
        if json_str is None:
            logger.debug("%s: no JSON block found in response", self.name)
            return None

        try:
            raw_list = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("%s: invalid JSON: %s", self.name, exc)
            return None

        if not isinstance(raw_list, list):
            logger.warning("%s: JSON value is not an array", self.name)
            return None

        # Build a set of valid timestamps from the index for fuzzy matching
        valid_dates = {d.strftime("%Y-%m-%d") for d in data.index}

        signals: list[Signal] = []
        for item in raw_list:
            try:
                ts = item.get("timestamp", "")
                action = str(item.get("action", "hold")).lower()
                confidence = float(item.get("confidence", 0.5))

                # Normalise and validate timestamp
                ts_normalised = ts[:10] if ts else ""
                if ts_normalised not in valid_dates:
                    logger.debug(
                        "%s: skipping signal with unknown date %s", self.name, ts
                    )
                    continue

                # Map action
                if action not in ("buy", "sell", "hold"):
                    logger.debug(
                        "%s: skipping signal with invalid action %s",
                        self.name, action,
                    )
                    continue

                # Look up price at this timestamp
                price = float(data.loc[ts_normalised, "Close"])

                metadata: dict = {}
                if "rationale" in item:
                    metadata["rationale"] = str(item["rationale"])

                signals.append(
                    Signal(
                        timestamp=ts_normalised,
                        symbol=data.attrs.get("symbol", ""),
                        action=action,
                        confidence=confidence,
                        price=price,
                        metadata=metadata,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("%s: skipping malformed signal: %s", self.name, exc)
                continue

        return signals

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Extract JSON from a markdown code block, or failing that from
        the raw text."""
        # Pattern 1: ```json ... ```
        m = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
        )
        if m:
            return m.group(1).strip()

        # Pattern 2: standalone [...] array at top level
        m = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        return None

    # ------------------------------------------------------------------
    # Position conversion (agent output → backtest engine input)
    # ------------------------------------------------------------------

    @staticmethod
    def signals_to_position(
        signals: list[Signal],
        index: pd.Index,
    ) -> pd.Series:
        """Convert a list of buy/sell Signals to a boolean position Series.

        The returned Series is ``True`` (in position) from each ``buy``
        signal until the next ``sell`` signal.  ``hold`` signals are ignored.

        Args:
            signals: Signals sorted chronologically (or will be sorted here).
            index: The target DatetimeIndex (typically from the OHLCV data).

        Returns:
            A boolean ``pd.Series`` over ``index``, suitable for passing to
            :meth:`BacktestEngine.run`.
        """
        position = pd.Series(False, index=index)

        if not signals:
            return position

        # Sort by timestamp
        sorted_sigs = sorted(signals, key=lambda s: s.timestamp)

        in_position = False
        for sig in sorted_sigs:
            if sig.action == "buy" and not in_position:
                # Find the index location of this timestamp
                mask = index >= sig.timestamp
                if mask.any():
                    position[mask] = True
                    in_position = True
            elif sig.action == "sell" and in_position:
                mask = index >= sig.timestamp
                if mask.any():
                    position[mask] = False
                    in_position = False

        return position
