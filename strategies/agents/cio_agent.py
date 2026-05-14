"""
CIO (Chief Investment Officer) agent.

Receives signals from multiple sub-agents and produces consensus trading
signals through weighted voting.

Default weights
---------------
- MomentumAgent: 0.60
- ValueAgent:    0.40

Decision rule
-------------
For each bar in the data, each agent's signals are converted to a vote:
  +1 (buy)  /  -1 (sell)  /  0 (hold / no signal).

The weighted sum is then thresholded:
  score > +threshold → BUY
  score < -threshold → SELL
  otherwise         → HOLD (no signal emitted)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base_agent import Signal

logger = logging.getLogger(__name__)

# Default weight table: agent name → weight
_DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum-agent": 0.40,
    "value-agent": 0.30,
    "claude-agent": 0.30,
}


class CIOAgent:
    """CIO agent for weighted consensus decision-making.

    Parameters
    ----------
    name:
        Agent name (default ``"cio"``).
    weights:
        Per-agent weights.  Agent names not in the dict get equal weight.
    buy_threshold:
        Weighted score above this value triggers a BUY signal (default 0.25).
    sell_threshold:
        Weighted score below this value triggers a SELL signal (default -0.25).
    """

    def __init__(
        self,
        name: str = "cio",
        weights: Optional[dict[str, float]] = None,
        buy_threshold: float = 0.25,
        sell_threshold: float = -0.25,
    ) -> None:
        self.name = name
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(
        self,
        agent_signals: dict[str, list[Signal]],
        data_index: pd.Index,
        symbol: str = "",
    ) -> list[Signal]:
        """Combine signals from multiple agents into consensus signals.

        Args:
            agent_signals: Mapping of ``agent_name → signals``.
            data_index:     DatetimeIndex to align votes against.
            symbol:         Symbol to attach to output signals.

        Returns:
            Consensus ``Signal`` list (sparse — only when the threshold
            is crossed in a new direction).
        """
        if not agent_signals:
            logger.warning("%s: no agent signals provided", self.name)
            return []

        # --- build vote DataFrame ---
        votes = pd.DataFrame(0.0, index=data_index, columns=list(agent_signals))

        for agent_name, sigs in agent_signals.items():
            if agent_name not in votes.columns:
                continue
            for s in sigs:
                try:
                    ts = pd.Timestamp(s.timestamp)
                    if ts not in votes.index:
                        continue
                    vote_val = 1.0 if s.action == "buy" else (-1.0 if s.action == "sell" else 0.0)
                    votes.at[ts, agent_name] = vote_val
                except (ValueError, TypeError):
                    continue

        # --- weighted sum ---
        weight_sum = sum(self.weights.get(c, 1.0 / len(agent_signals)) for c in votes.columns)
        weighted = pd.Series(0.0, index=data_index)
        for col in votes.columns:
            w = self.weights.get(col, 1.0 / len(agent_signals))
            weighted += w * votes[col]

        if weight_sum != 0:
            weighted /= weight_sum

        # --- state-machine: only emit on direction change ---
        signals: list[Signal] = []
        current_pos: int = 0  # -1 sell, 0 neutral, 1 buy

        for ts in data_index:
            score = weighted.loc[ts]
            price = 0.0  # will be filled by caller if needed

            if score > self.buy_threshold and current_pos != 1:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="buy",
                        confidence=min(score, 1.0),
                        price=price,
                        metadata={
                            "agent": self.name,
                            "consensus_score": round(float(score), 4),
                        },
                    )
                )
                current_pos = 1

            elif score < self.sell_threshold and current_pos != -1:
                signals.append(
                    Signal(
                        timestamp=str(ts),
                        symbol=symbol,
                        action="sell",
                        confidence=min(abs(score), 1.0),
                        price=price,
                        metadata={
                            "agent": self.name,
                            "consensus_score": round(float(score), 4),
                        },
                    )
                )
                current_pos = -1

        logger.info(
            "%s: aggregated %d signals → %d consensus signals",
            self.name,
            sum(len(v) for v in agent_signals.values()),
            len(signals),
        )
        return signals
