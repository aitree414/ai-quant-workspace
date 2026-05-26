"""
CIO (Chief Investment Officer) agent.

Receives signals from multiple sub-agents and produces consensus trading
signals through weighted voting.

Default weights
---------------
- MomentumAgent: 0.40
- ValueAgent:    0.30
- ClaudeAgent:   0.30

Decision rule
-------------
For each bar in the data, each agent's signals are converted to a vote:
  +1 (buy)  /  -1 (sell)  /  0 (hold / no signal).

The weighted sum is then thresholded:
  score > +threshold → BUY
  score < -threshold → SELL
  otherwise         → HOLD (no signal emitted)

Dynamic weights
---------------
WeightTracker tracks each agent's historical signal outcomes and
adjusts weights based on recent performance.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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

WEIGHT_HISTORY_FILE = Path("data/agent_weights_history.json")


# ---------------------------------------------------------------------------
# WeightTracker — dynamic weight adjustment based on agent performance
# ---------------------------------------------------------------------------


@dataclass
class AgentPerformanceSnapshot:
    """Performance metrics for a single agent over the rolling window."""
    agent_name: str
    window_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_forward_return: float
    current_weight: float


class WeightTracker:
    """Tracks agent signal outcomes and adjusts weights dynamically.

    Uses a rolling window with exponential decay so older signals matter less.
    """

    def __init__(
        self,
        window_days: int = 90,
        decay_factor: float = 0.95,
        min_signals_for_adjustment: int = 5,
        learning_rate: float = 0.1,
        history_file: str | Path = WEIGHT_HISTORY_FILE,
    ) -> None:
        self.window_days = window_days
        self.decay_factor = decay_factor
        self.min_signals = min_signals_for_adjustment
        self.learning_rate = learning_rate
        self.history_file = Path(history_file)
        self._history: dict[str, list[dict]] = self.load_history()

    def record_signal_outcome(
        self,
        agent_name: str,
        signal_action: str,
        signal_timestamp: str,
        forward_return: float,
        confidence: float,
    ) -> None:
        """Record whether a signal was correct.

        For buy signals: positive forward_return = win.
        For sell signals: negative forward_return = win.
        """
        is_win = False
        if signal_action == "buy":
            is_win = forward_return > 0
        elif signal_action == "sell":
            is_win = forward_return < 0

        record = {
            "timestamp": signal_timestamp,
            "action": signal_action,
            "forward_return": round(forward_return, 4),
            "confidence": round(confidence, 4),
            "is_win": is_win,
            "recorded_at": datetime.now().isoformat(),
        }

        if agent_name not in self._history:
            self._history[agent_name] = []
        self._history[agent_name].append(record)
        self.save_history()

    def get_performance(self, agent_name: str) -> AgentPerformanceSnapshot:
        """Compute weighted performance metrics over rolling window."""
        records = self._get_recent_records(agent_name)

        if not records:
            return AgentPerformanceSnapshot(
                agent_name=agent_name,
                window_signals=0, wins=0, losses=0,
                win_rate=0.0, avg_forward_return=0.0,
                current_weight=0.0,
            )

        # Apply exponential decay: more recent = higher weight
        total_weight = 0.0
        weighted_wins = 0.0
        weighted_return = 0.0

        for i, r in enumerate(reversed(records)):
            w = self.decay_factor ** i
            total_weight += w
            if r["is_win"]:
                weighted_wins += w
            weighted_return += r["forward_return"] * w

        win_rate = weighted_wins / total_weight if total_weight > 0 else 0
        avg_return = weighted_return / total_weight if total_weight > 0 else 0

        return AgentPerformanceSnapshot(
            agent_name=agent_name,
            window_signals=len(records),
            wins=sum(1 for r in records if r["is_win"]),
            losses=sum(1 for r in records if not r["is_win"]),
            win_rate=round(win_rate, 4),
            avg_forward_return=round(avg_return, 4),
            current_weight=0.0,
        )

    def update_weights(self, current_weights: dict[str, float]) -> dict[str, float]:
        """Adjust weights based on recent agent performance.

        Formula for each agent i:
            score_i = win_rate_i * max(0, 1 + avg_return_i)
            w_i_new = w_i_base * (1 + learning_rate * (score_i / avg_score - 1))

        Weights are then normalized to sum to 1.0.
        Agents with < min_signals_for_adjustment keep current weight.
        """
        new_weights = dict(current_weights)
        performances: dict[str, AgentPerformanceSnapshot] = {}
        scores: dict[str, float] = {}

        # Compute performance scores for all agents with enough data
        for name in current_weights:
            perf = self.get_performance(name)
            performances[name] = perf
            if perf.window_signals >= self.min_signals:
                score = perf.win_rate * max(0.0, 1.0 + perf.avg_forward_return)
                scores[name] = score
            else:
                scores[name] = 0.0

        avg_score = sum(scores.values()) / len(scores) if scores else 0

        if avg_score <= 0:
            logger.info("WeightTracker: avg_score <= 0, keeping current weights")
            return new_weights

        # Adjust weights
        for name in current_weights:
            perf = performances[name]
            if perf.window_signals >= self.min_signals and scores.get(name, 0) > 0:
                score_ratio = scores[name] / avg_score
                adjustment = 1.0 + self.learning_rate * (score_ratio - 1.0)
                new_weights[name] = current_weights[name] * adjustment
                logger.info(
                    "WeightTracker: %s score=%.4f ratio=%.2f → weight %.4f (was %.4f)",
                    name, scores[name], score_ratio, new_weights[name], current_weights[name],
                )
            else:
                logger.info(
                    "WeightTracker: %s insufficient data (%d < %d), weight unchanged",
                    name, perf.window_signals, self.min_signals,
                )

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            for name in new_weights:
                new_weights[name] /= total

        logger.info("WeightTracker: new weights = %s", new_weights)
        return new_weights

    def _get_recent_records(self, agent_name: str) -> list[dict]:
        """Get records within the rolling window."""
        records = self._history.get(agent_name, [])
        if not records:
            return []

        cutoff = (
            datetime.now().timestamp() - self.window_days * 86400
        )
        recent = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["timestamp"]).timestamp()
                if ts >= cutoff:
                    recent.append(r)
            except (ValueError, KeyError):
                recent.append(r)
        return recent

    def load_history(self) -> dict[str, list[dict]]:
        """Load recorded outcomes from JSON file."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                logger.info("WeightTracker: loaded history for %d agents", len(data))
                return data
            except Exception as e:
                logger.warning("WeightTracker: failed to load history: %s", e)
        return {}

    def save_history(self) -> None:
        """Write history to JSON file atomically."""
        tmp = str(self.history_file) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2, default=str)
            os.replace(str(tmp), str(self.history_file))
        except Exception as e:
            logger.warning("WeightTracker: failed to save history: %s", e)


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
        use_dynamic_weights: bool = False,
        weight_tracker: Optional[WeightTracker] = None,
    ) -> None:
        self.name = name
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_dynamic_weights = use_dynamic_weights
        self.weight_tracker = weight_tracker or WeightTracker()

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

        # --- dynamic weight adjustment ---
        if self.use_dynamic_weights:
            old_weights = dict(self.weights)
            self.weights = self.weight_tracker.update_weights(self.weights)
            for name, old_w in old_weights.items():
                new_w = self.weights.get(name, old_w)
                if abs(new_w - old_w) > 0.001:
                    logger.info(
                        "  %s weight adjusted: %.3f → %.3f (Δ%+.3f)",
                        name, old_w, new_w, new_w - old_w,
                    )

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
