"""
Base agent interface for trading signal generation.

All strategy agents should inherit from :class:`BaseAgent` and implement
:meth:`generate_signals`.  Signals are exchanged as ``Signal`` dataclass
instances and serialised to / deserialised from JSON for inter-process
or cross-AI (Manus ↔ Claude) communication.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal schema
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A single trading signal emitted by an agent.

    Attributes:
        timestamp: ISO-format datetime string (e.g. ``"2026-05-14T09:30:00"``).
        symbol:    Ticker symbol, fully qualified (e.g. ``"AAPL"``, ``"2330.TW"``).
        action:    One of ``"buy"``, ``"sell"``, or ``"hold"``.
        confidence: Agent confidence in this signal, in ``[0.0, 1.0]``.
        price:     Reference price at signal time.
        metadata:  Arbitrary extra info (e.g. indicator values, rationale).
    """

    timestamp: str
    symbol: str
    action: str  # "buy" | "sell" | "hold"
    confidence: float  # 0.0 – 1.0
    price: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class that every strategy agent must implement.

    Subclasses only need to override :meth:`generate_signals`; the rest
    (JSON serialisation, logging helpers) is provided.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    # ---- subclasses must implement ---------------------------------------

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Produce a list of trading signals from the input OHLCV data.

        Args:
            data: OHLCV DataFrame with a ``DatetimeIndex`` indexed in
                  ascending order.

        Returns:
            A (possibly empty) list of ``Signal`` instances.
        """
        ...

    # ---- provided helpers ------------------------------------------------

    def signals_to_json(self, signals: list[Signal], indent: int = 2) -> str:
        """Serialize signals to a JSON string."""
        return json.dumps(
            [asdict(s) for s in signals],
            ensure_ascii=False,
            indent=indent,
        )

    @staticmethod
    def signals_from_json(payload: str) -> list[Signal]:
        """Deserialize a JSON string back into ``Signal`` objects."""
        raw = json.loads(payload)
        return [Signal(**item) for item in raw]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
