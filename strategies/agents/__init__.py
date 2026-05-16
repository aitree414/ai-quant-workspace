from __future__ import annotations

import logging

from .base_agent import BaseAgent, Signal
from .momentum_agent import MomentumAgent
from .value_agent import ValueAgent
from .flow_agent import FlowAgent

logger = logging.getLogger(__name__)

# ClaudeAgent requires anthropic — lazy import
try:
    from .claude_agent import ClaudeAgent  # noqa: F401
except ImportError:
    logger.warning("anthropic not installed; ClaudeAgent unavailable")
    ClaudeAgent = None  # type: ignore

# SentimentAgent may need finnhub or other optional deps
try:
    from .sentiment_agent import SentimentAgent  # noqa: F401
except ImportError:
    SentimentAgent = None  # type: ignore

# MacroAgent may need optional deps
try:
    from .macro_agent import MacroAgent  # noqa: F401
except ImportError:
    MacroAgent = None  # type: ignore

__all__ = [
    "BaseAgent",
    "Signal",
    "ClaudeAgent",
    "MomentumAgent",
    "ValueAgent",
    "SentimentAgent",
    "MacroAgent",
    "FlowAgent",
]
