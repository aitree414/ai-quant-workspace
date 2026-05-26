"""Options Committee — multi-agent voting coordinator.

Runs 5 committee members on each symbol, aggregates votes,
applies majority rules and risk manager veto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .base_agent import OptionsSignal
from .fundamental_agent import FundamentalAgent
from .risk_manager import RiskManager
from .technical_agent import TechnicalAgent
from .volatility_agent import VolatilityAgent
from .flow_agent import OptionsFlowAgent
from .options_utils import get_option_chain, get_iv_history

logger = logging.getLogger(__name__)

# Default symbol pool
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT"]

# Decision thresholds
REQUIRED_APPROVALS = 4      # 4/5 majority
MIN_CONFIDENCE = 0.30        # Minimum confidence for a "yes" vote


@dataclass
class CommitteeVote:
    """Aggregated committee decision on one symbol."""

    symbol: str
    underlying_price: float
    recommended_strategy: str  # The winning strategy
    approved: bool             # 4/5 majority met?
    approved_count: int
    total_count: int
    vetoed: bool               # Risk manager vetoed?
    avg_confidence: float
    signals: list[OptionsSignal] = field(default_factory=list)
    timestamp: str = ""
    contract_count: int = 1    # Kelly-sized contract count (default 1 for backward compat)


class OptionsCommittee:
    """AI Committee for options trading decisions."""

    def __init__(self, symbols: Optional[list[str]] = None):
        self.symbols = symbols or DEFAULT_SYMBOLS.copy()
        self.members: list = [
            FundamentalAgent(),
            TechnicalAgent(),
            VolatilityAgent(),
            OptionsFlowAgent(),
            RiskManager(),
        ]
        self.results: dict[str, CommitteeVote] = {}

    def analyze_symbol(self, symbol: str, portfolio: Optional[dict] = None) -> Optional[CommitteeVote]:
        """Run all agents on one symbol and aggregate votes."""
        # Fetch data
        chain = get_option_chain(symbol)
        if "error" in chain:
            logger.warning("  Skipping %s: %s", symbol, chain.get("error"))
            return None

        underlying_price = chain.get("price", 0)
        iv_data = get_iv_history(symbol) if underlying_price else {}

        signals: list[OptionsSignal] = []
        for agent in self.members:
            try:
                signal = agent.analyze(
                    symbol=symbol,
                    underlying_price=underlying_price,
                    option_chain=chain,
                    hist_data={},
                    iv_data=iv_data,
                    portfolio=portfolio,
                )
                signals.append(signal)
            except Exception as e:
                logger.warning("  %s error on %s: %s", agent.name, symbol, e)
                signals.append(OptionsSignal(
                    symbol=symbol, strategy="none", action="hold",
                    confidence=0.0, reasoning=f"分析異常: {e}",
                ))

        # --- Voting ---
        # Count non-risk-manager votes
        strategy_votes: dict[str, int] = {}
        total_confidence = 0.0
        vote_count = 0
        risk_manager_signal = None

        for s in signals:
            if s.confidence >= MIN_CONFIDENCE and s.strategy != "none":
                strategy_votes[s.strategy] = strategy_votes.get(s.strategy, 0) + 1
                total_confidence += s.confidence
                vote_count += 1
            if s.confidence >= MIN_CONFIDENCE and s.action in ("enter",):
                total_confidence += s.confidence * 0.5

            # Track risk manager separately
            if isinstance(self.members[signals.index(s) if signals else 0], RiskManager) or s.reasoning.startswith("風控"):
                risk_manager_signal = s

        # Find risk manager signal
        for s in signals:
            if s.details.get("veto") is not None:
                risk_manager_signal = s
                break

        # Determine winning strategy
        recommended = "none"
        if strategy_votes:
            recommended = max(strategy_votes, key=strategy_votes.get)
            approved_count = strategy_votes.get(recommended, 0)
        else:
            approved_count = 0

        # Check risk manager veto
        vetoed = False
        if risk_manager_signal and risk_manager_signal.confidence < MIN_CONFIDENCE:
            vetoed = True

        # 4/5 majority (excluding risk manager if it vetoed)
        # Risk manager has explicit veto power
        if vetoed:
            approved = False
        else:
            approved = approved_count >= REQUIRED_APPROVALS

        avg_conf = total_confidence / max(vote_count, 1)

        vote = CommitteeVote(
            symbol=symbol,
            underlying_price=underlying_price,
            recommended_strategy=recommended,
            approved=approved,
            approved_count=approved_count,
            total_count=len([s for s in signals if s.confidence >= MIN_CONFIDENCE]),
            vetoed=vetoed,
            avg_confidence=round(avg_conf, 2),
            signals=signals,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        self.results[symbol] = vote
        return vote

    def run_analysis(self, portfolio: Optional[dict] = None) -> dict[str, CommitteeVote]:
        """Run committee analysis on all symbols."""
        logger.info("  🤖 AI Committee 期權分析啟動…")
        logger.info(f"  標的: {', '.join(self.symbols)}")
        logger.info(f"  委員: {', '.join(m.name for m in self.members)}\n")

        for symbol in self.symbols:
            try:
                result = self.analyze_symbol(symbol, portfolio)
                if result:
                    status = "✅ 通過" if result.approved else "❌ 未通過"
                    if result.vetoed:
                        status += " (風控否決)"
                    logger.info(f"  {symbol} (${result.underlying_price}): {status}")
                    logger.info(f"    策略: {result.recommended_strategy}")
                    logger.info(f"    票數: {result.approved_count}/{result.total_count}")
            except Exception as e:
                logger.warning(f"  {symbol} 分析失敗: {e}")

        return self.results

    def get_voting_summary(self) -> list[dict]:
        """Return human-readable voting summary."""
        summary = []
        for sym, vote in self.results.items():
            member_votes = []
            for s in vote.signals:
                member_votes.append({
                    "name": self.members[len(member_votes)].name if len(member_votes) < len(self.members) else "unknown",
                    "strategy": s.strategy,
                    "action": s.action,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                })
            summary.append({
                "symbol": sym,
                "price": vote.underlying_price,
                "strategy": vote.recommended_strategy,
                "approved": vote.approved,
                "vetoed": vote.vetoed,
                "avg_confidence": vote.avg_confidence,
                "member_votes": member_votes,
                "timestamp": vote.timestamp,
            })
        return summary
