"""
Portfolio Coordinator — integrates stock and options committee decisions.

Applies coordination rules:
  - Covered Call: stock long + sell OTM call
  - Protective Put: stock long + buy ATM put
  - Cash-Secured Put: options only (no stock needed)
  - Coordinated exit: sell stock + close related options
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoordinatedAction:
    """A coordinated trading action combining stock and options decisions."""
    symbol: str
    timestamp: str
    stock_action: str           # "buy" | "sell" | "hold" | ""
    stock_confidence: float
    stock_qty: int
    options_action: str         # "enter" | "exit" | "hold" | ""
    options_strategy: str       # "covered_call" | "cash_secured_put" | etc.
    options_contracts: int
    combined_reasoning: str
    approved: bool
    risk_warnings: list[str] = field(default_factory=list)


class PortfolioCoordinator:
    """Coordinates decisions between stock and options committees.

    Sits above both committees and applies coordination rules.
    """

    def __init__(
        self,
        portfolio: Any = None,  # PaperPortfolio reference for stock holdings
        options_portfolio: Any = None,  # OptionsPortfolio reference
    ) -> None:
        self._portfolio = portfolio
        self._options_portfolio = options_portfolio
        self._action_log: list[dict] = []

    def coordinate(
        self,
        stock_signals: dict[str, dict],
        options_votes: dict[str, Any],
        capital: float,
    ) -> list[CoordinatedAction]:
        """Run coordination rules and return approved actions.

        Args:
            stock_signals: ticker -> {action, confidence, price, ...}
            options_votes: ticker -> CommitteeVote objects
            capital: Total capital for position sizing checks.

        Returns:
            List of CoordinatedAction with combined decisions.
        """
        actions: list[CoordinatedAction] = []
        all_symbols = set(stock_signals.keys()) | set(options_votes.keys())

        for symbol in sorted(all_symbols):
            stock = stock_signals.get(symbol, {})
            vote = options_votes.get(symbol, None)

            stock_action = stock.get("action", "hold")
            stock_conf = stock.get("confidence", 0.0)
            stock_price = stock.get("price", 0.0)

            opt_action = ""
            opt_strategy = ""
            opt_contracts = 0
            if vote:
                opt_action = "enter" if vote.approved and not vote.vetoed else "exit" if not vote.approved else "hold"
                opt_strategy = vote.recommended_strategy if vote.approved and not vote.vetoed else "none"
                opt_contracts = getattr(vote, "contract_count", 1)

            action = self._apply_rules(
                symbol, stock_action, stock_conf, stock_price,
                opt_action, opt_strategy, opt_contracts,
                vote, capital,
            )
            actions.append(action)
            self._action_log.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "result": action.combined_reasoning,
                "approved": action.approved,
            })

        return actions

    def _apply_rules(
        self,
        symbol: str,
        stock_action: str,
        stock_conf: float,
        stock_price: float,
        opt_action: str,
        opt_strategy: str,
        opt_contracts: int,
        vote: Any,
        capital: float,
    ) -> CoordinatedAction:
        """Apply coordination rules for one symbol."""
        stock_qty = 0
        reasoning_parts: list[str] = []
        risk_warnings: list[str] = []

        # Check stock holdings
        holds_stock = False
        if self._portfolio:
            holdings = getattr(self._portfolio, "holdings", {})
            holds_stock = symbol in holdings
            if holds_stock:
                stock_qty = holdings[symbol].get("qty", 0)

        # Rule 1: Covered Call — stock long + sell call
        if stock_action == "buy" and opt_strategy == "covered_call" and vote and vote.approved:
            stock_qty = self._estimate_qty(capital, stock_price)
            reasoning_parts.append(f"CoveredCall: buy {stock_qty}sh + sell {opt_contracts} calls")
            approved = True
            exit_opt_action = "enter"

        # Rule 2: Protective Put — already holding stock, buy put
        elif holds_stock and opt_strategy == "protective_put" and vote and vote.approved:
            reasoning_parts.append(f"ProtectivePut: hold stock + buy {opt_contracts} puts")
            approved = True
            exit_opt_action = "enter"

        # Rule 3: Cash-Secured Put — options only
        elif opt_strategy == "cash_secured_put" and vote and vote.approved:
            reasoning_parts.append(f"CashSecuredPut: sell {opt_contracts} puts (no stock)")
            approved = True
            exit_opt_action = "enter"

        # Rule 4: Coordinated exit
        elif stock_action == "sell" and holds_stock:
            reasoning_parts.append("Coordinated exit: sell stock")
            approved = True
            exit_opt_action = "exit"

        # Rule 5: Independent stock action
        elif stock_action == "buy" and stock_conf >= 0.3:
            stock_qty = self._estimate_qty(capital, stock_price)
            reasoning_parts.append(f"Stock only: buy {stock_qty}sh")
            approved = True
            exit_opt_action = "hold"

        # Rule 6: Independent options action (not covered_call without stock)
        elif opt_action == "enter" and vote and vote.approved:
            if opt_strategy == "covered_call" and not holds_stock:
                reasoning_parts.append(f"Options only: {opt_strategy} (no stock to cover)")
                approved = True
                exit_opt_action = "enter"
            else:
                reasoning_parts.append(f"Options only: {opt_strategy} {opt_contracts}ct")
                approved = True
                exit_opt_action = "enter"

        # Rule 7: Vetoed by risk manager
        elif vote and vote.vetoed:
            reasoning_parts.append("Risk manager veto — no action")
            approved = False
            exit_opt_action = "hold"
            risk_warnings.append(f"Risk manager vetoed {symbol}")

        # Default: hold
        else:
            reasoning_parts.append("No signal — hold")
            approved = False
            exit_opt_action = "hold"

        # Check budget for stock buys
        if stock_action == "buy" and stock_qty > 0 and self._portfolio:
            cash_available = getattr(self._portfolio, "cash", capital)
            cost = stock_qty * stock_price
            if cost > cash_available:
                max_qty = int(cash_available / stock_price / 1000) * 1000
                stock_qty = max(max_qty, 0)
                reasoning_parts.append(f"(budget limited to {stock_qty}sh)")
                if stock_qty == 0:
                    reasoning_parts.append("insufficient cash")
                    if exit_opt_action in ("", "hold"):
                        approved = False

        return CoordinatedAction(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            stock_action=stock_action if approved else "hold",
            stock_confidence=stock_conf,
            stock_qty=stock_qty,
            options_action=exit_opt_action,
            options_strategy=opt_strategy,
            options_contracts=opt_contracts,
            combined_reasoning="; ".join(reasoning_parts),
            approved=approved,
            risk_warnings=risk_warnings,
        )

    def get_combined_pnl(self) -> dict:
        """Calculate combined P&L across stock and options portfolios."""
        stock_equity = 0.0
        options_equity = 0.0
        stock_capital = 0.0
        options_capital = 0.0

        if self._portfolio:
            stock_equity = getattr(self._portfolio, "total_equity", 0)
            stock_capital = getattr(self._portfolio, "initial", 0)

        if self._options_portfolio:
            opt_eq = getattr(self._options_portfolio, "total_equity", 0)
            options_equity = opt_eq() if callable(opt_eq) else opt_eq
            options_capital = getattr(self._options_portfolio, "initial", 0)

        if math.isnan(stock_equity):
            stock_equity = 0.0
        if math.isnan(options_equity):
            options_equity = 0.0

        total_equity = stock_equity + options_equity
        total_capital = (stock_capital or 0) + (options_capital or 0)
        total_pnl = total_equity - total_capital
        total_return = (total_pnl / total_capital * 100) if total_capital > 0 else 0.0

        return {
            "total_equity": round(total_equity, 2),
            "stock_equity": round(stock_equity, 2),
            "options_equity": round(options_equity, 2),
            "total_capital": round(total_capital, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "stock_pnl": round(stock_equity - stock_capital, 2),
            "options_pnl": round(options_equity - options_capital, 2),
        }

    def get_action_log(self, count: int = 50) -> list[dict]:
        """Return recent action log entries."""
        return self._action_log[-count:]

    def _estimate_qty(self, capital: float, price: float, max_pct: float = 0.20) -> int:
        """Estimate stock quantity for a position using simple % of capital."""
        if price <= 0:
            return 0
        dollar_amount = capital * max_pct
        raw_qty = int(dollar_amount / price / 1000) * 1000
        return max(raw_qty, 0)
