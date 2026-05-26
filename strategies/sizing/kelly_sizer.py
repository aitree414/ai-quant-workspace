"""
Kelly Criterion position sizing — optimal fraction for long-term growth.

Uses 1/4 Kelly for conservative sizing to reduce volatility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Result of a Kelly calculation."""
    full_kelly_fraction: float = 0.0    # f* = (p*b - q) / b
    conservative_fraction: float = 0.0  # full_kelly * conservative_ratio
    dollar_amount: float = 0.0          # capital * conservative_fraction
    total_trades_used: int = 0          # How many trades used in estimation
    confidence: str = "low"             # "high" if >30, "medium" if >10, "low" otherwise


class KellySizer:
    """Position sizer using the Kelly Criterion.

    f* = (p * b - q) / b
    where:
        p = win rate
        q = 1 - p (loss rate)
        b = avg_win / abs(avg_loss) (odds ratio)
    """

    def __init__(
        self,
        conservative_ratio: float = 0.25,
        min_trades_calc: int = 5,
        max_position_fraction: float = 0.20,
        high_conf_threshold: int = 30,
        med_conf_threshold: int = 10,
    ) -> None:
        self.conservative_ratio = conservative_ratio
        self.min_trades = min_trades_calc
        self.max_position_fraction = max_position_fraction
        self.high_conf = high_conf_threshold
        self.med_conf = med_conf_threshold

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> KellyResult:
        """Compute Kelly fraction from win/loss statistics.

        Args:
            win_rate: Win rate as a fraction (0.0 to 1.0).
            avg_win: Average winning trade return (positive).
            avg_loss: Average losing trade return (negative or positive absolute).
            capital: Current capital.

        Returns:
            KellyResult with computed fractions and dollar amount.
        """
        # Clamp win rate
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p

        # Handle edge cases
        if p == 1.0:
            # All wins — full Kelly would be infinite, clamp to max
            kelly = self.max_position_fraction
            conf = "low"
        elif p == 0.0:
            # All losses — no bet
            kelly = 0.0
            conf = "low"
        elif avg_loss == 0:
            # No measurable loss — treat as all positive
            kelly = min(p * 2, self.max_position_fraction)
            conf = "low"
        else:
            # Standard Kelly
            b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
            kelly = (p * b - q) / b
            kelly = max(0.0, min(kelly, self.max_position_fraction))

        conservative = kelly * self.conservative_ratio

        return KellyResult(
            full_kelly_fraction=round(kelly, 4),
            conservative_fraction=round(conservative, 4),
            dollar_amount=round(capital * conservative, 2),
            total_trades_used=0,
            confidence="low",
        )

    def estimate_from_history(
        self,
        trade_history: list[dict],
        capital: float,
        current_iv: float = 20.0,
        volatility_multiplier: float = 1.0,
    ) -> KellyResult:
        """Estimate Kelly from trade history.

        Args:
            trade_history: List of dicts with 'return_pct' or 'pnl' keys.
            capital: Current capital.
            current_iv: Current implied volatility for vol adjustment.
            volatility_multiplier: Pre-computed vol multiplier from RiskController.

        Returns:
            KellyResult with computed values.
        """
        n_trades = len(trade_history)
        result = KellyResult(total_trades_used=n_trades)

        if n_trades < self.min_trades:
            # Not enough data — very conservative default
            frac = min(0.01, self.max_position_fraction * self.conservative_ratio)
            result.full_kelly_fraction = frac
            result.conservative_fraction = frac
            result.dollar_amount = capital * frac
            result.confidence = "low"
            return result

        # Set confidence level
        if n_trades >= self.high_conf:
            result.confidence = "high"
        elif n_trades >= self.med_conf:
            result.confidence = "medium"

        # Compute win/loss stats
        wins = [t for t in trade_history if t.get("return_pct", 0) > 0]
        losses = [t for t in trade_history if t.get("return_pct", 0) <= 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = sum(t.get("return_pct", 0) for t in wins) / len(wins) / 100 if wins else 0
        avg_loss = abs(sum(t.get("return_pct", 0) for t in losses) / len(losses) / 100) if losses else 0

        # Use pnl if return_pct not available
        if avg_win == 0 and avg_loss == 0:
            win_pnls = [t.get("pnl", 0) for t in wins]
            loss_pnls = [abs(t.get("pnl", 0)) for t in losses]
            avg_win = sum(win_pnls) / len(win_pnls) / capital if win_pnls else 0
            avg_loss = sum(loss_pnls) / len(loss_pnls) / capital if loss_pnls else 0

        kelly_result = self.calculate(win_rate, avg_win, avg_loss, capital)

        # Apply volatility multiplier
        vol_adj = volatility_multiplier * self.get_iv_multiplier(current_iv)
        kelly_result.conservative_fraction = round(kelly_result.conservative_fraction * vol_adj, 4)
        kelly_result.dollar_amount = round(capital * kelly_result.conservative_fraction, 2)
        kelly_result.total_trades_used = n_trades

        return kelly_result

    def get_iv_multiplier(self, iv: float) -> float:
        """Simple IV-based reduction (independent of RiskController)."""
        if iv <= 20:
            return 1.0
        elif iv <= 30:
            return 0.80
        elif iv <= 40:
            return 0.60
        elif iv <= 60:
            return 0.40
        else:
            return 0.20

    def calculate_stock_position(
        self,
        result: KellyResult,
        capital: float,
        price: float,
        market: str = "TW",
        lot_size: int = 0,
        max_qty: int = 0,
    ) -> dict:
        """Convert Kelly dollar amount to stock shares/qty.

        Args:
            result: KellyResult from calculate() or estimate_from_history().
            capital: Current capital.
            price: Price per share.
            market: "TW" or "US" (determines rounding).
            lot_size: Override lot size. Default: 1000 for TW, 1 for US.
            max_qty: Maximum quantity cap (0 = no cap).

        Returns:
            {"qty": int, "dollar_amount": float, "fraction_used": float}
        """
        if lot_size == 0:
            lot_size = 1000 if market == "TW" else 1

        dollar_amount = min(result.dollar_amount, capital * self.max_position_fraction)
        raw_qty = dollar_amount / price if price > 0 else 0
        qty = int(raw_qty / lot_size) * lot_size

        if max_qty > 0:
            qty = min(qty, max_qty)

        if qty < lot_size:
            qty = 0

        fraction_used = (qty * price) / capital if capital > 0 else 0
        return {
            "qty": qty,
            "dollar_amount": qty * price,
            "fraction_used": round(fraction_used, 4),
        }

    def calculate_options_contracts(
        self,
        result: KellyResult,
        capital: float,
        premium: float,
        max_contracts: int = 10,
        min_contracts: int = 1,
    ) -> int:
        """Calculate number of option contracts from Kelly result.

        Args:
            result: KellyResult from calculate() or estimate_from_history().
            capital: Current capital.
            premium: Premium per contract.
            max_contracts: Maximum contracts allowed.
            min_contracts: Minimum contracts (default 1).

        Returns:
            Number of contracts (at least min_contracts if dollar_amount >= premium).
        """
        if premium <= 0:
            return min_contracts

        dollar_amount = min(result.dollar_amount, capital * self.max_position_fraction)
        raw_contracts = dollar_amount / premium
        contracts = max(min_contracts, int(raw_contracts))

        if max_contracts > 0:
            contracts = min(contracts, max_contracts)

        return contracts
