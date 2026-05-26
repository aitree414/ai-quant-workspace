"""Risk Manager agent — has veto power over committee decisions.

Evaluates portfolio-level risk: concentration, total exposure,
single-trade risk limits, and Greeks exposure.
"""

from __future__ import annotations

import logging

from .base_agent import BaseOptionsAgent, OptionsSignal

try:
    from strategies.risk import RiskController
    _risk_ctrl = RiskController()
except ImportError:
    _risk_ctrl = None

logger = logging.getLogger(__name__)

MAX_SINGLE_TRADE_RISK_PCT = 2.0  # Max 2% of capital per trade
MAX_TOTAL_POSITION_PCT = 50.0    # Max 50% of capital in positions
MAX_CONCENTRATION_PCT = 30.0     # Max 30% in one symbol
MAX_DELTA_EXPOSURE = 50.0        # Max net delta across portfolio


class RiskManager(BaseOptionsAgent):
    """Risk management agent with veto power."""

    name = "risk_manager"
    weight = 1.0

    def analyze(
        self,
        symbol: str,
        underlying_price: float,
        option_chain: dict,
        hist_data: dict,
        iv_data: dict,
        portfolio=None,
    ) -> OptionsSignal:
        reasons = []
        strategy = "none"
        action = "hold"
        confidence = 0.5

        # Get portfolio state
        if portfolio:
            total_equity = portfolio.get("total_equity", 100000)
            positions = portfolio.get("positions", [])
        else:
            total_equity = 100000
            positions = []

        # --- Position count check ---
        pos_count = len(positions)
        if pos_count >= 5:
            reasons.append(f"持倉數量 {pos_count} 已達上限")
            confidence -= 0.2
            if pos_count >= 8:
                reasons.append("持倉過度集中，拒絕新增")
                return OptionsSignal(
                    symbol=symbol, strategy="none", action="hold",
                    confidence=0.0, reasoning="風控拒絕：持倉數量過多",
                    details={"veto": True},
                )

        # --- Symbol concentration check ---
        symbol_exposure = 0.0
        for p in positions:
            if p.get("symbol") == symbol:
                symbol_exposure += abs(p.get("premium", 0))

        max_pos_risk = total_equity * (MAX_SINGLE_TRADE_RISK_PCT / 100)
        max_total_risk = total_equity * (MAX_TOTAL_POSITION_PCT / 100)
        max_concentration_val = total_equity * (MAX_CONCENTRATION_PCT / 100)

        if symbol_exposure >= max_concentration_val:
            reasons.append(f"{symbol} 集中度已超過 {MAX_CONCENTRATION_PCT:.0f}%")
            confidence -= 0.3
            return OptionsSignal(
                symbol=symbol, strategy="none", action="hold",
                confidence=0.1, reasoning=f"風控拒絕：{symbol} 集中度超限",
                details={"veto": True},
            )

        # --- Total exposure check ---
        total_exposure = sum(abs(p.get("premium", 0) or 0) for p in positions)
        if total_exposure >= max_total_risk:
            reasons.append(f"總曝險 {total_exposure/total_equity*100:.0f}% 超過上限 {MAX_TOTAL_POSITION_PCT:.0f}%")
            return OptionsSignal(
                symbol=symbol, strategy="none", action="hold",
                confidence=0.0, reasoning="風控拒絕：總曝險超限",
                details={"veto": True},
            )

        # --- Greeks exposure check (if portfolio has Greeks tracking) ---
        net_delta = sum(p.get("delta", 0) or 0 for p in positions)
        theta_decay = sum(p.get("theta", 0) or 0 for p in positions)

        if abs(net_delta) > MAX_DELTA_EXPOSURE:
            reasons.append(f"淨Delta {net_delta:.1f} 超過 {MAX_DELTA_EXPOSURE:.0f}")
            confidence -= 0.2

        # --- Option-specific risk check from chain ---
        if option_chain and "calls" in option_chain:
            # Check if bid-ask spreads are reasonable
            calls = option_chain.get("calls", [])
            if calls:
                avg_spread = 0
                count = 0
                for c in calls[:5]:
                    if c.get("bid", 0) > 0 and c.get("ask", 0) > 0:
                        spread = (c["ask"] - c["bid"]) / ((c["bid"] + c["ask"]) / 2) * 100
                        avg_spread += spread
                        count += 1
                if count > 0:
                    avg_spread /= count
                    if avg_spread > 20:
                        reasons.append(f"買賣價差 {avg_spread:.0f}% 過大（流動性差）")
                        confidence -= 0.15

        # --- Delta check for the specific symbol ---
        if option_chain:
            atm_call = None
            for c in option_chain.get("calls", []):
                if abs(c["strike"] - underlying_price) / underlying_price < 0.02:
                    atm_call = c
                    break
            if atm_call and atm_call.get("delta", 0) > 0.7:
                reasons.append("ATM Call Delta 偏高，注意方向性風險")
                confidence -= 0.05

        # --- Capital adequacy ---
        pos_value = max_concentration_val - symbol_exposure
        if pos_value < total_equity * 0.01:
            reasons.append("可用資金不足")
            confidence -= 0.2

        # --- Drawdown check via RiskController ---
        if _risk_ctrl:
            dd_result = _risk_ctrl.check_drawdown(total_equity)
            if not dd_result.passed:
                for reason in dd_result.reasons:
                    reasons.append(reason)
                confidence -= 0.3
                if _risk_ctrl.is_trading_paused():
                    return OptionsSignal(
                        symbol=symbol, strategy="none", action="hold",
                        confidence=0.0, reasoning="風控拒絕：斷路器啟動中",
                        details={"veto": True},
                    )

        confidence = max(0.0, min(1.0, confidence))

        # Only recommend "none" — risk manager either approves or vetoes
        if confidence >= 0.5:
            strategy = "none"  # No veto
            action = "hold"
            reasons.append("✅ 風控審核通過")
        else:
            strategy = "none"
            action = "hold"

        return OptionsSignal(
            symbol=symbol,
            strategy=strategy,
            action=action,
            confidence=round(confidence, 2),
            reasoning="；".join(reasons) if reasons else "風控無意見",
            details={
                "veto": confidence < 0.5,
                "total_equity": total_equity,
                "total_exposure": total_exposure,
                "symbol_exposure": symbol_exposure,
                "pos_count": pos_count,
            },
        )

    def has_veto(self, signal: OptionsSignal) -> bool:
        """Check if this signal represents a veto."""
        return signal.details.get("veto", False)
