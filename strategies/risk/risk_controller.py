"""
Risk control system — exposure limits, circuit breaker, volatility-based sizing.

This is the foundation layer that all other features build upon.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/circuit_breaker_state.json")


@dataclass
class CircuitBreakerState:
    """Persistent state for the drawdown circuit breaker."""
    is_active: bool = False
    triggered_at: str = ""
    trigger_reason: str = ""
    cooldown_until: str = ""
    max_drawdown_pct: float = 0.0
    equity_at_trigger: float = 0.0
    peak_equity: float = 0.0


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    sizing_multiplier: float = 1.0


class RiskController:
    """Central risk controller for the trading system.

    Handles:
    - Per-symbol exposure limits
    - Max drawdown circuit breaker
    - Volatility-adjusted position sizing
    """

    def __init__(
        self,
        max_daily_exposure_pct: float = 0.20,
        max_drawdown_pct: float = 0.15,
        drawdown_cooldown_days: int = 7,
        baseline_iv: float = 20.0,
        state_file: str | Path = STATE_FILE,
    ) -> None:
        self.max_exposure_pct = max_daily_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_days = drawdown_cooldown_days
        self.baseline_iv = baseline_iv
        self.state_file = Path(state_file)
        self._state = self._load_state()

    # ---- Exposure check ----

    def check_exposure(
        self,
        symbol: str,
        proposed_amount: float,
        current_exposure: dict[str, float],
        capital: float,
    ) -> RiskCheckResult:
        """Reject if adding position exceeds max exposure % of capital in one symbol."""
        result = RiskCheckResult(passed=True, reasons=[])

        if capital <= 0:
            result.passed = False
            result.reasons.append("Capital is zero or negative")
            return result

        current_symbol_exposure = current_exposure.get(symbol, 0.0)
        total_after = current_symbol_exposure + proposed_amount
        exposure_pct = total_after / capital

        if exposure_pct > self.max_exposure_pct:
            result.passed = False
            result.reasons.append(
                f"{symbol} exposure {exposure_pct:.1%} exceeds limit {self.max_exposure_pct:.0%}"
            )
            return result

        return result

    # ---- Drawdown circuit breaker ----

    def check_drawdown(self, current_equity: float) -> RiskCheckResult:
        """Check if drawdown exceeds threshold. Activates circuit breaker if so."""
        result = RiskCheckResult(passed=True, reasons=[])

        # Track peak equity
        if current_equity > self._state.peak_equity:
            self._state.peak_equity = current_equity
            self._save_state()

        # If already in cooldown, check if expired
        if self._state.is_active:
            cooldown = datetime.fromisoformat(self._state.cooldown_until)
            if datetime.now() >= cooldown:
                logger.info("Circuit breaker cooldown expired — resuming trading")
                self._state.is_active = False
                self._state.triggered_at = ""
                self._state.cooldown_until = ""
                self._save_state()
                return result
            else:
                remaining = (cooldown - datetime.now()).total_seconds() / 3600
                result.passed = False
                result.reasons.append(
                    f"Circuit breaker active — {remaining:.1f}h remaining "
                    f"(drawdown: {self._state.max_drawdown_pct:.1%})"
                )
                return result

        # Check current drawdown
        peak = self._state.peak_equity
        if peak <= 0:
            return result

        dd_pct = (peak - current_equity) / peak
        if dd_pct >= self.max_drawdown_pct:
            cooldown_until = (datetime.now() + timedelta(days=self.cooldown_days)).isoformat()
            self._state.is_active = True
            self._state.triggered_at = datetime.now().isoformat()
            self._state.trigger_reason = f"Drawdown {dd_pct:.1%} >= {self.max_drawdown_pct:.0%}"
            self._state.cooldown_until = cooldown_until
            self._state.max_drawdown_pct = dd_pct
            self._state.equity_at_trigger = current_equity
            self._save_state()

            result.passed = False
            result.reasons.append(
                f"Circuit breaker TRIGGERED — drawdown {dd_pct:.1%} >= {self.max_drawdown_pct:.0%}"
            )
            return result

        return result

    def is_trading_paused(self) -> bool:
        """Check if circuit breaker is currently blocking trades."""
        if not self._state.is_active:
            return False
        cooldown = datetime.fromisoformat(self._state.cooldown_until)
        if datetime.now() >= cooldown:
            self._state.is_active = False
            self._save_state()
            return False
        return True

    def get_drawdown_pct(self, current_equity: float) -> float:
        """Compute current drawdown percentage."""
        peak = max(self._state.peak_equity, current_equity)
        if peak <= 0:
            return 0.0
        return (peak - current_equity) / peak

    # ---- Volatility-adjusted sizing ----

    def get_volatility_multiplier(self, current_iv: float) -> float:
        """Return sizing multiplier based on current IV relative to baseline.

        Rules:
            IV <= baseline * 1.2  →  1.0
            IV <= baseline * 1.5  →  0.75
            IV <= baseline * 2.0  →  0.50
            IV  > baseline * 2.0  →  0.25
        """
        if current_iv <= 0:
            return 1.0

        ratio = current_iv / self.baseline_iv

        if ratio <= 1.2:
            return 1.0
        elif ratio <= 1.5:
            return 0.75
        elif ratio <= 2.0:
            return 0.50
        else:
            return 0.25

    # ---- State persistence ----

    def _load_state(self) -> CircuitBreakerState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return CircuitBreakerState(**data)
            except Exception as e:
                logger.warning("Failed to load circuit breaker state: %s", e)
        return CircuitBreakerState()

    def _save_state(self) -> None:
        tmp = str(self.state_file) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(self._state), f, ensure_ascii=False, indent=2)
            os.replace(str(tmp), str(self.state_file))
        except Exception as e:
            logger.warning("Failed to save circuit breaker state: %s", e)

    def reset_breaker(self) -> dict:
        """Manually clear circuit breaker. Returns status dict."""
        old_state = self._state
        self._state = CircuitBreakerState(peak_equity=old_state.peak_equity)
        self._save_state()
        logger.info("Circuit breaker manually reset")
        return {"status": "ok", "reset_at": datetime.now().isoformat()}

    def get_status(self) -> dict:
        """Return current risk status as dict."""
        return {
            "trading_paused": self.is_trading_paused(),
            "circuit_breaker_active": self._state.is_active,
            "triggered_at": self._state.triggered_at,
            "trigger_reason": self._state.trigger_reason,
            "cooldown_until": self._state.cooldown_until,
            "peak_equity": self._state.peak_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "baseline_iv": self.baseline_iv,
            "max_exposure_pct": self.max_exposure_pct,
        }
