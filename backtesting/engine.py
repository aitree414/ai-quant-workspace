"""
Lightweight vectorised backtesting engine.

Operates on OHLCV DataFrames and signal Series to produce portfolio-level
performance metrics using vectorised (batch) operations for speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single closed trade for post-hoc analysis."""

    entry_date: str
    exit_date: str
    side: str  # "long" | "short"  (short not yet supported, reserved)
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Aggregated performance metrics from a backtest run."""

    total_return_pct: float
    annualised_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Vectorised backtesting engine.

    Parameters:
        initial_capital: Starting portfolio capital (default 100 000).
        commission:      Round-trip commission rate as a fraction
                         (default 0.001 = 0.1 %).
        slippage:        Slippage per trade as a fraction (default 0.0).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
    ) -> BacktestResult:
        """Run a full backtest on a single symbol.

        Args:
            data: OHLCV DataFrame with a ``DatetimeIndex`` (must include a
                  ``Close`` column).
            signals: A **boolean** ``pd.Series`` indexed the same as ``data``.
                      ``True`` means "hold a position from this bar onward".

        Returns:
            A :class:`BacktestResult` with aggregated metrics and the
            equity curve.

        Raises:
            ValueError: If required columns are missing or index does not
                        align.
        """
        self._validate_input(data, signals)

        close = data["Close"]
        # --- vectorised daily returns ---
        daily_returns = close.pct_change().fillna(0.0)

        # --- position & equity curve ---
        # Shift signals by 1 to avoid look-ahead bias: we enter on the *next*
        # bar after the signal fires.
        position = signals.shift(1).fillna(False).astype(int)

        # Transaction costs: every change in position incurs commission
        trade_count = position.diff().abs().fillna(0)
        transaction_cost = trade_count * self.commission

        strategy_returns = position * daily_returns - transaction_cost

        equity = self.initial_capital * (1.0 + strategy_returns).cumprod()

        # --- metrics ---
        trades = self._extract_trades(position, close, daily_returns)
        metrics = self._compute_metrics(equity, daily_returns, trades)

        return BacktestResult(
            **metrics,
            equity_curve=equity,
            trades=trades,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_input(self, data: pd.DataFrame, signals: pd.Series) -> None:
        if "Close" not in data.columns:
            raise ValueError("data must contain a 'Close' column")
        if not data.index.equals(signals.index):
            raise ValueError("data and signals must share the same index")
        if signals.dtype != bool:
            raise TypeError("signals Series must have dtype bool")

    def _extract_trades(
        self,
        position: pd.Series,
        close: pd.Series,
        daily_returns: pd.Series,
    ) -> list[TradeRecord]:
        """Identify individual trades from the position series."""
        trades: list[TradeRecord] = []
        # Diff: 1 → entry, -1 → exit
        entries = position.diff()

        entry_idx = entries[entries == 1].index
        exit_idx = entries[entries == -1].index

        # If still holding at the end, force a final exit
        if position.iloc[-1] == 1:
            exit_idx = exit_idx.append(position.index[-1:])

        n = min(len(entry_idx), len(exit_idx))
        for i in range(n):
            e_in = entry_idx[i]
            e_out = exit_idx[i]
            p_in = close.loc[e_in]
            p_out = close.loc[e_out]
            ret = (p_out / p_in) - 1.0
            pnl = self.initial_capital * ret * 0.01  # approximate per-trade PnL
            trades.append(
                TradeRecord(
                    entry_date=str(e_in),
                    exit_date=str(e_out),
                    side="long",
                    entry_price=float(p_in),
                    exit_price=float(p_out),
                    size=1.0,
                    pnl=float(pnl),
                    return_pct=float(ret),
                )
            )
        return trades

    @staticmethod
    def _compute_metrics(
        equity: pd.Series,
        daily_returns: pd.Series,
        trades: list[TradeRecord],
    ) -> dict:
        """Calculate aggregate performance statistics."""
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

        n_days = len(equity)
        ann_factor = 252  # trading days per year
        years = n_days / ann_factor

        ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        # Volatility (annualised)
        strat_returns = daily_returns * (
            equity.pct_change().fillna(0).values  # re-compute from equity
        )
        vol = strat_returns.std() * np.sqrt(ann_factor) if years > 0 else 0.0

        sharpe = (ann_return / vol) if vol > 0 else 0.0

        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Win rate
        wins = sum(1 for t in trades if t.return_pct > 0)
        win_rate = wins / len(trades) if trades else 0.0

        return {
            "total_return_pct": float(total_return * 100),
            "annualised_return_pct": float(ann_return * 100),
            "volatility_pct": float(vol * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd * 100),
            "win_rate": float(win_rate * 100),
            "total_trades": len(trades),
        }
