"""
Investment Committee — orchestrates multiple AI agents and runs backtests.

Usage
-----
    python -m strategies.committee --ticker 2330 --start 2025-01-01 --fallback
    python -m strategies.committee --ticker AAPL --start 2024-01-01 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date
from typing import Optional

import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from utils.data_loader import load_single
from .agents import MomentumAgent, ValueAgent, ClaudeAgent, Signal
from .agents.cio_agent import CIOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("committee")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals_to_position(signals: list[Signal], index: pd.Index) -> pd.Series:
    """Convert a list of buy/sell Signals to a boolean position Series."""
    position = pd.Series(False, index=index)
    if not signals:
        return position

    sorted_sigs = sorted(signals, key=lambda s: s.timestamp)
    in_position = False
    for sig in sorted_sigs:
        if sig.action == "buy" and not in_position:
            position[index >= sig.timestamp] = True
            in_position = True
        elif sig.action == "sell" and in_position:
            position[index >= sig.timestamp] = False
            in_position = False
    return position


def _fill_prices(signals: list[Signal], data: pd.DataFrame) -> None:
    """Fill missing prices on CIO signals using the OHLCV close prices."""
    if "Close" not in data.columns:
        return
    for sig in signals:
        try:
            ts = pd.Timestamp(sig.timestamp)
            if ts in data.index and sig.price == 0.0:
                sig.price = float(data.loc[ts, "Close"])
        except (ValueError, TypeError):
            pass


def _result_to_dict(result: BacktestResult) -> dict:
    """Serialize a BacktestResult to a JSON-safe dict."""
    return {
        "total_return_pct": round(result.total_return_pct, 4),
        "annualised_return_pct": round(result.annualised_return_pct, 4),
        "volatility_pct": round(result.volatility_pct, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "max_drawdown_pct": round(result.max_drawdown_pct, 4),
        "win_rate": round(result.win_rate, 2),
        "total_trades": result.total_trades,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_committee(
    ticker: str = "2330",
    start: str = "2025-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    fallback: bool = False,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    deepseek_api_key: Optional[str] = None,
    claude_api_key: Optional[str] = None,
    use_claude: bool = True,
) -> BacktestResult:
    """Run the full investment committee pipeline.

    1. Load OHLCV data
    2. Generate signals from MomentumAgent, ValueAgent, and ClaudeAgent
    3. CIOAgent synthesises signals into consensus
    4. Convert consensus to position series
    5. Run backtest
    """
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    logger.info("=" * 55)
    logger.info("Investment Committee — %s  [%s → %s]", ticker, start, end)
    logger.info("=" * 55)

    # --- Step 1: Load data ---
    data = load_single(ticker, start=start, end=end, interval=interval)
    if data.empty:
        logger.error("No data returned for %s", ticker)
        raise SystemExit(1)

    data.attrs["symbol"] = ticker
    logger.info("Loaded %d bars for %s", len(data), ticker)

    # --- Step 2: Generate agent signals ---
    momentum = MomentumAgent()
    value = ValueAgent(api_key=deepseek_api_key, fallback_mode=fallback)

    logger.info("--- MomentumAgent (40%%) ---")
    mom_signals = momentum.generate_signals(data)
    logger.info("  → %d signal(s)", len(mom_signals))

    logger.info("--- ValueAgent (30%%) ---")
    val_signals = value.generate_signals(data)
    logger.info("  → %d signal(s)", len(val_signals))

    agent_signals: dict[str, list[Signal]] = {
        "momentum-agent": mom_signals,
        "value-agent": val_signals,
    }

    if use_claude:
        logger.info("--- ClaudeAgent (30%%) ---")
        try:
            claude = ClaudeAgent(api_key=claude_api_key, base_url="https://api.anthropic.com")
            claude_signals = claude.generate_signals(data)
            logger.info("  → %d signal(s)", len(claude_signals))
            agent_signals["claude-agent"] = claude_signals
        except Exception as exc:
            logger.error("ClaudeAgent failed: %s — proceeding without it", exc)

    # --- Step 3: CIO consensus ---
    cio = CIOAgent()
    consensus = cio.synthesise(agent_signals, data.index, symbol=ticker)
    _fill_prices(consensus, data)
    logger.info("--- CIO ---")
    logger.info("  → %d consensus signal(s)", len(consensus))

    # --- Step 4: Convert to position & backtest ---
    position = _signals_to_position(consensus, data.index)
    logger.info("Position days: %d / %d", int(position.sum()), len(position))

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
    )
    result = engine.run(data, position)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Investment Committee — multi-agent backtesting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="2330", help="Stock ticker")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default today)")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "1m"], help="Bar interval")
    parser.add_argument("--fallback", action="store_true", help="Use fallback rule mode for ValueAgent")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--no-claude", action="store_true", help="Disable ClaudeAgent")
    parser.add_argument(
        "--claude-api-key", default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args(argv)

    result = run_committee(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        fallback=args.fallback,
        initial_capital=args.capital,
        commission=args.commission,
        use_claude=not args.no_claude,
        claude_api_key=args.claude_api_key,
    )

    if args.json:
        out = _result_to_dict(result)
        out["ticker"] = args.ticker
        out["start"] = args.start
        out["end"] = args.end or date.today().strftime("%Y-%m-%d")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        _pretty_print(result, args)

    return 0


def _pretty_print(result: BacktestResult, args: argparse.Namespace) -> None:
    """Display results in a human-readable format."""
    print()
    print("=" * 50)
    print(f"  Committee Backtest — {args.ticker}")
    print(f"  Period: {args.start} → {args.end or date.today()}")
    print("=" * 50)
    print(f"  Total Return      : {result.total_return_pct:>+8.2f} %")
    print(f"  Annualised Return : {result.annualised_return_pct:>+8.2f} %")
    print(f"  Volatility        : {result.volatility_pct:>8.2f} %")
    print(f"  Sharpe Ratio      : {result.sharpe_ratio:>8.4f}")
    print(f"  Max Drawdown      : {result.max_drawdown_pct:>8.2f} %")
    print(f"  Win Rate          : {result.win_rate:>8.2f} %")
    print(f"  Total Trades      : {result.total_trades:>8d}")
    print("-" * 50)
    if result.trades:
        top = sorted(result.trades, key=lambda t: abs(t.return_pct), reverse=True)[:3]
        print("  Top 3 trades:")
        for t in top:
            print(f"    {t.entry_date} → {t.exit_date}  "
                  f"{t.return_pct:+.2f}%  (${t.pnl:+.0f})")
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
