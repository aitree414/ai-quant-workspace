#!/usr/bin/env python3
"""
Parameter optimiser — grid search over agent weights and indicator parameters
to find the best configuration for the Investment Committee.

Usage:
    # Quick optimisation with defaults
    python run_optimize.py --ticker AAPL --fallback

    # Custom parameter grid
    python run_optimize.py --ticker 2330 --fallback --rsi-range 10 20 --sma-range 30 60

    # Save results
    python run_optimize.py --ticker AAPL --fallback --output data/results/optimize_AAPL.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtesting.engine import BacktestEngine, BacktestResult
from strategies.agents.momentum_agent import MomentumAgent
from strategies.agents.value_agent import ValueAgent
from strategies.agents.cio_agent import CIOAgent
from strategies.agents.base_agent import Signal
from strategies.committee import _signals_to_position, _fill_prices
from utils.data_loader import load_single

logging.basicConfig(
    level=logging.WARNING,  # Quiet during optimisation
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Parameter grid definitions
# ---------------------------------------------------------------------------

DEFAULT_GRID = {
    # MomentumAgent parameters
    "rsi_period": [10, 14, 20],
    "macd_fast": [8, 12],
    "macd_slow": [21, 26],
    "sma_period": [30, 50, 100],

    # Agent weights (must sum to 1.0)
    "momentum_weight": [0.3, 0.4, 0.5, 0.6, 0.7],

    # CIO thresholds
    "buy_threshold": [0.15, 0.25, 0.35],
}


def build_param_combos(grid: dict) -> list[dict]:
    """Generate all combinations from the parameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # Derive value_weight from momentum_weight
        params["value_weight"] = round(1.0 - params["momentum_weight"], 2)
        # Sell threshold is negative of buy threshold
        params["sell_threshold"] = -params["buy_threshold"]
        combos.append(params)
    return combos


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_params(
    data: pd.DataFrame,
    params: dict,
    symbol: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> dict:
    """Run a single backtest with given parameters and return metrics."""
    try:
        # Create agents with custom parameters
        momentum = MomentumAgent(
            rsi_period=params["rsi_period"],
            macd_fast=params["macd_fast"],
            macd_slow=params["macd_slow"],
            sma_period=params["sma_period"],
        )
        value = ValueAgent(fallback_mode=True)

        # Generate signals
        mom_signals = momentum.generate_signals(data)
        val_signals = value.generate_signals(data)

        agent_signals = {
            "momentum-agent": mom_signals,
            "value-agent": val_signals,
        }

        # CIO with custom weights and thresholds
        cio = CIOAgent(
            weights={
                "momentum-agent": params["momentum_weight"],
                "value-agent": params["value_weight"],
            },
            buy_threshold=params["buy_threshold"],
            sell_threshold=params["sell_threshold"],
        )
        consensus = cio.synthesise(agent_signals, data.index, symbol=symbol)
        _fill_prices(consensus, data)

        # Backtest
        position = _signals_to_position(consensus, data.index)
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
        )
        result = engine.run(data, position)

        return {
            **params,
            "total_return_pct": round(result.total_return_pct, 4),
            "annualised_return_pct": round(result.annualised_return_pct, 4),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown_pct": round(result.max_drawdown_pct, 4),
            "win_rate": round(result.win_rate, 2),
            "total_trades": result.total_trades,
            "status": "ok",
        }

    except Exception as exc:
        return {
            **params,
            "total_return_pct": 0.0,
            "annualised_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "status": f"error: {exc}",
        }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_optimization(
    ticker: str,
    start: str,
    end: str,
    grid: Optional[dict] = None,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> pd.DataFrame:
    """Run grid search over all parameter combinations.

    Returns:
        DataFrame with one row per parameter combination, sorted by Sharpe.
    """
    if grid is None:
        grid = DEFAULT_GRID

    combos = build_param_combos(grid)
    logger.info("Parameter grid: %d combinations", len(combos))

    # Load data once
    data = load_single(ticker, start=start, end=end)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")

    data.attrs["symbol"] = ticker
    logger.info("Loaded %d bars for %s", len(data), ticker)

    results = []
    for i, params in enumerate(combos, 1):
        if i % 50 == 0 or i == 1:
            logger.info("Progress: %d / %d (%.0f%%)", i, len(combos), i / len(combos) * 100)

        row = evaluate_params(data, params, ticker, initial_capital, commission)
        results.append(row)

    df = pd.DataFrame(results)

    # Sort by Sharpe ratio (descending)
    ok_mask = df["status"] == "ok"
    df_ok = df[ok_mask].sort_values("sharpe_ratio", ascending=False)
    df_err = df[~ok_mask]
    df = pd.concat([df_ok, df_err], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_optimization_report(df: pd.DataFrame, ticker: str, top_n: int = 10) -> None:
    """Print the top N parameter combinations."""
    ok = df[df["status"] == "ok"]

    print()
    print("=" * 100)
    print(f"  AI 投資委員會 — 參數優化報告 ({ticker})")
    print(f"  總共測試：{len(df)} 組參數")
    print(f"  成功：{len(ok)} 組 | 失敗：{len(df) - len(ok)} 組")
    print("=" * 100)

    if ok.empty:
        print("  沒有成功的結果。")
        return

    print(f"\n  🏆 Top {min(top_n, len(ok))} 最佳參數組合（按 Sharpe Ratio 排序）：")
    print("  " + "─" * 96)

    header = (
        f"  {'#':>3} {'RSI':>4} {'MACD':>8} {'SMA':>4} "
        f"{'Mom%':>5} {'Val%':>5} {'Thres':>5} "
        f"{'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinR':>6} {'Trades':>6}"
    )
    print(header)
    print("  " + "─" * 96)

    for i, (_, row) in enumerate(ok.head(top_n).iterrows(), 1):
        macd_str = f"{int(row['macd_fast'])}/{int(row['macd_slow'])}"
        print(
            f"  {i:>3} "
            f"{int(row['rsi_period']):>4} "
            f"{macd_str:>8} "
            f"{int(row['sma_period']):>4} "
            f"{row['momentum_weight']:>5.0%} "
            f"{row['value_weight']:>5.0%} "
            f"{row['buy_threshold']:>5.2f} "
            f"{row['total_return_pct']:>+8.2f}% "
            f"{row['sharpe_ratio']:>8.2f} "
            f"{row['max_drawdown_pct']:>7.2f}% "
            f"{row['win_rate']:>5.1f}% "
            f"{int(row['total_trades']):>6}"
        )

    # Best params summary
    best = ok.iloc[0]
    print()
    print("  " + "─" * 96)
    print("  🎯 最佳參數：")
    print(f"     RSI Period:       {int(best['rsi_period'])}")
    print(f"     MACD Fast/Slow:   {int(best['macd_fast'])}/{int(best['macd_slow'])}")
    print(f"     SMA Period:       {int(best['sma_period'])}")
    print(f"     Momentum Weight:  {best['momentum_weight']:.0%}")
    print(f"     Value Weight:     {best['value_weight']:.0%}")
    print(f"     Buy Threshold:    {best['buy_threshold']:.2f}")
    print(f"     → Return: {best['total_return_pct']:+.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Investment Committee — parameter optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--fallback", action="store_true", help="(ignored, always fallback for speed)")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    parser.add_argument("--output", default=None, help="Save full results to CSV")
    parser.add_argument("--json", action="store_true", help="Output top results as JSON")

    # Custom grid ranges
    parser.add_argument("--rsi-range", nargs="+", type=int, default=None,
                        help="RSI periods to test (e.g. 10 14 20)")
    parser.add_argument("--sma-range", nargs="+", type=int, default=None,
                        help="SMA periods to test (e.g. 30 50 100)")

    args = parser.parse_args()
    end = args.end or date.today().strftime("%Y-%m-%d")

    # Build custom grid if specified
    grid = dict(DEFAULT_GRID)
    if args.rsi_range:
        grid["rsi_period"] = args.rsi_range
    if args.sma_range:
        grid["sma_period"] = args.sma_range

    logger.info("Starting optimisation for %s [%s → %s]", args.ticker, args.start, end)

    df = run_optimization(
        ticker=args.ticker,
        start=args.start,
        end=end,
        grid=grid,
        initial_capital=args.capital,
        commission=args.commission,
    )

    if args.json:
        ok = df[df["status"] == "ok"].head(args.top)
        print(json.dumps(ok.to_dict(orient="records"), ensure_ascii=False, indent=2))
    else:
        print_optimization_report(df, args.ticker, top_n=args.top)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("Full results saved to %s", path)


if __name__ == "__main__":
    main()
