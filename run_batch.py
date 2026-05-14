#!/usr/bin/env python3
"""
Batch runner — run the Investment Committee across multiple stocks and
produce a comparison report.

Usage:
    # Quick batch with fallback mode (no API needed)
    python run_batch.py --fallback

    # Custom stock list
    python run_batch.py --tickers AAPL NVDA TSLA 2330 2382 --fallback

    # Full AI mode with Claude + DeepSeek
    python run_batch.py --tickers AAPL NVDA --start 2024-01-01 --end 2025-01-01

    # Export to JSON
    python run_batch.py --fallback --json --output results/batch_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtesting.engine import BacktestResult
from strategies.committee import run_committee, _result_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch")

# ---------------------------------------------------------------------------
# Default stock universe
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    # US Tech
    "AAPL", "NVDA", "TSLA", "MSFT", "GOOG",
    # Taiwan
    "2330", "2382", "2317",
    # Hong Kong
    "0700.HK", "9988.HK",
]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    tickers: list[str],
    start: str = "2024-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    fallback: bool = True,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    use_claude: bool = False,
    claude_api_key: Optional[str] = None,
) -> list[dict]:
    """Run the committee pipeline on each ticker and collect results.

    Returns:
        List of result dicts, one per ticker.
    """
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    results: list[dict] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("━" * 55)
        logger.info("Batch [%d/%d]: %s", i, len(tickers), ticker)
        logger.info("━" * 55)

        try:
            bt_result = run_committee(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                fallback=fallback,
                initial_capital=initial_capital,
                commission=commission,
                use_claude=use_claude,
                claude_api_key=claude_api_key,
            )
            row = _result_to_dict(bt_result)
            row["ticker"] = ticker
            row["status"] = "ok"
            results.append(row)

        except Exception as exc:
            logger.error("Failed for %s: %s", ticker, exc)
            results.append({
                "ticker": ticker,
                "status": "error",
                "error": str(exc),
                "total_return_pct": 0.0,
                "annualised_return_pct": 0.0,
                "volatility_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            })

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_batch_report(
    results: list[dict],
    start: str,
    end: str,
) -> None:
    """Print a formatted comparison table."""
    print()
    print("=" * 90)
    print("  AI 投資委員會 — 多股票批量回測報告")
    print(f"  分析期間：{start} → {end}")
    print("=" * 90)

    # Header
    print(f"  {'股票':<12} {'報酬率':>10} {'年化':>10} {'波動率':>10} "
          f"{'Sharpe':>10} {'回撤':>10} {'勝率':>8} {'交易':>6} {'狀態':>6}")
    print("  " + "─" * 86)

    for r in results:
        if r["status"] == "error":
            print(f"  {r['ticker']:<12} {'—':>10} {'—':>10} {'—':>10} "
                  f"{'—':>10} {'—':>10} {'—':>8} {'—':>6} {'ERROR':>6}")
            continue

        print(
            f"  {r['ticker']:<12} "
            f"{r['total_return_pct']:>+9.2f}% "
            f"{r['annualised_return_pct']:>+9.2f}% "
            f"{r['volatility_pct']:>9.2f}% "
            f"{r['sharpe_ratio']:>10.2f} "
            f"{r['max_drawdown_pct']:>9.2f}% "
            f"{r['win_rate']:>7.1f}% "
            f"{r['total_trades']:>6d} "
            f"{'OK':>6}"
        )

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_return = sum(r["total_return_pct"] for r in ok_results) / len(ok_results)
        best = max(ok_results, key=lambda r: r["total_return_pct"])
        worst = min(ok_results, key=lambda r: r["total_return_pct"])

        print("  " + "─" * 86)
        print(f"  平均報酬率：{avg_return:+.2f}%")
        print(f"  最佳表現：{best['ticker']} ({best['total_return_pct']:+.2f}%)")
        print(f"  最差表現：{worst['ticker']} ({worst['total_return_pct']:+.2f}%)")
        print(f"  成功分析：{len(ok_results)} / {len(results)} 檔")

    print("=" * 90)


def save_json_report(results: list[dict], output_path: str) -> None:
    """Save results as a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON report saved to %s", path)


def save_csv_report(results: list[dict], output_path: str) -> None:
    """Save results as a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("CSV report saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Investment Committee — batch multi-stock backtesting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="List of stock tickers to analyse",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--interval", default="1d", help="Bar interval")
    parser.add_argument("--fallback", action="store_true", help="Use fallback mode (no LLM)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--use-claude", action="store_true", help="Enable ClaudeAgent")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--output", default=None, help="Save report to file (JSON or CSV)")

    args = parser.parse_args()
    end = args.end or date.today().strftime("%Y-%m-%d")

    results = run_batch(
        tickers=args.tickers,
        start=args.start,
        end=end,
        interval=args.interval,
        fallback=args.fallback,
        initial_capital=args.capital,
        commission=args.commission,
        use_claude=args.use_claude,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_batch_report(results, args.start, end)

    if args.output:
        if args.output.endswith(".csv"):
            save_csv_report(results, args.output)
        else:
            save_json_report(results, args.output)


if __name__ == "__main__":
    main()
