#!/usr/bin/env python3
"""
Telegram Monitor Bot — sends daily AI investment committee analysis
to your Telegram chat.

Features:
  - Daily market scan across your watchlist
  - Sends buy/sell alerts when the committee reaches consensus
  - Morning briefing with portfolio overview
  - On-demand analysis via Telegram commands

Setup:
  1. Create a bot via @BotFather on Telegram
  2. Get your chat ID via @userinfobot
  3. Set environment variables:
     export TELEGRAM_BOT_TOKEN="your-bot-token"
     export TELEGRAM_CHAT_ID="your-chat-id"

Usage:
    # Run once (single scan)
    python telegram_monitor.py --once

    # Run as daemon (scans every N hours)
    python telegram_monitor.py --interval 6

    # Custom watchlist
    python telegram_monitor.py --once --tickers AAPL NVDA 2330 2382

    # Test message
    python telegram_monitor.py --test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from backtesting.engine import BacktestResult
from strategies.committee import run_committee, _result_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("telegram_monitor")


# ---------------------------------------------------------------------------
# Default watchlist
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL", "NVDA", "TSLA", "MSFT",
    "2330", "2382", "2317",
]


# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------

class TelegramBot:
    """Simple Telegram Bot API wrapper."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        if not self.token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not set. "
                "Create a bot via @BotFather and set the env var."
            )
        if not self.chat_id:
            raise ValueError(
                "TELEGRAM_CHAT_ID not set. "
                "Get your chat ID via @userinfobot and set the env var."
            )

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_preview: bool = True,
    ) -> bool:
        """Send a message to the configured chat."""
        if requests is None:
            logger.error("requests package not installed")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                logger.info("Message sent to Telegram")
                return True
            else:
                logger.error(
                    "Telegram API error %d: %s",
                    resp.status_code, resp.text,
                )
                return False
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Analysis & Formatting
# ---------------------------------------------------------------------------

def analyse_stock(
    ticker: str,
    lookback_days: int = 90,
    fallback: bool = True,
    use_claude: bool = False,
) -> Optional[dict]:
    """Run committee analysis on a single stock."""
    end = date.today().strftime("%Y-%m-%d")
    # Calculate start date
    from datetime import timedelta
    start_date = date.today() - timedelta(days=lookback_days)
    start = start_date.strftime("%Y-%m-%d")

    try:
        result = run_committee(
            ticker=ticker,
            start=start,
            end=end,
            fallback=fallback,
            use_claude=use_claude,
        )
        metrics = _result_to_dict(result)
        metrics["ticker"] = ticker
        metrics["status"] = "ok"

        # Determine current signal direction from recent trades
        if result.trades:
            last_trade = result.trades[-1]
            metrics["last_trade_date"] = last_trade.entry_date
            metrics["last_trade_return"] = round(last_trade.return_pct * 100, 2)
        else:
            metrics["last_trade_date"] = "N/A"
            metrics["last_trade_return"] = 0.0

        return metrics

    except Exception as exc:
        logger.error("Analysis failed for %s: %s", ticker, exc)
        return {
            "ticker": ticker,
            "status": "error",
            "error": str(exc),
        }


def format_daily_report(results: list[dict]) -> str:
    """Format analysis results as an HTML Telegram message."""
    today = date.today().strftime("%Y-%m-%d")

    lines = [
        f"<b>📊 AI 投資委員會 — 每日報告</b>",
        f"<i>{today}</i>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Separate into categories
    bullish = []
    bearish = []
    neutral = []
    errors = []

    for r in results:
        if r.get("status") != "ok":
            errors.append(r)
            continue

        ret = r.get("total_return_pct", 0)
        sharpe = r.get("sharpe_ratio", 0)

        if sharpe > 0.5 and ret > 0:
            bullish.append(r)
        elif sharpe < -0.5 or ret < -5:
            bearish.append(r)
        else:
            neutral.append(r)

    # Bullish signals
    if bullish:
        lines.append("")
        lines.append("🟢 <b>看多信號：</b>")
        for r in sorted(bullish, key=lambda x: x["total_return_pct"], reverse=True):
            emoji = "🚀" if r["total_return_pct"] > 10 else "📈"
            lines.append(
                f"  {emoji} <b>{r['ticker']}</b>: "
                f"{r['total_return_pct']:+.2f}% "
                f"(Sharpe: {r['sharpe_ratio']:.2f})"
            )

    # Bearish signals
    if bearish:
        lines.append("")
        lines.append("🔴 <b>看空信號：</b>")
        for r in sorted(bearish, key=lambda x: x["total_return_pct"]):
            lines.append(
                f"  📉 <b>{r['ticker']}</b>: "
                f"{r['total_return_pct']:+.2f}% "
                f"(MaxDD: {r['max_drawdown_pct']:.2f}%)"
            )

    # Neutral
    if neutral:
        lines.append("")
        lines.append("⚪ <b>觀望：</b>")
        for r in neutral:
            lines.append(
                f"  ➖ <b>{r['ticker']}</b>: "
                f"{r['total_return_pct']:+.2f}%"
            )

    # Errors
    if errors:
        lines.append("")
        lines.append("⚠️ <b>分析失敗：</b>")
        for r in errors:
            lines.append(f"  ❌ {r['ticker']}: {r.get('error', 'unknown')}")

    # Summary
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        avg_ret = sum(r["total_return_pct"] for r in ok_results) / len(ok_results)
        best = max(ok_results, key=lambda x: x["total_return_pct"])
        worst = min(ok_results, key=lambda x: x["total_return_pct"])

        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📋 <b>摘要</b>")
        lines.append(f"  平均報酬：{avg_ret:+.2f}%")
        lines.append(f"  最佳：{best['ticker']} ({best['total_return_pct']:+.2f}%)")
        lines.append(f"  最差：{worst['ticker']} ({worst['total_return_pct']:+.2f}%)")
        lines.append(f"  分析股數：{len(ok_results)} / {len(results)}")

    lines.append("")
    lines.append("<i>— AI Investment Committee Bot 🤖</i>")

    return "\n".join(lines)


def format_alert(result: dict) -> str:
    """Format a single stock alert message."""
    ticker = result["ticker"]
    ret = result.get("total_return_pct", 0)
    sharpe = result.get("sharpe_ratio", 0)
    max_dd = result.get("max_drawdown_pct", 0)

    if sharpe > 1.0 and ret > 5:
        emoji = "🚨🟢"
        action = "強力看多"
    elif sharpe < -1.0 or ret < -10:
        emoji = "🚨🔴"
        action = "強力看空"
    elif ret > 0:
        emoji = "🟢"
        action = "溫和看多"
    else:
        emoji = "🔴"
        action = "溫和看空"

    return (
        f"{emoji} <b>AI 投資委員會警報 — {ticker}</b>\n"
        f"\n"
        f"信號：<b>{action}</b>\n"
        f"90日報酬：{ret:+.2f}%\n"
        f"Sharpe Ratio：{sharpe:.2f}\n"
        f"最大回撤：{max_dd:.2f}%\n"
        f"\n"
        f"<i>{date.today()}</i>"
    )


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def run_scan(
    tickers: list[str],
    bot: TelegramBot,
    fallback: bool = True,
    use_claude: bool = False,
    alert_threshold: float = 1.0,
) -> None:
    """Run a full scan and send results to Telegram."""
    logger.info("Starting scan of %d stocks...", len(tickers))

    results = []
    for ticker in tickers:
        logger.info("Analysing %s...", ticker)
        r = analyse_stock(ticker, fallback=fallback, use_claude=use_claude)
        if r:
            results.append(r)

    # Send daily report
    report = format_daily_report(results)
    bot.send_message(report)

    # Send individual alerts for strong signals
    for r in results:
        if r.get("status") != "ok":
            continue
        sharpe = abs(r.get("sharpe_ratio", 0))
        if sharpe > alert_threshold:
            alert = format_alert(r)
            bot.send_message(alert)
            time.sleep(1)  # Rate limit

    logger.info("Scan complete. %d stocks analysed.", len(results))


def run_daemon(
    tickers: list[str],
    bot: TelegramBot,
    interval_hours: int = 6,
    fallback: bool = True,
    use_claude: bool = False,
) -> None:
    """Run continuous monitoring loop."""
    logger.info(
        "Starting monitor daemon (interval: %dh, stocks: %d)",
        interval_hours, len(tickers),
    )

    # Send startup message
    bot.send_message(
        f"🤖 <b>AI 投資委員會監控啟動</b>\n"
        f"監控股票：{', '.join(tickers)}\n"
        f"掃描間隔：每 {interval_hours} 小時\n"
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
    )

    while True:
        try:
            run_scan(tickers, bot, fallback=fallback, use_claude=use_claude)
        except Exception as exc:
            logger.error("Scan error: %s", exc)
            bot.send_message(f"⚠️ 掃描錯誤：{exc}")

        logger.info("Next scan in %d hours...", interval_hours)
        time.sleep(interval_hours * 3600)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Investment Committee — Telegram Monitor Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_WATCHLIST,
        help="Watchlist tickers",
    )
    parser.add_argument("--once", action="store_true", help="Run single scan then exit")
    parser.add_argument("--interval", type=int, default=6, help="Scan interval in hours")
    parser.add_argument("--fallback", action="store_true", help="Use fallback mode (no LLM)")
    parser.add_argument("--use-claude", action="store_true", help="Enable ClaudeAgent")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    parser.add_argument("--token", default=None, help="Telegram bot token")
    parser.add_argument("--chat-id", default=None, help="Telegram chat ID")

    args = parser.parse_args()

    bot = TelegramBot(token=args.token, chat_id=args.chat_id)

    if args.test:
        bot.send_message(
            "✅ <b>AI 投資委員會 Bot 測試成功！</b>\n"
            f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        return

    if args.once:
        run_scan(
            args.tickers, bot,
            fallback=args.fallback,
            use_claude=args.use_claude,
        )
    else:
        run_daemon(
            args.tickers, bot,
            interval_hours=args.interval,
            fallback=args.fallback,
            use_claude=args.use_claude,
        )


if __name__ == "__main__":
    main()
