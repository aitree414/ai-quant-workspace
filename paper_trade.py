#!/usr/bin/env python3
"""
📋 AI 紙上交 trade — 用 Committee 即時分析，模擬槓桿交易。

用法
----
    python paper_trade.py                              # 檢視目前狀態
    python paper_trade.py --update                      # 更新數據 + 重新判斷
    python paper_trade.py --watch                       # 持續監控（每 60 秒）
    python paper_trade.py --reset                       # 重置帳戶

支援台股（自動補 .TW）與美股，可設定槓桿。
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
from pathlib import Path
from typing import Optional

import pandas as pd

from strategies.agents import MomentumAgent, ValueAgent, FlowAgent, Signal
from strategies.agents.cio_agent import CIOAgent
from utils.data_loader import load_single

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("paper_trade")

PORTFOLIO_FILE = Path("paper_portfolio.json")

# 預設追蹤標的
DEFAULT_WATCHLIST = [
    {"ticker": "2330", "name": "台積電", "leverage": 2.5, "market": "TW", "stop_loss_pct": -10.0, "take_profit_pct": 30.0},
    {"ticker": "2317", "name": "鴻海",   "leverage": 2.5, "market": "TW", "stop_loss_pct": -10.0, "take_profit_pct": 30.0},
    {"ticker": "AAPL", "name": "Apple",  "leverage": 2.0, "market": "US", "stop_loss_pct": -10.0, "take_profit_pct": 25.0},
    {"ticker": "TSLA", "name": "Tesla",  "leverage": 2.0, "market": "US", "stop_loss_pct": -15.0, "take_profit_pct": 35.0},
    {"ticker": "TX",  "name": "台指期", "leverage": 10.0, "market": "TW", "yahoo_ticker": "^TWII", "stop_loss_pct": -5.0, "take_profit_pct": 15.0, "max_qty": 2},
]


# ---------------------------------------------------------------------------
# 紙上帳戶
# ---------------------------------------------------------------------------

class PaperPortfolio:
    """紙上交 trade 帳戶，狀態持久化至 JSON。"""

    def __init__(self, initial_capital: float = 400_000) -> None:
        self.initial = initial_capital
        self.cash = initial_capital
        self.holdings: dict[str, dict] = {}  # ticker → {qty, entry_price, leverage}
        self.trades: list[dict] = []
        self.equity_history: list[dict] = []  # [{date, equity, cash, holdings_value}]
        self.last_updated: str = ""
        self.signals_cache: dict[str, dict] = {}  # ticker → latest analysis result

        if PORTFOLIO_FILE.exists():
            self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        try:
            data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
            self.cash = data.get("cash", self.initial)
            self.holdings = data.get("holdings", {})
            self.trades = data.get("trades", [])
            self.equity_history = data.get("equity_history", [])
            self.last_updated = data.get("last_updated", "")
            self.initial = data.get("initial", self.initial)
            self.signals_cache = data.get("signals_cache", {})
        except (json.JSONDecodeError, KeyError):
            pass

    def save(self) -> None:
        PORTFOLIO_FILE.write_text(
            json.dumps({
                "initial": self.initial,
                "cash": self.cash,
                "holdings": self.holdings,
                "trades": self.trades,
                "equity_history": self.equity_history,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "signals_cache": self.signals_cache,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.cash = self.initial
        self.holdings = {}
        self.trades = []
        self.equity_history = []
        self.last_updated = ""
        self.save()
        print("  🔄 帳戶已重置")

    # ---- 交易執行 ----

    @property
    def holdings_value(self) -> float:
        """當前持倉市值。"""
        return sum(h.get("current_value", 0) for h in self.holdings.values())

    @property
    def total_loan(self) -> float:
        """融資借款總額。"""
        total = 0.0
        for h in self.holdings.values():
            loan_per_share = h["entry_price"] * (1 - 1.0 / h["leverage"])
            total += loan_per_share * h["qty"]
        return total

    @property
    def total_equity(self) -> float:
        """總權益 = 現金 + 持倉市值 - 融資借款。"""
        return self.cash + self.holdings_value - self.total_loan

    @property
    def return_pct(self) -> float:
        return ((self.total_equity / self.initial) - 1) * 100

    def buy(self, ticker: str, price: float, leverage: float, name: str = "", market: str = "TW", max_qty: int = 0) -> None:
        """以融資買入。"""
        if ticker in self.holdings:
            print(f"  ⏭️  {ticker} 已有持倉，跳過")
            return

        margin_ratio = 1.0 / leverage
        cost_per_share = price * margin_ratio
        # 台股 1000 股為單位，美股/期貨 1 單位
        futures_tickers = {"TX", "MTX", "FITX", "FIMTX"}
        lot_size = 1 if ticker in futures_tickers else (1000 if market == "TW" else 1)
        max_shares = int(self.cash / (cost_per_share * lot_size)) * lot_size
        max_shares = max(max_shares, 0)
        if max_qty > 0:
            max_shares = min(max_shares, max_qty)
        if max_shares < lot_size:
            print(f"  ❌ {ticker} 資金不足 (需 NT${cost_per_share * lot_size:,.0f}/{lot_size}股)")
            return

        cost = price * margin_ratio * max_shares
        self.cash -= cost
        self.holdings[ticker] = {
            "qty": max_shares,
            "entry_price": price,
            "leverage": leverage,
            "margin_used": cost,
            "current_value": price * max_shares,
            "name": name or ticker,
            "market": market,
        }
        self.trades.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker, "side": "BUY", "price": price,
            "qty": max_shares, "leverage": leverage,
        })
        print(f"  ✅ BUY {ticker} ×{max_shares} @ {price:,.0f} 槓桿{leverage}×  (NT${cost:,.0f})")
        self.save()

    def sell(self, ticker: str, price: float) -> None:
        """賣出平倉。"""
        if ticker not in self.holdings:
            print(f"  ⏭️  {ticker} 無持倉")
            return

        h = self.holdings[ticker]
        # 返還保證金 + 損益
        pnl = (price - h["entry_price"]) * h["qty"]
        proceeds = h["margin_used"] + pnl
        self.cash += proceeds

        r = (pnl / h["margin_used"]) * 100 if h["margin_used"] > 0 else 0
        print(f"  📤 SELL {ticker} @ {price:,.0f} 損益 {pnl:+,.0f} ({r:+.2f}%)")

        self.trades.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker, "side": "SELL", "price": price,
            "qty": h["qty"], "pnl": round(pnl, 2),
            "return_pct": round(r, 2),
        })
        del self.holdings[ticker]
        self.save()

    def update_prices(self, prices: dict[str, float]) -> None:
        """更新持倉現價。"""
        for ticker, price in prices.items():
            if ticker in self.holdings:
                self.holdings[ticker]["current_value"] = price * self.holdings[ticker]["qty"]
        # 記錄權益
        self.equity_history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "equity": round(self.total_equity, 2),
            "cash": round(self.cash, 2),
        })
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.save()

    # ---- Risk: stop-loss / take-profit ----

    def check_sl_tp(self, prices: dict[str, float], watchlist: list[dict]) -> list[tuple[str, float]]:
        """Check stop-loss and take-profit for all holdings. Returns [(ticker, price)] to sell."""
        to_sell: list[tuple[str, float]] = []
        for item in watchlist:
            ticker = item["ticker"]
            if ticker not in self.holdings:
                continue
            h = self.holdings[ticker]
            price = prices.get(ticker, 0)
            if price == 0:
                continue
            ep = h["entry_price"]
            pnl_pct = (price - ep) / ep * 100
            sl = item.get("stop_loss_pct", -8.0)
            tp = item.get("take_profit_pct", 25.0)
            if pnl_pct <= sl:
                print(f"  🛑 {ticker} 觸及停損 ({pnl_pct:+.2f}% <= {sl:.2f}%)")
                to_sell.append((ticker, price))
            elif pnl_pct >= tp:
                print(f"  ✅ {ticker} 觸及停利 ({pnl_pct:+.2f}% >= {tp:.2f}%)")
                to_sell.append((ticker, price))
        return to_sell

    # ---- Risk management ----

    @property
    def total_exposure(self) -> float:
        """Total position value (long)."""
        return sum(h.get("current_value", h["entry_price"] * h["qty"]) for h in self.holdings.values())

    @property
    def exposure_ratio(self) -> float:
        """Exposure as % of equity ( > 1.0 means net leveraged )."""
        eq = self.total_equity
        return self.total_exposure / eq if eq else 0

    @property
    def max_concentration(self) -> float:
        """Largest position as % of total exposure."""
        exp = self.total_exposure
        if not exp:
            return 0.0
        max_pos = max((h.get("current_value", h["entry_price"] * h["qty"]) for h in self.holdings.values()), default=0)
        return max_pos / exp * 100

    @property
    def position_count(self) -> int:
        return len(self.holdings)

    def risk_summary(self) -> dict:
        """Return risk metrics as dict."""
        eq = self.total_equity
        return {
            "total_exposure": self.total_exposure,
            "exposure_ratio": round(self.exposure_ratio, 2),
            "max_concentration_pct": round(self.max_concentration, 1),
            "total_loan": self.total_loan,
            "leverage_ratio": round(self.total_exposure / eq, 2) if eq else 0,
            "cash_ratio": round(self.cash / eq * 100, 1) if eq else 0,
            "position_count": self.position_count,
        }

    def reduce_position(self, ticker: str, qty_to_sell: int, price: float) -> Optional[float]:
        """部分減倉 ticker，賣出 qty_to_sell 股，回傳收回資金 (margin + pnl)。"""
        if ticker not in self.holdings:
            return None
        h = self.holdings[ticker]
        qty_to_sell = min(qty_to_sell, h["qty"])
        if qty_to_sell <= 0:
            return None
        ratio = qty_to_sell / h["qty"]
        # 按比例釋放保證金 + 實現損益
        released_margin = h["margin_used"] * ratio
        pnl = (price - h["entry_price"]) * qty_to_sell
        proceeds = released_margin + pnl
        self.cash += proceeds
        print(f"  ✂️ 減倉 {ticker} ×{qty_to_sell} @ {price:,.0f} 回收 NT${proceeds:+,.0f} (損益 {pnl:+,.0f})")
        h["qty"] -= qty_to_sell
        h["margin_used"] -= released_margin
        h["current_value"] = price * h["qty"]
        if h["qty"] <= 0:
            del self.holdings[ticker]
        self.trades.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker, "side": "SELL (風控減倉)", "price": price,
            "qty": qty_to_sell, "pnl": round(pnl, 2),
            "return_pct": round((pnl / released_margin) * 100 if released_margin else 0, 2),
        })
        self.save()
        return proceeds

    def auto_reduce_concentration(self, prices: dict[str, float], target_pct: float = 40.0) -> list[str]:
        """集中度超過 target_pct 時自動減倉最大的標的。回傳風控動作紀錄。"""
        actions: list[str] = []
        if self.position_count < 2:
            return actions
        for _ in range(5):
            max_pct = self.max_concentration
            if max_pct <= target_pct or self.position_count < 2:
                break
            exp = self.total_exposure
            # 找出最大持倉
            ticker = max(self.holdings, key=lambda t: self.holdings[t].get("current_value", self.holdings[t]["entry_price"] * self.holdings[t]["qty"]))
            h = self.holdings[ticker]
            cur_val = h.get("current_value", h["entry_price"] * h["qty"])
            price = prices.get(ticker, h["entry_price"])
            lot_size = 1000 if h.get("market") == "TW" else 1
            # 超標金額
            excess_val = cur_val - (target_pct / 100) * exp
            if excess_val <= 0:
                break
            # 賣超標的 60%，儘量一次到位
            qty_to_sell = max(int(excess_val * 0.6 / price / lot_size) * lot_size, lot_size) if lot_size else max(int(excess_val * 0.6 / price), 1)
            if qty_to_sell >= h["qty"]:
                # 全賣才能解決 → 賣一半
                qty_to_sell = max(h["qty"] // 2, lot_size)
                if qty_to_sell >= h["qty"]:
                    break  # 只有 1 lot 賣不掉
            self.reduce_position(ticker, qty_to_sell, price)
            actions.append(f"集中度 {max_pct:.0f}% > {target_pct:.0f}%，減倉 {ticker} ×{qty_to_sell}")
        return actions

    def check_risk_limits(self, max_exposure_ratio: float = 3.0, max_concentration: float = 40.0) -> list[str]:
        """Check risk limits, return list of warnings."""
        warnings: list[str] = []
        if self.exposure_ratio > max_exposure_ratio:
            warnings.append(f"⚠️ 總曝險 {self.exposure_ratio:.1f}倍 超過上限 {max_exposure_ratio}倍")
        if self.max_concentration > max_concentration:
            warnings.append(f"⚠️ 單一持股集中度 {self.max_concentration:.0f}% 超過上限 {max_concentration}%")
        return warnings

    def summary(self) -> str:
        """回傳目前狀態摘要。"""
        lines = [
            f"\n{'='*50}",
            f"  📋 紙上交 trade 帳戶",
            f"  更新: {self.last_updated}",
            f"{'='*50}",
            f"  初始資金     NT$ {self.initial:>12,.0f}",
            f"  可用現金     NT$ {self.cash:>12,.0f}",
            f"  持倉市值     NT$ {self.holdings_value:>12,.0f}",
            f"  融資借款     NT$ {self.total_loan:>12,.0f}",
            f"  ─────────────────────────────",
            f"  總權益       NT$ {self.total_equity:>12,.0f}",
        ]
        risk = self.risk_summary()
        lines.append(f"  總曝險倍數     {risk['exposure_ratio']:>10.1f}×")
        lines.append(f"  集中度         {risk['max_concentration_pct']:>10.1f}%")
        r = self.return_pct
        color = "+" if r >= 0 else ""
        lines.append(f"  總報酬率         {color}{r:>10.2f}%")
        lines.append(f"  交易次數     {len(self.trades):>12}")

        if self.holdings:
            lines.append(f"\n  📂 目前持倉:")
            for t, h in self.holdings.items():
                loan = h["entry_price"] * h["qty"] * (1 - 1.0 / h["leverage"])
                equity_in_pos = h["current_value"] - loan
                pnl = equity_in_pos - h["margin_used"]
                pnl_r = (pnl / h["margin_used"]) * 100 if h["margin_used"] else 0
                lines.append(
                    f"    {t:<8} {h['qty']:>5}股  @{h['entry_price']:>7,.0f}  "
                    f"槓桿{h['leverage']}×  "
                    f"損益 {pnl:+,.0f} ({pnl_r:+.2f}%)"
                )

        lines.append(f"{'='*50}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Committee 即時分析
# ---------------------------------------------------------------------------

def run_analysis(
    ticker: str,
    name: str = "",
    lookback_days: int = 180,
    fallback: bool = True,
    yahoo_ticker: Optional[str] = None,
) -> Optional[dict]:
    """對單一標的執行 Committee 分析，回傳買賣建議。

    Returns:
        {"ticker", "name", "price", "action", "confidence", "reason"}
        或 None（分析失敗）。
    """
    end = date.today()
    start_date = date.today()
    # 下載足夠的歷史數據
    load_ticker = yahoo_ticker or ticker
    try:
        data = load_single(
            load_ticker,
            start=start_date.replace(year=start_date.year - 1).strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as exc:
        print(f"  ⚠️  {ticker} 數據載入失敗: {exc}")
        return None

    if data.empty:
        return None

    data.attrs["symbol"] = ticker

    # 只取最近 N 天分析
    recent = data.tail(lookback_days)

    # 執行 Agent
    momentum = MomentumAgent()
    value = ValueAgent(fallback_mode=fallback)
    flow = FlowAgent()
    mom_sigs = momentum.generate_signals(recent)
    val_sigs = value.generate_signals(recent)
    flow_sigs = flow.generate_signals(recent)

    agent_signals: dict[str, list[Signal]] = {
        "momentum-agent": mom_sigs,
        "value-agent": val_sigs,
        "flow-agent": flow_sigs,
    }

    weights = {"momentum-agent": 0.50, "value-agent": 0.30, "flow-agent": 0.20}
    cio = CIOAgent(weights=weights)
    consensus = cio.synthesise(agent_signals, recent.index, symbol=ticker)

    current_price = float(recent.iloc[-1]["Close"])

    # 找最近（最後一個）信號
    latest_action = "hold"
    latest_conf = 0.0
    for sig in sorted(consensus, key=lambda s: s.timestamp):
        latest_action = sig.action
        latest_conf = sig.confidence

    # 統計各 Agent 投票
    total = len(mom_sigs) + len(val_sigs) + len(flow_sigs)
    buys = sum(1 for s in mom_sigs + val_sigs + flow_sigs if s.action == "buy")
    sells = sum(1 for s in mom_sigs + val_sigs + flow_sigs if s.action == "sell")

    return {
        "ticker": ticker,
        "name": name or ticker,
        "price": current_price,
        "action": latest_action,
        "confidence": latest_conf,
        "momentum_signals": len(mom_sigs),
        "value_signals": len(val_sigs),
        "total_signals": total,
        "buys": buys,
        "sells": sells,
    }


def print_analysis(result: dict) -> None:
    """顯示分析結果。"""
    action = result["action"]
    conf = result["confidence"]

    if action == "buy":
        action_str = "🟢 BUY"
    elif action == "sell":
        action_str = "🔴 SELL"
    else:
        action_str = "⚪ HOLD"

    print(
        f"  {result['ticker']:<8} {result['name']:<6}"
        f" NT${result['price']:>8,.0f}  "
        f"{action_str}  (信賴度 {conf:.2f})  "
        f"📈{result['buys']} 📉{result['sells']}"
    )


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    """顯示帳戶狀態。"""
    pf = PaperPortfolio()
    print(pf.summary())
    if pf.holdings:
        print("\n  💡 執行 --update 更新股價並取得 new 建議")


def cmd_update(watchlist: list[dict], fallback: bool = True) -> None:
    """更新所有標的數據，執行分析，自動交易。"""
    pf = PaperPortfolio()
    print(f"\n  🔍 Committee 即時分析 ({date.today()})")
    print(f"{'='*50}")

    prices: dict[str, float] = {}
    new_buys: list[dict] = []
    new_sells: list[dict] = []

    for item in watchlist:
        ticker = item["ticker"]
        result = run_analysis(ticker, name=item.get("name", ""), fallback=fallback, yahoo_ticker=item.get("yahoo_ticker"))
        if result is None:
            print(f"  {ticker:<8} ⚠️ 分析失敗")
            continue
        print_analysis(result)
        prices[ticker] = result["price"]
        pf.signals_cache[ticker] = {
            "action": result["action"],
            "confidence": result["confidence"],
            "price": result["price"],
            "buys": result.get("buys", 0),
            "sells": result.get("sells", 0),
            "total": result.get("total_signals", 0),
        }

        # 自動交易邏輯
        if ticker in pf.holdings:
            if result["action"] == "sell":
                new_sells.append((ticker, result["price"]))
        else:
            if result["action"] == "buy" and result["confidence"] >= 0.3:
                new_buys.append((ticker, result["price"], item.get("leverage", 2.0)))

    # 更新持倉價格
    pf.update_prices(prices)

    # 檢查停損停利
    sl_tp_sells = pf.check_sl_tp(prices, watchlist)
    for ticker, price in sl_tp_sells:
        if ticker not in new_sells:
            new_sells.append((ticker, price))

    # 風控檢查
    risk_warnings = pf.check_risk_limits()
    if risk_warnings:
        print(f"\n  🛡️ 風控警示:")
        for w in risk_warnings:
            print(f"    {w}")

    # 執行賣出
    print(f"\n  📤 賣出信號:")
    for ticker, price in new_sells:
        pf.sell(ticker, price)

    # 執行買入（期貨優先，確保保證金足夠）
    futures_tickers = {"TX", "MTX", "FITX", "FIMTX"}
    new_buys.sort(key=lambda x: (0 if x[0] in futures_tickers else 1, x[1]))
    print(f"\n  📥 買入信號:")
    for ticker, price, leverage in new_buys:
        info = next((i for i in watchlist if i["ticker"] == ticker), {})
        pf.buy(ticker, price, leverage, name=info.get("name", ""), market=info.get("market", "TW"), max_qty=info.get("max_qty", 0))

    # 風控：集中度超過 40% 自動減倉
    risk_actions = pf.auto_reduce_concentration(prices, target_pct=40.0)
    if risk_actions:
        print(f"\n  🛡️ 風控自動減倉:")
        for a in risk_actions:
            print(f"    {a}")

    print()
    print(pf.summary())

    # 自動更新儀表板
    try:
        from generate_dashboard import generate as gen_dash
        html = gen_dash()
        Path("dashboard.html").write_text(html, encoding="utf-8")
        print(f"\n  📊 儀表板已更新: dashboard.html")
    except Exception:
        pass

    print(f"\n  💰 可下 auto_trade 參數:")
    for ticker, p in prices.items():
        print(f"     python auto_trade.py --ticker {ticker} --capital {pf.cash:,.0f} --leverage 2.0")


def cmd_watch(watchlist: list[dict], interval: int = 120, fallback: bool = True) -> None:
    """持續監控模式。"""
    print(f"\n  👁️  持續監控中（每 {interval} 秒更新）")
    print(f"  Press Ctrl+C 停止\n")
    try:
        while True:
            cmd_update(watchlist, fallback=fallback)
            for i in range(interval, 0, -1):
                print(f"\r  下次更新: {i}秒  ", end="", flush=True)
                time.sleep(1)
            print()
    except KeyboardInterrupt:
        print("\n\n  👋 監控結束")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="📋 AI 紙上交 trade — Committee 即時分析 + 模擬交易",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--update", action="store_true", help="更新數據 + 執行分析與交易")
    parser.add_argument("--watch", action="store_true", help="持續監控模式")
    parser.add_argument("--interval", type=int, default=120, help="監控間隔（秒）")
    parser.add_argument("--reset", action="store_true", help="重置帳戶")
    parser.add_argument("--no-fallback", action="store_true", help="不使用 fallback（需 API key）")

    args = parser.parse_args()

    if args.reset:
        PaperPortfolio().reset()
        return 0

    if args.watch:
        cmd_watch(DEFAULT_WATCHLIST, interval=args.interval, fallback=not args.no_fallback)
    elif args.update:
        cmd_update(DEFAULT_WATCHLIST, fallback=not args.no_fallback)
    else:
        cmd_status()

    return 0


if __name__ == "__main__":
    sys.exit(main())
