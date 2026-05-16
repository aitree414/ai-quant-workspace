#!/usr/bin/env python3
"""
AI 自動投資系統 — 由 Investment Committee 驅動的模擬交易。

用法
----
    python auto_trade.py --ticker 2330 --start 2025-01-01 --capital 400000
    python auto_trade.py --ticker AAPL --capital 400000 --leverage 2.5
    python auto_trade.py --ticker TX  --capital 400000 --future
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategies.committee import run_committee, BacktestResult
from strategies.agents import Signal

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("auto_trade")


# ---------------------------------------------------------------------------
# 模擬交易引擎
# ---------------------------------------------------------------------------

class SimulatedAccount:
    """模擬交易帳戶 — 支援現股、融資、期貨。

    Parameters
    ----------
    initial_capital:
        初始資金（台幣）。
    leverage:
        槓桿倍數 (1 = 現股, 2.5 = 融資, 10 = 期貨)。
    future_mode:
        True 表示使用期貨計算方式（點數 × 每點價格）。
    future_multiplier:
        每點價值（TX=200, MTX=50）。
    commission:
        手續費率。
    """

    def __init__(
        self,
        initial_capital: float = 400_000,
        leverage: float = 1.0,
        future_mode: bool = False,
        future_multiplier: int = 200,
        commission: float = 0.001425,
    ) -> None:
        self.initial = initial_capital
        self.cash = initial_capital
        self.leverage = leverage
        self.future_mode = future_mode
        self.future_mult = future_multiplier
        self.commission = commission

        self.position = 0  # +qty long, -qty short, 0 flat
        self.entry_price = 0.0
        self.trades: list[dict] = []
        self.equity_curve: list[float] = [initial_capital]
        self.dates: list[str] = []

    def execute_signal(self, action: str, price: float, timestamp: str) -> None:
        """根據 CIO 信號執行交易。

        action: "buy" (進場), "sell" (出場)
        """
        if action == "buy" and self.position == 0:
            # 進場
            cost = self._calc_cost(price)
            qty = max(1, int(self.cash / cost))
            self.position = qty
            self.entry_price = price
            used = cost * qty
            self.cash -= used
            self.trades.append({
                "time": timestamp, "side": "BUY", "price": price,
                "qty": qty, "used": used,
            })

        elif action == "sell" and self.position > 0:
            # 出場
            proceeds = self._calc_proceeds(price)
            pnl = self._calc_pnl(price)
            self.cash += proceeds
            self.trades[-1].update({
                "exit_time": timestamp,
                "exit_price": price,
                "pnl": pnl,
                "return_pct": (pnl / self.trades[-1]["used"]) * 100,
            })
            self.position = 0
            self.entry_price = 0.0

    def mark_to_market(self, price: float, timestamp: str) -> None:
        """每日市價結算，記錄權益曲線。

        有倉位時:  equity = 初始資金 + 未實現損益
        無倉位時:  equity = cash（正確反映已實現損益後的資金）
        """
        if self.position > 0:
            # 總權益 = 初始資金 + (現價 - 進場價) × 數量
            equity = self.initial + self._calc_pnl(price)
        else:
            equity = self.cash
        # 確保 equity 不會負債（除非是期貨）
        if not self.future_mode:
            equity = max(equity, 0)
        self.equity_curve.append(equity)
        self.dates.append(timestamp)

    def _calc_cost(self, price: float) -> float:
        if self.future_mode:
            return price * self.future_mult * (1 / self.leverage)
        return price * (1 / self.leverage)

    def _calc_proceeds(self, price: float) -> float:
        """返還保證金 + 損益（使用進場價計算保證金）。"""
        initial_margin = self.entry_price * self.position
        if self.future_mode:
            initial_margin *= self.future_mult * (1 / self.leverage)
        else:
            initial_margin *= (1 / self.leverage)
        return initial_margin + self._calc_pnl(price)

    def _calc_pnl(self, price: float) -> float:
        if self.future_mode:
            return (price - self.entry_price) * self.future_mult * self.position
        return (price - self.entry_price) * self.position

    @property
    def total_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial

    @property
    def total_return_pct(self) -> float:
        return ((self.total_equity / self.initial) - 1) * 100

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity_curve)
        dd = ((self.equity_curve - peak) / peak).min()
        return float(dd * 100)

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 5:
            return 0.0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def summary(self) -> dict:
        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        return {
            "initial": self.initial,
            "final": round(self.total_equity, 2),
            "return_pct": round(self.total_return_pct, 2),
            "max_dd_pct": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 4),
            "total_trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1) if self.trades else 0,
        }


# ---------------------------------------------------------------------------
# 模擬器主流程
# ---------------------------------------------------------------------------

def run_auto_trade(
    ticker: str = "2330",
    start: str = "2025-01-01",
    end: Optional[str] = None,
    capital: float = 400_000,
    leverage: float = 1.0,
    future_mode: bool = False,
    future_multiplier: int = 200,
) -> SimulatedAccount:
    """由 Investment Committee 驅動的全自動模擬交易。

    流程：
      1. Committee 分析歷史數據 → 產生 buy/sell 信號
      2. 模擬帳戶依序執行每個信號
      3. 每日市價結算 → 權益曲線
    """
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    # --- 載入數據 ---
    from utils.data_loader import load_single
    from strategies.agents import MomentumAgent, ValueAgent
    from strategies.agents.cio_agent import CIOAgent

    data = load_single(ticker, start=start, end=end, interval="1d")
    if data.empty:
        logger.error("No data for %s", ticker)
        raise SystemExit(1)
    data.attrs["symbol"] = ticker
    logger.info("Loaded %d bars for %s", len(data), ticker)

    # 產生 Agent 信號
    momentum = MomentumAgent()
    value = ValueAgent(fallback_mode=True)
    mom_sigs = momentum.generate_signals(data)
    val_sigs = value.generate_signals(data)

    agent_signals = {"momentum-agent": mom_sigs, "value-agent": val_sigs}

    weights = {"momentum-agent": 0.60, "value-agent": 0.40}
    cio = CIOAgent(weights=weights)
    consensus = cio.synthesise(agent_signals, data.index, symbol=ticker)

    # 建立帳戶並執行交易
    account = SimulatedAccount(
        initial_capital=capital,
        leverage=leverage,
        future_mode=future_mode,
        future_multiplier=future_multiplier,
    )

    # 正規化 CIO 信號 timestamp（只取日期部分）
    sorted_sigs = sorted(consensus, key=lambda s: s.timestamp[:10])
    # 建立日期 → 動作 的查詢表
    signal_map: dict[str, str] = {}
    for sig in sorted_sigs:
        day = sig.timestamp[:10]
        signal_map[day] = sig.action

    in_position = False

    for ts in data.index:
        day = str(ts.date())
        price = float(data.loc[ts, "Close"])

        action = signal_map.get(day)
        if action == "buy" and not in_position:
            account.execute_signal("buy", price, day)
            in_position = True
        elif action == "sell" and in_position:
            account.execute_signal("sell", price, day)
            in_position = False

        account.mark_to_market(price, day)

    # 若收盤仍持倉，強制平倉
    if account.position > 0:
        last_price = float(data.iloc[-1]["Close"])
        account.execute_signal("sell", last_price, str(data.index[-1].date()))
        account.mark_to_market(last_price, str(data.index[-1].date()))

    return account


# ---------------------------------------------------------------------------
# HTML 報表
# ---------------------------------------------------------------------------

def generate_html_report(
    account: SimulatedAccount,
    ticker: str,
    start: str,
    end: str,
) -> str:
    """產出 HTML 績效報表。"""
    s = account.summary()
    pnl_color = "#00d4aa" if s["return_pct"] >= 0 else "#ff4757"

    # 交易明細表
    rows = ""
    for t in account.trades:
        pnl = t.get("pnl", 0)
        ret = t.get("return_pct", 0)
        color = "#00d4aa" if pnl >= 0 else "#ff4757"
        rows += f"""<tr>
            <td>{t['time']}</td>
            <td>{t.get('exit_time', '—')}</td>
            <td>BUY</td>
            <td>{t['qty']}</td>
            <td>{t['price']:,.0f}</td>
            <td>{t.get('exit_price', 0):,.0f}</td>
            <td style="color:{color}">{pnl:+,.0f}</td>
            <td style="color:{color}">{ret:+.2f}%</td>
        </tr>"""

    # 權益曲線 JS 資料
    eq_data = ",".join(f"{v:.0f}" for v in account.equity_curve)

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8"><title>AI 自動投資報告 — {ticker}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0a0e17; color:#e0e6f0; padding:24px; }}
h1 {{ font-size:24px; margin-bottom:6px; background:linear-gradient(135deg,#00d4aa,#00a8ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.sub {{ color:#667; font-size:13px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:24px; }}
.card {{ background:#121a2a; border:1px solid #1e2a3a; border-radius:10px; padding:14px; }}
.card .lbl {{ font-size:11px; color:#667; text-transform:uppercase; }}
.card .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
.chart-box {{ background:#121a2a; border:1px solid #1e2a3a; border-radius:10px; padding:16px; margin-bottom:24px; }}
.chart-box .lbl {{ font-size:11px; color:#667; margin-bottom:8px; }}
canvas {{ width:100%; height:300px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ color:#667; font-size:10px; text-transform:uppercase; text-align:left; padding:8px 6px; border-bottom:1px solid #1e2a3a; }}
td {{ padding:8px 6px; border-bottom:1px solid #152030; font-variant-numeric:tabular-nums; }}
tr:hover td {{ background:#121a2a66; }}
.green {{ color:#00d4aa; }}
.red {{ color:#ff4757; }}
</style>
</head>
<body>
<h1>🤖 AI 自動投資報告</h1>
<div class="sub">{ticker} | {start} → {end} | 初始 NT$ {s['initial']:,.0f}</div>

<div class="grid">
    <div class="card"><div class="lbl">最終權益</div><div class="val" style="color:{pnl_color}">NT$ {s['final']:,.0f}</div></div>
    <div class="card"><div class="lbl">總報酬率</div><div class="val" style="color:{pnl_color}">{s['return_pct']:+.2f}%</div></div>
    <div class="card"><div class="lbl">最大回撤</div><div class="val red">{s['max_dd_pct']:.2f}%</div></div>
    <div class="card"><div class="lbl">Sharpe</div><div class="val" style="color:#00a8ff">{s['sharpe']}</div></div>
    <div class="card"><div class="lbl">交易次數</div><div class="val">{s['total_trades']}</div></div>
    <div class="card"><div class="lbl">勝率</div><div class="val gold">{s['win_rate']}%</div></div>
</div>

<div class="chart-box">
    <div class="lbl">📈 權益曲線</div>
    <canvas id="chart"></canvas>
</div>

<h2 style="font-size:16px;margin-bottom:10px;">📋 交易明細</h2>
<table>
<thead><tr>
    <th>進場</th><th>出場</th><th>方向</th><th>數量</th><th>進場價</th><th>出場價</th><th>損益</th><th>報酬</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<script>
const data = [{eq_data}];
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
canvas.width = canvas.parentElement.clientWidth;
canvas.height = 300;
const w = canvas.width, h = canvas.height;
const min = Math.min(...data) * 0.95, max = Math.max(...data) * 1.05, r = max - min || 1;
ctx.clearRect(0,0,w,h);
ctx.beginPath();
const up = data[data.length-1] >= data[0];
ctx.strokeStyle = up ? '#00d4aa' : '#ff4757';
ctx.lineWidth = 2;
for(let i=0;i<data.length;i++) {{
    const x = (i/(data.length-1))*w, y = h-((data[i]-min)/r)*(h-32)-16;
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
}}
ctx.stroke();
ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath();
const g = ctx.createLinearGradient(0,0,0,h);
g.addColorStop(0, up ? 'rgba(0,212,170,0.12)' : 'rgba(255,71,87,0.12)');
g.addColorStop(1, 'rgba(0,0,0,0)');
ctx.fillStyle = g; ctx.fill();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI 自動投資系統 — Committee 驅動的模擬交易",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="2330", help="股票代號")
    parser.add_argument("--start", default="2025-01-01", help="起始日")
    parser.add_argument("--end", default=None, help="結束日")
    parser.add_argument("--capital", type=float, default=400_000, help="初始資金")
    parser.add_argument("--leverage", type=float, default=1.0, help="槓桿倍數")
    parser.add_argument("--future", action="store_true", help="期貨模式")
    parser.add_argument("--multiplier", type=int, default=200, help="期貨每點價值")
    parser.add_argument("--no-report", action="store_true", help="不產出 HTML 報表")
    args = parser.parse_args()

    end = args.end or date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"  AI 自動投資系統")
    print(f"  {args.ticker}  {args.start} → {end}")
    print(f"  資金: NT$ {args.capital:,.0f}  |  槓桿: {args.leverage}×  {'(期貨)' if args.future else '(股票)'}")
    print(f"{'='*50}\n")

    print("  ⏳ Committee 分析中...")
    account = run_auto_trade(
        ticker=args.ticker,
        start=args.start,
        end=end,
        capital=args.capital,
        leverage=args.leverage,
        future_mode=args.future,
        future_multiplier=args.multiplier,
    )

    s = account.summary()
    print(f"  ✅ 交易完成\n")
    print(f"  {'初始資金':<12} NT$ {s['initial']:>12,.0f}")
    print(f"  {'最終權益':<12} NT$ {s['final']:>12,.0f}")
    print(f"  {'總報酬率':<12} {s['return_pct']:>+11.2f}%")
    print(f"  {'最大回撤':<12} {s['max_dd_pct']:>11.2f}%")
    print(f"  {'Sharpe':<12} {s['sharpe']:>11.4f}")
    print(f"  {'交易次數':<12} {s['total_trades']:>11}")
    print(f"  {'勝率':<12} {s['win_rate']:>10.1f}%")
    print()

    if not args.no_report:
        report = generate_html_report(account, args.ticker, args.start, end)
        path = Path(f"auto_trade_{args.ticker}_{args.start}_{end}.html")
        path.write_text(report, encoding="utf-8")
        print(f"  📊 報表已儲存: {path}")
        os.system(f"open {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
