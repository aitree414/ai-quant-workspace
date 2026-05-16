#!/usr/bin/env python3
"""
🔍 AI 選股掃描器 — 對全市場執行 Committee 分析，找出買賣訊號。

用法
----
    python stock_scanner.py                        # 掃描預設台股清單
    python stock_scanner.py --market tw             # 台股
    python stock_scanner.py --market us             # 美股
    python stock_scanner.py --market all            # 全部
    python stock_scanner.py --min-score 0.3         # 只顯示信賴度 >= 0.3
    python stock_scanner.py --top 10                # 只顯示前 10 名
    python stock_scanner.py --html                  # 產出 HTML 報表
    python stock_scanner.py --watch                 # 持續監控模式
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

from strategies.agents import MomentumAgent, ValueAgent, Signal
from strategies.agents.cio_agent import CIOAgent
from utils.data_loader import load_single

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stock_scanner")

REPORT_FILE = Path("scan_report.html")

# 台灣 50 指數成分股 + 熱門股 (ticker)
TW_LIQUID = [
    "2330", "2317", "2454", "2412", "2308", "2382", "2881", "2882",
    "2886", "2885", "2891", "2892", "5880", "1303", "1301", "1326",
    "2002", "2015", "1101", "1216", "2912", "3045", "3008", "3711",
    "4904", "4938", "5876", "2303", "2357", "2379", "3037", "3231",
    "3406", "3443", "3661", "3702", "5274", "6239", "6415", "6669",
    "8046", "9910", "2207", "1590", "1476", "8454", "1125",
]

# 熱門美股
US_POPULAR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "JPM", "V", "MA", "UNH", "HD", "COST", "DIS", "NFLX", "ADBE",
    "CRM", "INTC", "AMD", "PYPL", "QCOM", "TXN", "NKE", "SBUX",
    "BA", "CAT", "GE", "XOM", "CVX", "PFE", "MRK", "ABBV", "LLY",
]

# 預設掃描清單（依市場分類）
SCAN_LISTS = {
    "tw": [{"ticker": t, "market": "TW"} for t in TW_LIQUID],
    "us": [{"ticker": t, "market": "US"} for t in US_POPULAR],
    "all": (
        [{"ticker": t, "market": "TW"} for t in TW_LIQUID]
        + [{"ticker": t, "market": "US"} for t in US_POPULAR]
    ),
}


# ---------------------------------------------------------------------------
# 掃描引擎
# ---------------------------------------------------------------------------

def scan_ticker(
    ticker: str,
    lookback: int = 180,
    fallback: bool = True,
) -> Optional[dict]:
    """對單一標的執行分析，回傳評分結果。"""
    end = date.today()
    try:
        data = load_single(
            ticker,
            start=end.replace(year=end.year - 1).strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as exc:
        return None

    if data.empty:
        return None

    data.attrs["symbol"] = ticker
    recent = data.tail(lookback)

    momentum = MomentumAgent()
    value = ValueAgent(fallback_mode=fallback)
    mom_sigs = momentum.generate_signals(recent)
    val_sigs = value.generate_signals(recent)

    agent_signals = {"momentum-agent": mom_sigs, "value-agent": val_sigs}
    weights = {"momentum-agent": 0.60, "value-agent": 0.40}
    cio = CIOAgent(weights=weights)
    consensus = cio.synthesise(agent_signals, recent.index, symbol=ticker)

    current_price = float(recent.iloc[-1]["Close"])
    latest_action = "hold"
    latest_conf = 0.0
    for sig in sorted(consensus, key=lambda s: s.timestamp):
        latest_action = sig.action
        latest_conf = sig.confidence

    buys = sum(1 for s in mom_sigs + val_sigs if s.action == "buy")
    sells = sum(1 for s in mom_sigs + val_sigs if s.action == "sell")

    return {
        "ticker": ticker,
        "price": current_price,
        "action": latest_action,
        "confidence": latest_conf,
        "buys": buys,
        "sells": sells,
        "momentum_score": len(mom_sigs),
        "value_score": len(val_sigs),
    }


def run_scan(
    watchlist: list[dict],
    min_confidence: float = 0.0,
    top_n: Optional[int] = None,
    fallback: bool = True,
) -> list[dict]:
    """掃描整個 watchlist，回傳排序後的結果。"""
    results: list[dict] = []
    total = len(watchlist)
    print(f"\n  🔍 掃描 {total} 檔標的...\n")

    for i, item in enumerate(watchlist, 1):
        ticker = item["ticker"]
        market = item.get("market", "TW")
        sys.stdout.write(f"\r    [{i}/{total}] {ticker:<6} ... ")
        sys.stdout.flush()

        result = scan_ticker(ticker, fallback=fallback)
        if result is None:
            print(f"\r    [{i}/{total}] {ticker:<6} ⚠️ 失敗")
            continue

        result["market"] = market
        results.append(result)

        action_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}.get(result["action"], "⚪")
        conf = result["confidence"]
        sys.stdout.write(
            f"\r    [{i}/{total}] {ticker:<6} "
            f"{action_icon} {result['action']:<5} "
            f"(信賴度 {conf:.2f})  "
            f"NT${result['price']:>8,.0f}  "
            f"📈{result['buys']} 📉{result['sells']}   "
        )

    print(f"\n  ✅ 掃描完成 ({len(results)}/{total} 成功)\n")

    # 過濾 + 排序：買進信號高 confidence 優先
    results.sort(
        key=lambda r: (
            1 if r["action"] == "buy" else (0 if r["action"] == "sell" else -1),
            r["confidence"],
        ),
        reverse=True,
    )

    if min_confidence > 0:
        results = [r for r in results if r["confidence"] >= min_confidence]

    if top_n is not None:
        results = results[:top_n]

    return results


# ---------------------------------------------------------------------------
# 顯示
# ---------------------------------------------------------------------------

def print_results(results: list[dict]) -> None:
    """在 terminal 顯示掃描結果。"""
    if not results:
        print("  ⚠️ 沒有符合條件的標的")
        return

    buys = [r for r in results if r["action"] == "buy"]
    sells = [r for r in results if r["action"] == "sell"]
    holds = [r for r in results if r["action"] == "hold"]

    if buys:
        print(f"\n  🟢 BUY 信號 ({len(buys)}):")
        print(f"  {'代碼':<6} {'市場':<4} {'信賴度':<8} {'股價':<10} {'📈買':<4} {'📉賣':<4}")
        print(f"  {'-'*40}")
        for r in buys:
            m = r.get("market", "")
            print(f"  {r['ticker']:<6} {m:<4} {r['confidence']:<8.2f} {r['price']:<10,.0f} {r['buys']:<4} {r['sells']:<4}")

    if sells:
        print(f"\n  🔴 SELL 信號 ({len(sells)}):")
        for r in sells:
            m = r.get("market", "")
            print(f"  {r['ticker']:<6} {m:<4} {r['confidence']:<8.2f} {r['price']:<10,.0f}")

    if holds:
        print(f"\n  ⚪ HOLD ({len(holds)} 檔)")

    print(f"\n  📊 總計: {len(buys)} 🟢 / {len(sells)} 🔴 / {len(holds)} ⚪")


# ---------------------------------------------------------------------------
# HTML 報表
# ---------------------------------------------------------------------------

def generate_html_report(results: list[dict]) -> str:
    """產出 HTML 掃描報表。"""
    buys = [r for r in results if r["action"] == "buy"]
    sells = [r for r in results if r["action"] == "sell"]

    rows = ""
    for r in results:
        act = r["action"]
        conf = r["confidence"]
        icon = "🟢" if act == "buy" else ("🔴" if act == "sell" else "⚪")
        color = "#00d4aa" if act == "buy" else ("#ff4757" if act == "sell" else "#889")
        m = r.get("market", "")
        rows += f"""<tr>
            <td><strong>{r['ticker']}</strong></td>
            <td>{m}</td>
            <td>{r['price']:,.0f}</td>
            <td style="color:{color};font-weight:600">{icon} {act.upper()}</td>
            <td style="color:{color}">{conf:.2f}</td>
            <td>{r['buys']}</td>
            <td>{r['sells']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔍 選股掃描報告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#080c18; color:#e0e6f0; padding:24px; max-width:1000px; margin:0 auto; }}
h1 {{ font-size:22px; margin-bottom:4px; background:linear-gradient(135deg,#00d4aa,#00a8ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.sub {{ color:#667; font-size:13px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }}
.card {{ background:#0e1726; border:1px solid #1a2a40; border-radius:10px; padding:14px; }}
.card .lbl {{ font-size:10px; color:#667; text-transform:uppercase; }}
.card .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ color:#667; font-size:10px; text-transform:uppercase; text-align:left; padding:10px 8px; border-bottom:1px solid #1a2a40; }}
td {{ padding:10px 8px; border-bottom:1px solid #111c2e; }}
tr:hover td {{ background:rgba(255,255,255,0.02); }}
.section-title {{ font-size:13px; font-weight:600; margin-bottom:12px; display:flex; align-items:center; gap:8px; }}
.section-title::after {{ content:''; flex:1; height:1px; background:#1a2a40; }}
@media (max-width:640px) {{ .grid {{ grid-template-columns:1fr 1fr; }} body {{ padding:12px; }} }}
</style>
</head>
<body>
<h1>🔍 選股掃描報告</h1>
<div class="sub">{date.today()} | 共 {len(results)} 檔標的</div>

<div class="grid">
    <div class="card"><div class="lbl">🟢 買進訊號</div><div class="val" style="color:#00d4aa">{len(buys)}</div></div>
    <div class="card"><div class="lbl">🔴 賣出訊號</div><div class="val" style="color:#ff4757">{len(sells)}</div></div>
    <div class="card"><div class="lbl">📊 掃描總數</div><div class="val">{len(results)}</div></div>
</div>

<div class="section-title">📋 掃描結果</div>
<table>
<thead><tr><th>代碼</th><th>市場</th><th>股價</th><th>訊號</th><th>信賴度</th><th>📈</th><th>📉</th></tr></thead>
<tbody>{rows or '<tr><td colspan="7" style="text-align:center;color:#556;">無結果</td></tr>'}</tbody>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="🔍 AI 選股掃描器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", choices=["tw", "us", "all"], default="tw", help="掃描市場")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="最低信賴度")
    parser.add_argument("--top", type=int, default=None, help="只顯示前 N 名")
    parser.add_argument("--html", action="store_true", help="產出 HTML 報表")
    parser.add_argument("--no-fallback", action="store_true", help="不使用 fallback")
    parser.add_argument("--watch", action="store_true", help="持續監控（每 30 分鐘）")
    args = parser.parse_args()

    watchlist = SCAN_LISTS[args.market]
    fallback = not args.no_fallback

    if args.watch:
        print(f"\n  👁️  持續掃描中（每 30 分鐘）")
        try:
            while True:
                results = run_scan(watchlist, min_confidence=args.min_confidence, top_n=args.top, fallback=fallback)
                print_results(results)
                for i in range(1800, 0, -1):
                    print(f"\r  下次掃描: {i//60}分{i%60}秒  ", end="", flush=True)
                    time.sleep(1)
                print()
        except KeyboardInterrupt:
            print("\n\n  👋 掃描結束")
        return 0

    results = run_scan(watchlist, min_confidence=args.min_confidence, top_n=args.top, fallback=fallback)
    print_results(results)

    if args.html:
        REPORT_FILE.write_text(generate_html_report(results), encoding="utf-8")
        print(f"\n  📊 報表已儲存: {REPORT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
