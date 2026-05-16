#!/usr/bin/env python3
"""Generate trading dashboard HTML from paper_portfolio.json."""

import json
from pathlib import Path
from datetime import datetime

PORTFOLIO_FILE = Path("paper_portfolio.json")
DASHBOARD_FILE = Path("dashboard.html")


def generate() -> str:
    if not PORTFOLIO_FILE.exists():
        return "<html><body><h1>No trading data yet</h1><p>Run python paper_trade.py --update</p></body></html>"

    data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    cash = data.get("cash", 0)
    initial = data.get("initial", 400000)
    holdings = data.get("holdings", {})
    trades = data.get("trades", [])
    eq_hist = data.get("equity_history", [])
    last_upd = data.get("last_updated", "")

    signals_cache = data.get("signals_cache", {})

    # USD/TWD exchange rate
    USD_TWD = 31.56

    # ---- Separate TW/US holdings & calculate equity ----
    tw_holdings = {}
    us_holdings = {}
    for sym, h in holdings.items():
        mkt = h.get("market", "")
        if mkt == "TW" or (not mkt and (sym.upper().endswith(".TW") or (sym.replace(".","").isdigit() and len(sym.replace(".","")) <= 4))):
            tw_holdings[sym] = h
        else:
            us_holdings[sym] = h

    def _market_equity(holdings_dict: dict) -> tuple[float, float, float]:
        """Returns (total_value, total_loan, equity) for a group of holdings."""
        tv = 0.0
        tl = 0.0
        for h in holdings_dict.values():
            ep = h["entry_price"]
            q = h["qty"]
            lev = h["leverage"]
            cv = h.get("current_value", ep * q)
            tv += cv
            tl += ep * q * (1 - 1.0 / lev)
        return tv, tl, tv - tl

    tw_val, tw_loan, tw_eq = _market_equity(tw_holdings)
    us_val, us_loan, us_eq = _market_equity(us_holdings)

    # Portfolio model treats everything in TWD-equivalent internally
    # (no real FX conversion — US entry_prices stored as-is)
    total_loan = tw_loan + us_loan
    total_value = tw_val + us_val
    equity_combined = cash + tw_eq + us_eq
    cash_twd = cash
    equity_usd_display = us_eq / USD_TWD if us_eq else 0  # approximate USD display only
    equity_twd = cash + tw_eq + us_eq
    ret = ((equity_combined / initial) - 1) * 100 if initial else 0
    ret_color = "#00d4aa" if ret >= 0 else "#ff4757"
    ret_sign = "+" if ret >= 0 else ""
    total_pnl_amount = equity_combined - initial
    total_pnl_color = "#00d4aa" if total_pnl_amount >= 0 else "#ff4757"

    # ---- Chart data ----
    dates_str = json.dumps([e["date"] for e in eq_hist])
    equity_vals_str = json.dumps([e["equity"] for e in eq_hist])

    # ---- Holdings table ----
    def _holding_row(sym: str, h: dict, exchange: str, currency: str) -> str:
        ep = h["entry_price"]
        q = h["qty"]
        lev = h["leverage"]
        cv = h.get("current_value", ep * q)
        lp = ep * (1 - 1.0 / lev)
        eq_in_pos = cv - lp * q
        pnl = eq_in_pos - h.get("margin_used", ep * q / lev)
        pnl_r = (pnl / h.get("margin_used", 1)) * 100
        col = "#00d4aa" if pnl >= 0 else "#ff4757"
        name = h.get("name", sym)
        # Determine investment type
        if sym in ("TX", "MTX", "FITX", "FIMTX"):
            inv_type = "期貨"
            type_color = "#a855f7"
        elif lev > 1:
            inv_type = "融資"
            type_color = "#f97316"
        else:
            inv_type = "現股"
            type_color = "#889"
        return f"""<tr>
            <td><span style="color:#667;font-size:10px">{exchange}</span></td>
            <td><strong>{sym}</strong></td>
            <td><span style="color:#889;font-size:11px">{name}</span></td>
            <td><span style="color:{type_color};font-size:11px;font-weight:600">{inv_type}</span></td>
            <td>{q}</td>
            <td>{currency}{ep:,.0f}</td>
            <td>{currency}{cv/q:,.0f}</td>
            <td style="font-weight:600">{currency}{cv:,.0f}</td>
            <td style="color:{col};font-weight:600">{currency}{pnl:+,.0f}</td>
            <td style="color:{col}">{pnl_r:+.2f}%</td>
            <td style="color:#889">{lev}×</td>
        </tr>"""

    tw_rows = ""
    us_rows = ""
    for sym, h in tw_holdings.items():
        tw_rows += _holding_row(sym, h, "TWSE", "NT$")
    for sym, h in us_holdings.items():
        us_rows += _holding_row(sym, h, "NASDAQ", "US$")

    # Holdings donut data (ticker → weight % of total_value)
    total_val_all = sum(h.get("current_value", h["entry_price"] * h["qty"]) for h in holdings.values()) or 1
    donut_labels = ",".join(f'"{s}"' for s in holdings)
    donut_vals = ",".join(f"{h.get('current_value', h['entry_price'] * h['qty']) / total_val_all * 100:.1f}" for s, h in holdings.items())
    donut_colors_arr = ["#00d4aa", "#00a8ff", "#ffb700", "#ff4757", "#a855f7", "#f97316", "#06b6d4", "#ec4899"]
    donut_colors = json.dumps([donut_colors_arr[i % len(donut_colors_arr)] for i in range(len(holdings))])

    # ---- Trades table ----
    trade_rows = ""
    wins = 0
    losses = 0
    completed_trades = 0
    total_pnl = 0.0
    for t in reversed(trades[-80:]):
        pnl = t.get("pnl", 0)
        r = t.get("return_pct", 0)
        total_pnl += pnl
        if "pnl" in t:
            completed_trades += 1
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1
        col = "#00d4aa" if pnl >= 0 else "#ff4757"
        trade_rows += f"""<tr>
            <td>{t.get('time','—')[:10]}</td>
            <td>{t['ticker']}</td>
            <td>{t['side']}</td>
            <td>{t.get('qty','—')}</td>
            <td>{t.get('price',0):,.0f}</td>
            <td style="color:{col};font-weight:600">NT${pnl:+,.0f}</td>
            <td style="color:{col}">{r:+.2f}%</td>
        </tr>"""

    total_trades = len(trades)
    win_rate = (wins / completed_trades * 100) if completed_trades else 0

    # ---- P&L distribution data ----
    recent_pnls = [t.get("pnl", 0) for t in trades[-40:]]
    pl_max = max(recent_pnls) if recent_pnls else 1
    pl_min = min(recent_pnls) if recent_pnls else -1
    pl_range = max(abs(pl_max), abs(pl_min)) or 1

    # ---- Equity curve enhancement ----
    eq_vals = [e["equity"] for e in eq_hist]
    eq_max = max(eq_vals) if eq_vals else equity_combined
    eq_min = min(eq_vals) if eq_vals else equity_combined

    # ---- Risk metrics ----
    exposure_ratio = total_value / equity_combined if equity_combined else 0
    max_concentration_pct = 0.0
    if total_value:
        max_pos_val = max((h.get("current_value", h["entry_price"] * h["qty"]) for h in holdings.values()), default=0)
        max_concentration_pct = max_pos_val / total_value * 100
    # Margin utilization
    margin_util_pct = total_loan / (total_value + 1) * 100 if total_value else 0
    margin_maintenance_pct = 70.0
    margin_call_pct = 85.0
    margin_util_color = "#00d4aa" if margin_util_pct < margin_maintenance_pct else ("#ffb700" if margin_util_pct < margin_call_pct else "#ff4757")

    # ---- Type counts ----
    futures_tickers = {"TX", "MTX", "FITX", "FIMTX"}
    margin_count = 0
    futures_count = 0
    cash_count = 0
    for sym, h in holdings.items():
        if sym in futures_tickers:
            futures_count += 1
        elif h.get("leverage", 1) > 1:
            margin_count += 1
        else:
            cash_count += 1

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 交易儀表板</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #080c18; color: #e0e6f0; padding: 24px;
    max-width: 1280px; margin: 0 auto;
}}
h1 {{ font-size: 26px; margin-bottom: 2px; }}
h1 .grad {{
    background: linear-gradient(135deg,#00d4aa,#00a8ff,#a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.subtitle {{ color: #667; font-size: 13px; margin-bottom: 20px; }}

/* Grid */
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.card {{
    background: linear-gradient(145deg, #111c2e, #0e1726);
    border: 1px solid #1a2a40; border-radius: 12px; padding: 18px;
    position: relative; overflow: hidden;
}}
.card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 12px 12px 0 0;
}}
.card.ret::before {{ background: linear-gradient(90deg, {ret_color}, transparent); }}
.card.cash::before {{ background: linear-gradient(90deg, #00a8ff, transparent); }}
.card.pos::before {{ background: linear-gradient(90deg, #ffb700, transparent); }}
.card.tw::before {{ background: linear-gradient(90deg, #00d4aa, transparent); }}
.card.us::before {{ background: linear-gradient(90deg, #00a8ff, transparent); }}
.card .lbl {{ font-size: 10px; color: #667; text-transform: uppercase; letter-spacing: .5px; }}
.card .val {{ font-size: 24px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.card .sub {{ font-size: 11px; color: #556; margin-top: 2px; }}

/* Charts */
.chart-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 20px; }}
.chart-box {{
    background: #0e1726; border: 1px solid #1a2a40; border-radius: 12px; padding: 16px;
}}
.chart-box .lbl {{ font-size: 11px; color: #667; margin-bottom: 8px; }}
canvas {{ width: 100%; height: 280px; display: block; }}

/* Tables */
.table-wrap {{ background: #0e1726; border: 1px solid #1a2a40; border-radius: 12px; padding: 16px; }}
.table-scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ color: #667; font-size: 10px; text-transform: uppercase; letter-spacing: .3px; text-align: left; padding: 10px 8px; border-bottom: 1px solid #1a2a40; }}
td {{ padding: 10px 8px; border-bottom: 1px solid #111c2e; font-variant-numeric: tabular-nums; white-space: nowrap; }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
.section-title {{ font-size: 13px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
.section-title::after {{ content: ''; flex: 1; height: 1px; background: #1a2a40; }}
.empty {{ color: #556; text-align: center; padding: 32px; font-size: 13px; }}

/* Buttons & toolbar */
.toolbar {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
.btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 18px; border-radius: 8px; border: none; font-size: 12px;
    font-weight: 600; cursor: pointer; transition: all .2s; text-decoration: none;
    background: #1a2a40; color: #e0e6f0;
}}
.btn:hover {{ background: #243555; transform: translateY(-1px); }}
.btn-primary {{ background: linear-gradient(135deg,#00d4aa,#00b894); color: #080c18; }}
.btn-primary:hover {{ filter: brightness(1.1); }}
.btn-primary:disabled {{ opacity: .4; cursor: wait; filter: none; }}

/* Connection bar */
.connection-bar {{
    display: flex; gap: 16px; font-size: 11px; color: #667;
    padding: 6px 12px; background: #0a1020; border-radius: 8px; margin-bottom: 16px;
}}
.connection-bar .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }}

/* Toast */
.toast {{
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    padding: 12px 28px; border-radius: 10px; font-size: 13px; z-index: 999;
    background: #111c2e; border: 1px solid #1a2a40; color: #e0e6f0;
    opacity: 0; transition: opacity .3s; pointer-events: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}}
.toast.show {{ opacity: 1; }}
.toast.success {{ border-color: #00d4aa; }}
.toast.error {{ border-color: #ff4757; }}

/* Donut legend */
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; }}
.legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

/* Responsive */
@media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
@media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} body {{ padding: 12px; }} .card .val {{ font-size: 18px; }} }}

/* Animations */
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
.card, .chart-box, .table-wrap {{ animation: fadeIn .4s ease-out; }}
</style>
</head>
<body>

<h1><span class="grad">📊 交易儀表板</span></h1>
<div class="subtitle">最後更新: {last_upd}</div>

<div class="connection-bar" id="connBar">
    <span><span class="dot" id="serverDot" style="background:#556;"></span> 伺服器: <span id="serverStatus">檢查中…</span></span>
    <span>📡 <span id="connInfo">http://localhost:8080</span></span>
</div>

<div class="toolbar">
    <button class="btn btn-primary" id="refreshBtn" onclick="refreshData()">🔄 更新</button>
    <button class="btn" onclick="exportCSV()">📥 CSV</button>
    <button class="btn" onclick="exportJSON()">📥 JSON</button>
    <a href="trading_sim.html" class="btn">🎮 模擬器</a>
    <a href="/" class="btn">🏠 儀表板</a>
</div>

<!-- Summary cards -->
<div class="grid">
    <div class="card ret">
        <div class="lbl">💰 總權益（合計）</div>
        <div class="val" style="color:{ret_color}">NT$ {equity_combined:,.0f}</div>
        <div class="sub">初始 NT$ {initial:,.0f} · 匯率 1 USD = {USD_TWD} TWD</div>
    </div>
    <div class="card tw">
        <div class="lbl">🇹🇼 台幣權益</div>
        <div class="val" style="color:#00d4aa">NT$ {equity_twd:,.0f}</div>
        <div class="sub">現金 NT$ {cash_twd:,.0f} + 台股 {tw_eq:,.0f}</div>
    </div>
    <div class="card us">
        <div class="lbl">🇺🇸 美金權益（約）</div>
        <div class="val" style="color:#00a8ff">US$ {equity_usd_display:,.0f}</div>
        <div class="sub">約 NT$ {us_eq:,.0f}</div>
    </div>
    <div class="card ret">
        <div class="lbl">總報酬率</div>
        <div class="val" style="color:{ret_color}">{ret_sign}{ret:.2f}%</div>
        <div class="sub" style="color:{total_pnl_color}">NT$ {total_pnl_amount:+,.0f}</div>
    </div>
    <div class="card cash">
        <div class="lbl">可用現金</div>
        <div class="val" style="color:#00a8ff">NT$ {cash:,.0f}</div>
        <div class="sub">融資借款 NT$ {total_loan:,.0f}</div>
    </div>
    <!-- Margin utilization -->
    <div class="card" style="border-color:#f9731640;">
        <div class="lbl">🔒 保證金使用率</div>
        <div style="margin-top:6px;background:#1a2a40;border-radius:4px;height:6px;overflow:hidden;">
            <div style="width:{min(margin_util_pct, 100):.0f}%;background:{margin_util_color};height:6px;border-radius:4px;transition:width .5s;"></div>
        </div>
        <div class="val" style="font-size:16px;margin-top:4px;color:{margin_util_color};font-variant-numeric:tabular-nums;">{margin_util_pct:.0f}%</div>
        <div class="sub">維持保證金 {margin_maintenance_pct:.0f}% · 追繳線  {margin_call_pct:.0f}%</div>
    </div>
    <div class="card pos">
        <div class="lbl">持倉數量</div>
        <div class="val" style="color:#ffb700">{len(holdings)}</div>
        <div class="sub">{total_trades} 筆交易 · 勝率 {win_rate:.0f}%</div>
    </div>
</div>

<!-- Position type summary -->
<div style="display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap;align-items:center;">
    {_type_badge("融資", margin_count, "#f97316")}
    {_type_badge("期貨", futures_count, "#a855f7")}
    {_type_badge("現股", cash_count, "#889")}
    <span style="color:#556;font-size:11px;">曝險 {exposure_ratio:.1f}×</span>
    <span style="color:#556;font-size:11px;">集中度 {max_concentration_pct:.0f}%</span>
    <span style="color:#556;font-size:11px;">融資借款 NT$ {total_loan:,.0f}</span>
</div>

<!-- Futures signal monitor -->
{futures_monitor(signals_cache)}

<!-- Charts row -->
<div class="chart-row">
    <div class="chart-box">
        <div class="lbl">📈 權益曲線</div>
        <canvas id="eqChart"></canvas>
    </div>
    <div class="chart-box">
        <div class="lbl">🧩 持倉佔比</div>
        <canvas id="donutChart"></canvas>
        <div class="legend" id="donutLegend"></div>
    </div>
</div>

<!-- P&L bar chart -->
<div class="chart-box" style="margin-bottom:20px;">
    <div class="lbl">📊 近期交易損益分布</div>
    <canvas id="plChart" style="height:120px;"></canvas>
</div>

<!-- Holdings & Trades -->
<div class="chart-row">
    <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="table-wrap">
            <div class="section-title">🇹🇼 台股</div>
            {_table(tw_rows, tw_holdings, ['交易所','代碼','名稱','類型','持股','均價','現價','市值','未實現損益','報酬率','槓桿'], "尚無台股持倉")}
        </div>
        <div class="table-wrap">
            <div class="section-title">🇺🇸 美股</div>
            {_table(us_rows, us_holdings, ['交易所','代碼','名稱','類型','持股','均價','現價','市值','未實現損益','報酬率','槓桿'], "尚無美股持倉")}
        </div>
    </div>
    <div class="table-wrap">
        <div class="section-title">📋 近期交易</div>
        {_table(trade_rows, trades, ['日期','商品','方向','數量','價格','損益','報酬'], "尚無交易")}
    </div>
</div>

<div id="toast" class="toast"></div>

<script>
// --- Donut Chart ---
const donutLabels = [{donut_labels}];
const donutVals = [{donut_vals}];
const donutColors = {donut_colors};

(function drawDonut() {{
    const c = document.getElementById('donutChart');
    const ctx = c.getContext('2d');
    const w = c.parentElement.clientWidth, h = 280;
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const cx = w/2, cy = h/2 - 10, r = Math.min(w, h) * 0.28, ir = r * 0.55;
    let a0 = -Math.PI/2;
    const total = donutVals.reduce((a,b)=>a+b, 0) || 1;
    for (let i = 0; i < donutVals.length; i++) {{
        if (donutVals[i] === 0) continue;
        const a = (donutVals[i]/total) * 2 * Math.PI;
        ctx.beginPath(); ctx.arc(cx, cy, r, a0, a0 + a); ctx.arc(cx, cy, ir, a0 + a, a0, true); ctx.closePath();
        ctx.fillStyle = donutColors[i % donutColors.length]; ctx.fill();
        a0 += a;
    }}
    // center text
    ctx.fillStyle = '#e0e6f0'; ctx.font = 'bold 22px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(donutLabels.length + ' 檔', cx, cy - 2);
    ctx.fillStyle = '#667'; ctx.font = '10px system-ui';
    ctx.fillText('持倉佔比', cx, cy + 14);

    // legend
    const lg = document.getElementById('donutLegend'); lg.innerHTML = '';
    for (let i = 0; i < donutLabels.length; i++) {{
        const d = document.createElement('span'); d.className = 'legend-item';
        d.innerHTML = '<span class=legend-dot style=background:'+donutColors[i%donutColors.length]+'></span>'+
            donutLabels[i] + ' ' + donutVals[i].toFixed(1) + '%';
        lg.appendChild(d);
    }}
}})();

// --- Equity Curve ---
(function drawEquity() {{
    const data = [{equity_vals_str}];
    const c = document.getElementById('eqChart');
    const ctx = c.getContext('2d');
    const w = c.parentElement.clientWidth, h = 280;
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0,0,w,h);

    if (data.length < 2) {{
        ctx.fillStyle='#667'; ctx.font='14px system-ui'; ctx.textAlign='center';
        ctx.fillText('等待更多數據…', w/2, h/2); return;
    }}

    const pad = 40, graphW = w - pad*2, graphH = h - pad*2;
    const min = Math.min(...data), max = Math.max(...data), range = (max - min) || 1;
    const up = data[data.length-1] >= data[0];

    // grid lines
    ctx.strokeStyle = '#111c2e'; ctx.lineWidth = 0.5;
    for (let i=0;i<4;i++) {{ const y = pad + graphH/4*i; ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(w-pad,y); ctx.stroke(); }}

    // area fill
    ctx.beginPath();
    for (let i=0;i<data.length;i++) {{
        const x = pad + (i/(data.length-1))*graphW;
        const y = pad + graphH - ((data[i]-min)/range)*graphH;
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }}
    ctx.lineTo(pad+graphW, pad+graphH); ctx.lineTo(pad, pad+graphH); ctx.closePath();
    const grad = ctx.createLinearGradient(0,pad,0,pad+graphH);
    grad.addColorStop(0, up ? 'rgba(0,212,170,0.15)' : 'rgba(255,71,87,0.15)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad; ctx.fill();

    // line
    ctx.beginPath();
    for (let i=0;i<data.length;i++) {{
        const x = pad + (i/(data.length-1))*graphW;
        const y = pad + graphH - ((data[i]-min)/range)*graphH;
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }}
    ctx.strokeStyle = up ? '#00d4aa' : '#ff4757';
    ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.stroke();

    // current value label
    const lastY = pad + graphH - ((data[data.length-1]-min)/range)*graphH;
    ctx.fillStyle = up ? '#00d4aa' : '#ff4757';
    ctx.font = 'bold 13px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('NT$ ' + data[data.length-1].toLocaleString(), pad+graphW+4, lastY+4);

    // min/max labels
    ctx.fillStyle = '#445'; ctx.font = '10px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText('NT$ ' + max.toLocaleString(), pad-4, pad+10);
    ctx.fillText('NT$ ' + min.toLocaleString(), pad-4, pad+graphH-2);
}})();

// --- P&L Bar Chart ---
(function drawPL() {{
    const data = [{json.dumps(recent_pnls)}];
    const c = document.getElementById('plChart');
    const ctx = c.getContext('2d');
    const w = c.parentElement.clientWidth, h = 120;
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0,0,w,h);

    if (data.length < 2) return;
    const pad = 0, bw = Math.max(2, (w - pad*2) / data.length - 1);
    const mx = Math.max(...data.map(Math.abs), 1);
    const cy = h/2;

    for (let i=0;i<data.length;i++) {{
        const x = pad + i * (bw + 1);
        const bh = (Math.abs(data[i]) / mx) * (h/2 - 8);
        ctx.fillStyle = data[i] >= 0 ? '#00d4aa' : '#ff4757';
        ctx.globalAlpha = 0.7;
        ctx.fillRect(x, data[i] >=0 ? cy - bh : cy, bw, bh);
        ctx.globalAlpha = 1;
    }}
    // zero line
    ctx.strokeStyle = '#1a2a40'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(w,cy); ctx.stroke();
}})();

// --- Server status ---
(async function() {{
    try {{
        const r = await fetch('/api/status');
        if (r.ok) {{
            document.getElementById('serverDot').style.background = '#00d4aa';
            document.getElementById('serverStatus').textContent = '已連線';
            document.getElementById('connInfo').textContent = window.location.href;
        }}
    }} catch(e) {{
        document.getElementById('serverDot').style.background = '#ff4757';
        document.getElementById('serverStatus').textContent = '離線（本機模式）';
        document.getElementById('connInfo').textContent = '僅供本機檢視';
    }}
}})();

// --- Refresh ---
async function refreshData() {{
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true; btn.textContent = '⏳ 更新中…';
    try {{
        const r = await fetch('/api/update', {{ method: 'POST' }});
        showToast('✅ 數據已更新，請重新整理頁面', 'success');
        setTimeout(() => location.reload(), 1500);
    }} catch(e) {{
        showToast('❌ 更新失敗: 請執行 python paper_trade.py --update', 'error');
        btn.disabled = false; btn.textContent = '🔄 更新';
    }}
}}
function showToast(msg, type) {{
    const t = document.getElementById('toast');
    t.textContent = msg; t.className = 'toast show ' + type;
    clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('show'), 3000);
}}

// --- Export ---
async function fetchPortfolio() {{
    const r = await fetch('/api/status');
    return await r.json();
}}

async function exportJSON() {{
    try {{
        const data = await fetchPortfolio();
        const blob = new Blob([JSON.stringify(data, null, 2)], {{type:'application/json'}});
        downloadBlob(blob, 'portfolio_export.json');
        showToast('✅ JSON 已下載', 'success');
    }} catch(e) {{
        showToast('❌ 匯出失敗: ' + e.message, 'error');
    }}
}}

async function exportCSV() {{
    try {{
        const data = await fetchPortfolio();
        let csv = '\\uFEFF';
        csv += '=== 持倉 ===\\n';
        csv += '代碼,名稱,交易所,持股,均價,現價,市值,槓桿,未實現損益,報酬率%\\n';
        for (const [sym, h] of Object.entries(data.holdings || {{}})) {{
            const ep = h.entry_price, q = h.qty, cv = h.current_value || ep * q;
            const loan = ep * q * (1 - 1/h.leverage);
            const eqPos = cv - loan;
            const pnl = eqPos - (h.margin_used || ep * q / h.leverage);
            const pnlR = pnl / (h.margin_used || 1) * 100;
            csv += `${{sym}},${{h.name || sym}},,${{q}},${{ep}},${{(cv/q).toFixed(0)}},${{cv.toFixed(0)}},${{h.leverage}}x,${{pnl.toFixed(0)}},${{pnlR.toFixed(2)}}\\n`;
        }}
        csv += '\\n=== 交易紀錄 ===\\n';
        csv += '時間,代碼,方向,數量,價格,損益,報酬率%\\n';
        for (const t of data.trades || []) {{
            csv += `${{t.time || ''}},${{t.ticker}},${{t.side}},${{t.qty || ''}},${{t.price || 0}},${{t.pnl || 0}},${{t.return_pct || 0}}\\n`;
        }}
        csv += '\\n=== 帳戶摘要 ===\\n';
        csv += `初始資金,NT$${{data.initial}}\\n`;
        csv += `可用現金,NT$${{data.cash}}\\n`;
        csv += `持倉數,${{Object.keys(data.holdings || {{}}).length}}\\n`;
        csv += `交易次數,${{(data.trades || []).length}}\\n`;
        csv += `最後更新,${{data.last_updated}}\\n`;
        const blob = new Blob([csv], {{type:'text/csv;charset=utf-8'}});
        downloadBlob(blob, 'portfolio_export.csv');
        showToast('✅ CSV 已下載', 'success');
    }} catch(e) {{
        showToast('❌ 匯出失敗: ' + e.message, 'error');
    }}
}}

function downloadBlob(blob, name) {{
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}
</script>
</body>
</html>"""


def _type_badge(label: str, count: int, color: str) -> str:
    dim = count == 0
    bg = f"{color}08" if dim else f"{color}15"
    txt = f"{color}60" if dim else color
    border = f"{color}15" if dim else f"{color}30"
    return f"""<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:6px;font-size:11px;font-weight:600;background:{bg};color:{txt};border:1px solid {border};">
    <span style="font-size:14px;">{count}</span> {label}
</span>"""


def futures_monitor(signals_cache: dict) -> str:
    """Show futures analysis signal from cache, even when no position held."""
    tx = signals_cache.get("TX", {})
    if not tx:
        return """<div style="background:#0e1726;border:1px solid #1a2a40;border-radius:10px;padding:12px 16px;margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:16px;">🔄</span>
                <span style="color:#667;font-size:12px;">期貨信號: 執行 --update 以取得分析</span>
            </div>
        </div>"""

    action = tx.get("action", "hold")
    conf = tx.get("confidence", 0)
    price = tx.get("price", 0)
    buys = tx.get("buys", 0)
    sells = tx.get("sells", 0)

    if action == "buy":
        icon, act_color = "🟢", "#00d4aa"
    elif action == "sell":
        icon, act_color = "🔴", "#ff4757"
    else:
        icon, act_color = "⚪", "#889"

    holding = ""  # signal strength bar
    strength_pct = min(conf * 100, 100)
    bar_color = "#00d4aa" if action == "buy" else ("#ff4757" if action == "sell" else "#889")

    return f"""<div style="background:#0e1726;border:1px solid #a855f740;border-radius:10px;padding:12px 16px;margin-bottom:20px;">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-size:16px;font-weight:600;color:#a855f7;">📡 期貨信號</span>
        <span style="color:#667;font-size:12px;">TX 台指期</span>
        <span style="font-size:12px;color:#889;">NT${price:,.0f}</span>
        <span style="color:{act_color};font-weight:600;font-size:13px;">{icon} {action.upper()}</span>
        <span style="color:{bar_color};font-size:12px;">信賴度 {conf:.2f}</span>
        <div style="flex:1;min-width:80px;max-width:140px;background:#1a2a40;border-radius:4px;height:4px;overflow:hidden;">
            <div style="width:{strength_pct:.0f}%;background:{bar_color};height:4px;border-radius:4px;"></div>
        </div>
        <span style="color:#667;font-size:11px;">📈{buys} 📉{sells}</span>
        <span style="color:#667;font-size:11px;">槓桿 10×</span>
        <span style="color:#667;font-size:11px;">保證金 NT${price * 0.1:,.0f}</span>
        <span style="font-size:11px;color:#a855f760;">| 未持倉</span>
    </div>
</div>"""


def _table(rows, items, headers: list, empty_text: str) -> str:
    if not items:
        return f'<div class="empty">{empty_text}</div>'
    cols = "</th><th>".join(headers)
    return f"""<div class="table-scroll"><table><thead><tr><th>{cols}</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def main() -> None:
    html = generate()
    DASHBOARD_FILE.write_text(html, encoding="utf-8")
    print(f"  📊 Dashboard updated: {DASHBOARD_FILE}")


if __name__ == "__main__":
    main()
