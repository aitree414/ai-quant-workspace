#!/usr/bin/env python3
"""
🌐 交易儀表板伺服器 — 從任何地方查看你的投資組合。

用法
----
    python server.py                    # 啟動（本機 http://localhost:8080）
    python server.py --port 5000        # 自訂埠號
    python server.py --public           # 啟用 ngrok 公開網址

遠端存取方式
-----------
1. Tailscale（推薦）：最安全，安裝後用 tailscale IP 連線
   https://tailscale.com

2. ngrok：臨時公開網址，適合測試
   安裝後執行：ngrok http 8080

3. 部署到雲端伺服器（VPS/Heroku/Railway）
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("server")

PORTFOLIO_FILE = Path("paper_portfolio.json")
DASHBOARD_FILE = Path("dashboard.html")
US_DASHBOARD_FILE = Path("us_dashboard.html")
OPTIONS_PORTFOLIO_FILE = Path("options_portfolio.json")
OPTIONS_DASHBOARD_FILE = Path("options_dashboard.html")
TAIWAN_DASHBOARD_FILE = Path("taiwan_dashboard.html")

# Auto-trader instance (set by main())
auto_trader: Optional["AutoTrader"] = None
taiwan_trader: Optional["TaiwanAutoTrader"] = None

# ---------------------------------------------------------------------------
# 簡單 HTTP 伺服器（不需 Flask）
# ---------------------------------------------------------------------------

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    logger.error("需要 Python 標準庫 http.server")
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    """處理所有 HTTP 請求。"""

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"

        if path == "/":
            self._serve_file(TAIWAN_DASHBOARD_FILE, "text/html; charset=utf-8")
        elif path == "/us-dashboard":
            self._serve_file(US_DASHBOARD_FILE, "text/html; charset=utf-8")
        elif path == "/options":
            self._serve_file(OPTIONS_DASHBOARD_FILE, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._serve_json(PORTFOLIO_FILE)
        elif path == "/api/update":
            self._run_update()
        elif path.startswith("/api/history"):
            self._serve_history()
        elif path == "/api/options/status":
            self._serve_options_status()
        elif path == "/api/options/analyze":
            self._run_options_analysis()
        elif path == "/api/options/votes":
            self._serve_options_votes()
        elif path == "/api/options/history":
            self._serve_options_history()
        elif path == "/api/autotrade/status":
            self._serve_autotrade_status()
        elif path == "/api/autotrade/log":
            self._serve_autotrade_log()
        elif path == "/api/taiwan/status":
            self._serve_taiwan_status()
        elif path == "/api/taiwan/log":
            self._serve_taiwan_log()
        elif path == "/api/taiwan/portfolio":
            self._serve_taiwan_portfolio()
        elif path == "/api/taiwan/history":
            self._serve_taiwan_history()
        elif path == "/api/taiwan/chain":
            self._serve_taiwan_chain()
        elif path == "/api/taiwan/txo-history":
            self._serve_taiwan_txo_history()
        elif path == "/api/coordinator/status":
            self._serve_coordinator_status()
        elif path == "/api/coordinator/actions":
            self._serve_coordinator_actions()
        elif path == "/api/risk/status":
            self._serve_risk_status()
        elif path == "/research":
            self._serve_file(Path("research_dashboard.html"), "text/html; charset=utf-8")
        elif path.startswith("/api/research/predictions"):
            self._serve_research_predictions()
        elif path.startswith("/api/research/analytics/"):
            self._serve_research_analytics()
        elif path.startswith("/api/research/sensitivity/"):
            self._serve_research_sensitivity()
        elif path.startswith("/api/research/recommendations"):
            self._serve_research_recommendations()
        elif path == "/api/research/stats":
            self._serve_research_stats()
        else:
            # 嘗試當作靜態檔案提供
            file_path = Path(self.path.lstrip("/"))
            if file_path.exists() and file_path.is_file():
                mime = "text/html; charset=utf-8" if file_path.suffix in (".html", ".htm") else "application/octet-stream"
                self._serve_file(file_path, mime)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error":"not found"}')

    def do_POST(self) -> None:
        if self.path == "/api/update":
            self._run_update()
        elif self.path == "/api/options/analyze":
            self._run_options_analysis()
        elif self.path == "/api/autotrade/start":
            self._post_autotrade_start()
        elif self.path == "/api/autotrade/stop":
            self._post_autotrade_stop()
        elif self.path == "/api/autotrade/pause":
            self._post_autotrade_pause()
        elif self.path == "/api/autotrade/resume":
            self._post_autotrade_resume()
        elif self.path == "/api/autotrade/interval":
            self._post_autotrade_interval()
        elif self.path == "/api/risk/reset-breaker":
            self._post_risk_reset_breaker()
        elif self.path == "/api/taiwan/start":
            self._post_taiwan_start()
        elif self.path == "/api/taiwan/stop":
            self._post_taiwan_stop()
        elif self.path == "/api/taiwan/pause":
            self._post_taiwan_pause()
        elif self.path == "/api/taiwan/resume":
            self._post_taiwan_resume()
        elif self.path == "/api/taiwan/transfer-reserve":
            self._post_taiwan_transfer_reserve()
        elif self.path == "/api/research/backfill":
            self._post_research_backfill()
        else:
            self.send_response(404)
            self.end_headers()

    # ---- 內部方法 ----

    def _serve_file(self, path: Path, mime: str) -> None:
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def _serve_json(self, path: Path) -> None:
        if not path.exists():
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "empty",
                "message": "尚無交易資料，請執行 python paper_trade.py --update",
            }).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def _serve_history(self) -> None:
        """回傳權益曲線歷史（JSON 陣列）。"""
        if not PORTFOLIO_FILE.exists():
            self._serve_json(PORTFOLIO_FILE)
            return
        try:
            data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
            history = data.get("equity_history", [])
        except Exception:
            history = []
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(history).encode())

    def _run_update(self) -> None:
        """Execute paper_trade.py --update and return results."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        lock_held = False
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            Handler.auto_trader.file_lock.acquire()
            lock_held = True
        try:
            result = subprocess.run(
                [sys.executable, "paper_trade.py", "--update"],
                capture_output=True, text=True, timeout=120,
            )
            self.wfile.write(result.stdout.encode("utf-8"))
            if result.stderr:
                self.wfile.write(("\nSTDERR:\n" + result.stderr).encode("utf-8"))
        except subprocess.TimeoutExpired:
            self.wfile.write("更新超時（超過 120 秒）".encode("utf-8"))
        except Exception as exc:
            self.wfile.write(f"更新失敗: {exc}".encode("utf-8"))
        finally:
            if lock_held:
                Handler.auto_trader.file_lock.release()

    # ---- Options endpoints ----

    def _serve_options_status(self) -> None:
        """Return options portfolio status as JSON."""
        try:
            from strategies.options_agents import OptionsPortfolio
            pf = OptionsPortfolio()
            data = pf.get_summary()
            # Add positions
            data["positions"] = [
                {k: v for k, v in p.items() if k != "pnl"}
                for p in pf.positions if p["status"] == "open"
            ]
            self._send_json(data)
        except Exception as e:
            logger.warning(f"  Options status error: {e}")
            self._send_json({"status": "error", "message": str(e)})

    def _run_options_analysis(self) -> None:
        """Run committee analysis and return results."""
        lock_held = False
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            Handler.auto_trader.file_lock.acquire()
            lock_held = True
        try:
            from strategies.options_agents import OptionsCommittee, OptionsPortfolio
            pf = OptionsPortfolio()
            committee = OptionsCommittee()
            results = committee.run_analysis(portfolio=pf.get_state())
            summary = committee.get_voting_summary()
            pf.record_equity()
            pf.save()
            self._send_json({
                "status": "ok",
                "symbols_analyzed": len(summary),
                "votes": summary,
            })
        except Exception as e:
            logger.warning(f"  Options analysis error: {e}")
            self._send_json({"status": "error", "message": str(e)})
        finally:
            if lock_held:
                Handler.auto_trader.file_lock.release()

    def _serve_options_votes(self) -> None:
        """Return latest committee voting results."""
        import json
        if OPTIONS_PORTFOLIO_FILE.exists():
            try:
                data = json.loads(OPTIONS_PORTFOLIO_FILE.read_text(encoding="utf-8"))
                log = data.get("committee_log", [])
                self._send_json({"votes": log[-20:]})
            except Exception:
                self._send_json({"votes": []})
        else:
            self._send_json({"votes": []})

    def _serve_options_history(self) -> None:
        """Return options equity history as JSON."""
        import json
        if OPTIONS_PORTFOLIO_FILE.exists():
            try:
                data = json.loads(OPTIONS_PORTFOLIO_FILE.read_text(encoding="utf-8"))
                history = data.get("equity_history", [])
                self._send_json(history)
            except Exception:
                self._send_json([])
        else:
            self._send_json([])

    # ---- Auto-trader endpoints ----

    def _serve_autotrade_status(self) -> None:
        """Return auto-trader status."""
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            self._send_json(Handler.auto_trader.get_status())
        else:
            self._send_json({"running": False, "error": "Auto-trader not initialized"})

    def _serve_autotrade_log(self) -> None:
        """Return auto-trader activity log."""
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            self._send_json({"entries": Handler.auto_trader.get_log(50)})
        else:
            self._send_json({"entries": []})

    def _post_autotrade_start(self) -> None:
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            ok = Handler.auto_trader.start()
            self._send_json({"status": "ok" if ok else "already_running"})
        else:
            self._send_json({"status": "error", "message": "Auto-trader not initialized"})

    def _post_autotrade_stop(self) -> None:
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            Handler.auto_trader.stop(wait=False)
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_autotrade_pause(self) -> None:
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            Handler.auto_trader.pause()
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_autotrade_resume(self) -> None:
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            Handler.auto_trader.resume()
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_autotrade_interval(self) -> None:
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                body = self.rfile.read(length).decode()
                import json
                data = json.loads(body)
                interval = int(data.get("interval", 300))
                Handler.auto_trader.set_interval(interval)
                self._send_json({"status": "ok", "interval": interval})
            else:
                self._send_json({"status": "error", "message": "no body"})
        else:
            self._send_json({"status": "error"})

    def _send_json(self, data) -> None:
        """Send JSON response."""
        import json
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ---- Taiwan trader endpoints ----

    def _serve_taiwan_status(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            self._send_json(Handler.taiwan_trader.get_status())
        else:
            self._send_json({"running": False, "market": "taiwan", "error": "Taiwan trader not initialized"})

    def _serve_taiwan_log(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            self._send_json({"entries": Handler.taiwan_trader.get_log(50)})
        else:
            self._send_json({"entries": []})

    def _serve_taiwan_portfolio(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            bridge = Handler.taiwan_trader.get_bridge()
            self._send_json(bridge.get_account_summary())
        else:
            self._send_json({"error": "Taiwan trader not initialized"})

    def _serve_taiwan_history(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            bridge = Handler.taiwan_trader.get_bridge()
            self._send_json(bridge.get_equity_history())
        else:
            self._send_json([])

    def _serve_taiwan_chain(self) -> None:
        """Return TXO option chain data."""
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            bridge = Handler.taiwan_trader.get_bridge()
            taiex = bridge.get_taiex()
            chain = bridge.get_txo_chain(taiex["price"])
            self._send_json(chain)
        else:
            self._send_json({"error": "Taiwan trader not initialized"})

    def _serve_taiwan_txo_history(self) -> None:
        """Return closed TXO trade history."""
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            bridge = Handler.taiwan_trader.get_bridge()
            self._send_json(bridge.get_closed_txo_positions(50))
        else:
            self._send_json([])

    def _post_taiwan_start(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            ok = Handler.taiwan_trader.start()
            self._send_json({"status": "ok" if ok else "already_running"})
        else:
            self._send_json({"status": "error"})

    def _post_taiwan_stop(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            Handler.taiwan_trader.stop(wait=False)
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_taiwan_pause(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            Handler.taiwan_trader.pause()
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_taiwan_resume(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            Handler.taiwan_trader.resume()
            self._send_json({"status": "ok"})
        else:
            self._send_json({"status": "error"})

    def _post_taiwan_transfer_reserve(self) -> None:
        if hasattr(Handler, 'taiwan_trader') and Handler.taiwan_trader:
            length = int(self.headers.get("Content-Length", 0))
            amount = 100_000  # default 10萬
            if length > 0:
                body = json.loads(self.rfile.read(length).decode())
                amount = int(body.get("amount", 100_000))
            bridge = Handler.taiwan_trader.get_bridge()
            result = bridge.transfer_from_reserve(amount)
            self._send_json(result)
        else:
            self._send_json({"status": "error", "message": "Trader not initialized"})

    # ---- Coordinator / Risk endpoints ----

    def _serve_coordinator_status(self) -> None:
        """Return combined P&L across stock and options portfolios."""
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            pnl = getattr(Handler.auto_trader, '_combined_pnl', {})
            if pnl:
                self._send_json(pnl)
                return
        # Fallback: compute live
        try:
            from strategies.coordinator import PortfolioCoordinator
            from strategies.options_agents import OptionsPortfolio
            from paper_trade import PaperPortfolio
            coord = PortfolioCoordinator(PaperPortfolio(), OptionsPortfolio())
            self._send_json(coord.get_combined_pnl())
        except Exception as e:
            self._send_json({"error": str(e)})

    def _serve_coordinator_actions(self) -> None:
        """Return recent coordinator actions."""
        if hasattr(Handler, 'auto_trader') and Handler.auto_trader:
            actions = getattr(Handler.auto_trader, '_last_coordinated_actions', [])
            self._send_json(actions[-50:])
        else:
            self._send_json([])

    def _serve_risk_status(self) -> None:
        """Return circuit breaker and risk status."""
        try:
            from strategies.risk import RiskController
            rc = RiskController()
            self._send_json(rc.get_status())
        except Exception as e:
            self._send_json({"error": str(e)})

    def _post_risk_reset_breaker(self) -> None:
        """Manually clear circuit breaker."""
        try:
            from strategies.risk import RiskController
            rc = RiskController()
            result = rc.reset_breaker()
            self._send_json(result)
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)})

    # ---- Research / Prediction Analytics endpoints ----

    def _serve_research_predictions(self) -> None:
        from prediction_tracker import get_tracker
        from urllib.parse import urlparse, parse_qs

        params = parse_qs(urlparse(self.path).query)
        system = params.get("system", [None])[0]
        status = params.get("status", ["all"])[0]
        backfill = params.get("backfill", ["true"])[0] != "false"

        tracker = get_tracker()
        preds = tracker.get_predictions(system=system, status=status, include_backfill=backfill)
        self._send_json(preds)

    def _serve_research_analytics(self) -> None:
        from prediction_tracker import get_tracker
        from analytics_engine import AnalyticsEngine

        system = self.path.replace("/api/research/analytics/", "").split("?")[0]
        if system not in ("taiwan_txo", "us_options"):
            self._send_json({"error": f"Unknown system: {system}"})
            return

        tracker = get_tracker()
        engine = AnalyticsEngine(tracker)
        analytics = engine.compute(system)
        self._send_json(analytics)

    def _serve_research_sensitivity(self) -> None:
        from prediction_tracker import get_tracker
        from strategy_optimizer import StrategyOptimizer

        system = self.path.replace("/api/research/sensitivity/", "").split("?")[0]
        if system not in ("taiwan_txo", "us_options"):
            self._send_json({"error": f"Unknown system: {system}"})
            return

        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(self.path).query)
        param_name = params.get("param", [None])[0]

        tracker = get_tracker()
        opt = StrategyOptimizer(tracker)
        if param_name:
            result = opt.analyze_parameter_sensitivity(system, param_name)
        else:
            result = opt.generate_recommendations(system)
        self._send_json(result)

    def _serve_research_recommendations(self) -> None:
        from prediction_tracker import get_tracker
        from strategy_optimizer import StrategyOptimizer
        from urllib.parse import urlparse, parse_qs

        params = parse_qs(urlparse(self.path).query)
        system = params.get("system", ["taiwan_txo"])[0]

        tracker = get_tracker()
        opt = StrategyOptimizer(tracker)
        recs = opt.generate_recommendations(system)
        self._send_json(recs)

    def _serve_research_stats(self) -> None:
        from prediction_tracker import get_tracker
        tracker = get_tracker()
        self._send_json(tracker.get_stats())

    def _post_research_backfill(self) -> None:
        from prediction_tracker import get_tracker
        tracker = get_tracker()
        n = tracker.backfill_from_portfolios()
        stats = tracker.get_stats()
        self._send_json({"status": "ok", "backfilled": n, "total_predictions": stats["total_predictions"]})

    def log_message(self, fmt: str, *args: tuple) -> None:
        """簡化 log，不顯示靜態檔請求。"""
        msg = fmt % args
        if "GET /api/" in msg or "POST" in msg:
            logger.info(f"  🌐 {msg}")


# ---------------------------------------------------------------------------
# 伺服器啟動
# ---------------------------------------------------------------------------

def start_ngrok(port: int) -> Optional[str]:
    """嘗試啟動 ngrok 並回傳公開網址。"""
    try:
        proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(2)
        # 用 ngrok API 取得網址
        import urllib.request
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels")
            data = json.loads(resp.read())
            url = data["tunnels"][0]["public_url"]
            logger.info(f"  🔗 ngrok 公開網址: {url}")
            return url
        except Exception:
            logger.warning("  ⚠️  ngrok 啟動但無法取得網址")
            return None
    except FileNotFoundError:
        logger.warning("  ⚠️  未安裝 ngrok，跳過")
        return None


def get_local_ips() -> list[str]:
    """取得本機 IP 列表。"""
    import socket
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            if addr.startswith("192.") or addr.startswith("10.") or addr.startswith("172."):
                if addr not in ips:
                    ips.append(addr)
    except Exception:
        pass
    return ips


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="🌐 交易儀表板伺服器")
    parser.add_argument("--port", type=int, default=8080, help="埠號")
    parser.add_argument("--public", action="store_true", help="啟用 ngrok 公開網址")
    args = parser.parse_args()

    port = args.port

    # 確保 dashboard.html 存在
    if not DASHBOARD_FILE.exists():
        logger.info("  ⏳ 首次啟動，產生儀表板…")
        subprocess.run([sys.executable, "generate_dashboard.py"])

    # 啟動背景自動交易（美股）
    from auto_trader import AutoTrader
    Handler.auto_trader = AutoTrader(interval=300)
    Handler.auto_trader.start()
    logger.info(f"  🤖 美股自動交易: 已啟動（每 5 分鐘）")

    # 建立台灣模擬交易（等待按下啟動）
    from taiwan_trader import TaiwanAutoTrader
    Handler.taiwan_trader = TaiwanAutoTrader(initial_cash=400_000, reserve_cash=500_000)
    logger.info(f"  🇹🇼 台股模擬交易: 待命（資金 NT$40 萬 + 備用 NT$50 萬）")

    # 啟動伺服器
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"\n{'='*50}")
    logger.info(f"  🌐 交易儀表板伺服器")
    logger.info(f"{'='*50}")
    logger.info(f"  📍 本機:     http://localhost:{port}")
    for ip in get_local_ips():
        logger.info(f"  📍 區域網路: http://{ip}:{port}")
    logger.info(f"  📊 API 狀態: http://localhost:{port}/api/status")
    logger.info(f"  🔄 API 更新: curl -X POST http://localhost:{port}/api/update")
    logger.info(f"{'='*50}\n")

    # 可選 ngrok
    public_url = None
    if args.public:
        public_url = start_ngrok(port)

    if not public_url:
        logger.info("  💡 要從外部存取，可以：")
        logger.info("     1. 安裝 Tailscale (https://tailscale.com) — 最安全")
        logger.info("     2. 安裝 ngrok 並加上 --public 參數")
        logger.info("     3. 部署到雲端伺服器\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n  👋 伺服器已停止")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
