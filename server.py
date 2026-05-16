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
            self._serve_file(DASHBOARD_FILE, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._serve_json(PORTFOLIO_FILE)
        elif path == "/api/update":
            self._run_update()
        elif path.startswith("/api/history"):
            self._serve_history()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')

    def do_POST(self) -> None:
        if self.path == "/api/update":
            self._run_update()
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
        """執行 paper_trade.py --update 並回傳結果。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
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
