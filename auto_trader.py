"""Auto-trader — background automated simulated trading.

Runs periodic analysis + trade execution for both stocks/futures and options.
Integrates into server.py as a daemon thread.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from prediction_tracker import get_tracker, PredictionRecord

from strategies.risk import RiskController
from strategies.sizing import KellySizer

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 300  # 5 minutes between cycles
MIN_INTERVAL = 60
MAX_INTERVAL = 3600
MAX_LOG_ENTRIES = 100


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


class AutoTrader:
    """Background auto-trading coordinator.

    Runs a daemon thread that periodically:
    1. Updates stocks/futures (prices + analysis + trading)
    2. Updates options (committee analysis + auto-execute)
    3. Regenerates the dashboard
    """

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._state_lock = threading.Lock()
        self.file_lock = threading.Lock()

        self._interval = max(MIN_INTERVAL, min(interval, MAX_INTERVAL))
        self._running = False
        self._paused = False
        self._last_run: Optional[str] = None
        self._next_run: Optional[str] = None
        self._cycle_count = 0
        self._last_stock_update: Optional[str] = None
        self._last_options_update: Optional[str] = None
        self._current_phase = "idle"
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._log_entries: list[dict] = []
        self._start_time: Optional[str] = None

        # Integrated modules
        self._risk_controller = RiskController()
        self._kelly_sizer = KellySizer()
        self._last_coordinated_actions: list[dict] = []
        self._combined_pnl: dict = {}

    # ---- Public API (thread-safe) ----

    def start(self) -> bool:
        """Start the background thread. Returns False if already running."""
        with self._state_lock:
            if self._running:
                return False
            self._running = True
            self._paused = False
            self._stop_event.clear()
            self._wake_event.clear()
            self._cycle_count = 0
            self._last_error = None
            self._start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

        self._log("Auto-trader started", "info")
        return True

    def stop(self, wait: bool = True) -> None:
        """Signal stop and optionally wait for thread to join."""
        with self._state_lock:
            self._running = False
            self._paused = False
        self._stop_event.set()
        self._wake_event.set()
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._log("Auto-trader stopped", "info")

    def pause(self) -> None:
        """Pause after current cycle finishes."""
        with self._state_lock:
            self._paused = True
        self._log("Auto-trader paused", "info")

    def resume(self) -> None:
        """Resume from paused state."""
        with self._state_lock:
            self._paused = False
        self._wake_event.set()
        self._log("Auto-trader resumed", "info")

    def set_interval(self, seconds: int) -> None:
        """Change cycle interval (clamped 60-3600)."""
        clamped = max(MIN_INTERVAL, min(seconds, MAX_INTERVAL))
        with self._state_lock:
            self._interval = clamped

    def get_status(self) -> dict:
        """Return snapshot of current status (thread-safe)."""
        with self._state_lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "interval": self._interval,
                "last_run": self._last_run,
                "next_run": self._next_run,
                "cycle_count": self._cycle_count,
                "current_phase": self._current_phase,
                "last_stock_update": self._last_stock_update,
                "last_options_update": self._last_options_update,
                "last_error": self._last_error,
                "error_count": self._error_count,
                "start_time": self._start_time,
            }

    def get_log(self, count: int = 50) -> list:
        """Return last N log entries."""
        with self._state_lock:
            return self._log_entries[-count:]

    # ---- Internal: the run loop ----

    def _run_loop(self) -> None:
        """Main loop executed in the background thread."""
        while not self._stop_event.is_set():
            if self._paused:
                self._wake_event.wait(timeout=1)
                self._wake_event.clear()
                continue

            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._state_lock:
                self._next_run = now_ts

            self._execute_cycle()

            with self._state_lock:
                self._last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._cycle_count += 1
                self._current_phase = "idle"

            interval = self._interval
            for _ in range(interval):
                if self._stop_event.is_set() or self._paused:
                    break
                self._wake_event.wait(timeout=1)
                self._wake_event.clear()

    def _execute_cycle(self) -> None:
        """One full coordinated update cycle under file lock."""
        with self.file_lock:
            # Phase 0: Risk check — circuit breaker
            self._current_phase = "risk_check"
            if self._risk_controller.is_trading_paused():
                status = self._risk_controller.get_status()
                cooldown = status.get("cooldown_until", "unknown")
                self._log(f"Circuit breaker active — skipping cycle (cooldown until {cooldown})", "warn")
                return

            # Phase 1: Stock/futures update (delegates to paper_trade, includes trend filter)
            self._execute_stock_update()

            # Phase 2: Options committee + coordinated execution
            self._execute_coordinated_options()

            # Phase 3: Dashboard
            self._regenerate_dashboard()

    # ---- Stock/futures update ----

    def _execute_stock_update(self) -> None:
        """Run stock/futures analysis and trading."""
        with self._state_lock:
            self._current_phase = "stocks"

        try:
            from paper_trade import DEFAULT_WATCHLIST, cmd_update

            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                cmd_update(DEFAULT_WATCHLIST, fallback=True)
            finally:
                sys.stdout = old_stdout

            output = buf.getvalue()
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if any(kw in line for kw in ("📤", "📥", "買入", "賣出", "🔍", "📊", "💰")):
                    self._log(line, "trade" if "買入" in line or "賣出" in line else "info")

            with self._state_lock:
                self._last_stock_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            self._log(f"Stock update error: {e}", "error")
            with self._state_lock:
                self._last_error = str(e)
                self._error_count += 1

    # ---- Coordinated options update (with Kelly sizing) ----

    def _execute_coordinated_options(self) -> None:
        """Run options committee + coordinator + Kelly-based execution."""
        with self._state_lock:
            self._current_phase = "options"

        try:
            from strategies.options_agents import OptionsCommittee, OptionsPortfolio
            from strategies.options_agents.options_utils import get_option_chain
            from strategies.coordinator import PortfolioCoordinator
            from paper_trade import PaperPortfolio

            opt_pf = OptionsPortfolio()
            paper_pf = PaperPortfolio()
            committee = OptionsCommittee(symbols=["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT"])
            results = committee.run_analysis(portfolio=opt_pf.get_state())
            summary = committee.get_voting_summary()

            # Build stock signals from paper portfolio cache
            stock_signals: dict[str, dict] = {}
            for ticker, cached in paper_pf.signals_cache.items():
                stock_signals[ticker] = cached

            # Run coordinator
            coordinator = PortfolioCoordinator(
                portfolio=paper_pf,
                options_portfolio=opt_pf,
            )
            capital = paper_pf.total_equity + opt_pf.total_equity()
            actions = coordinator.coordinate(stock_signals, results, capital)

            # Execute coordinated actions
            for action in actions:
                if not action.approved:
                    continue

                symbol = action.symbol
                chain = get_option_chain(symbol)

                if not chain or "error" in chain:
                    continue

                underlying_price = chain.get("price", 0)
                if underlying_price <= 0:
                    continue

                # Skip if chain expiring within 3 days
                chain_dte = chain.get("dte", 30)
                if chain_dte <= 3:
                    self._log(f"Skip {symbol}: chain expiring DTE={chain_dte}", "warn")
                    continue

                # Stock action
                if action.stock_action == "buy" and action.stock_qty > 0:
                    try:
                        price = stock_signals.get(symbol, {}).get("price", underlying_price)
                        leverage = 2.0  # default
                        paper_pf.buy(symbol, price, leverage, kelly_qty=action.stock_qty)
                    except Exception as e:
                        self._log(f"Coordinated stock buy {symbol} error: {e}", "error")

                # Options action
                if action.options_action == "enter" and action.options_strategy != "none":
                    existing = [p for p in opt_pf.positions if p["symbol"] == symbol and p["status"] == "open"]
                    if not existing:
                        # Get contract count from Kelly sizing
                        contracts = action.options_contracts
                        if contracts > 0:
                            self._auto_enter_option_sized(
                                opt_pf, symbol, action.options_strategy,
                                chain, contracts,
                            )

            # Standard exit checks
            self._auto_exit_options(opt_pf, summary)

            # Update position prices
            for pos in opt_pf.positions:
                if pos["status"] == "open":
                    try:
                        chain = get_option_chain(pos["symbol"])
                        if chain and "error" not in chain:
                            opt_pf.update_prices(chain)
                    except Exception:
                        pass

            # Log committee decisions
            for vote in summary:
                symbol = vote.get("symbol", "")
                opt_pf.log_committee_decision(symbol, vote)

            opt_pf.record_equity()
            opt_pf.save()
            paper_pf.save()

            # Track combined P&L
            self._combined_pnl = coordinator.get_combined_pnl()
            self._last_coordinated_actions = coordinator.get_action_log()
            pnl = self._combined_pnl
            self._log(
                f"Combined equity: ${pnl['total_equity']:.2f} "
                f"(stock ${pnl['stock_equity']:.2f} + options ${pnl['options_equity']:.2f})",
                "info",
            )

            with self._state_lock:
                self._last_options_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            self._log(f"Coordinated options error: {e}", "error")
            with self._state_lock:
                self._last_error = str(e)
                self._error_count += 1

    def _auto_enter_option_sized(
        self, pf, symbol: str, strategy: str, chain: dict, contracts: int,
    ) -> None:
        """Enter an options position with Kelly-sized contracts."""
        try:
            underlying_price = chain.get("price", 0)
            if underlying_price <= 0:
                return

            # Get expiry
            chain_expiry = chain.get("expiry", "")
            chain_dte = chain.get("dte", 0)
            if chain_dte <= 3:
                return

            contract = self._select_option_for_strategy(chain, strategy, underlying_price)
            if not contract:
                self._log(f"No suitable contract for {symbol} {strategy}", "warn")
                return

            option_type = "put" if strategy in ("cash_secured_put", "bull_put_spread", "protective_put") else "call"
            strike = contract.get("strike", 0)
            premium = self._option_mid_price(contract)
            iv = contract.get("iv", 20)

            if strike <= 0 or not chain_expiry or premium <= 0:
                return

            # Record prediction
            pred_id = f"US-{datetime.now().strftime('%H%M%S%f')}"
            rec = PredictionRecord(
                id=pred_id, system="us_options",
                timestamp=datetime.now().isoformat(timespec="seconds"),
                direction=strategy, signal_strength=0.0,
                confidence=0.5, market_price=round(underlying_price, 2),
                market_change_pct=0.0, params_snapshot={},
                metadata={"symbol": symbol, "strategy": strategy, "contracts": contracts},
            )
            get_tracker().record_prediction(rec)

            pos = pf.enter_position(
                symbol=symbol, strategy=strategy,
                strike=strike, expiry=chain_expiry,
                option_type=option_type, contracts=contracts,
                entry_premium=premium, underlying_price=underlying_price,
                iv=iv, prediction_id=pred_id,
            )
            if pos:
                pos["auto_entered"] = True
                get_tracker().link_position(pred_id, str(pos["id"]))
                get_tracker().update_entry(pred_id, premium, {"symbol": symbol, "strike": strike, "expiry": chain_expiry})
                pf.save()
                self._log(
                    f"ENTER {symbol} {strategy} ${premium:.2f} ×{contracts}ct @{strike:.0f} exp{chain_expiry}",
                    "trade",
                )
        except Exception as e:
            self._log(f"Auto-enter {symbol} error: {e}", "error")

    def _auto_enter_option(
        self, pf, symbol: str, strategy: str, chain: dict, vote: dict,
        prediction_id: str | None = None,
    ) -> None:
        """Enter an options position based on committee decision."""
        try:
            underlying_price = chain.get("price", 0)
            if underlying_price <= 0:
                return

            # Get expiry from chain level
            chain_expiry = chain.get("expiry", "")
            chain_dte = chain.get("dte", 0)
            # Skip if chain expires within 3 days
            if chain_dte <= 3:
                self._log(f"Skip {symbol}: chain expiring DTE={chain_dte}", "warn")
                return

            contract = self._select_option_for_strategy(chain, strategy, underlying_price)
            if not contract:
                self._log(f"No suitable contract for {symbol} {strategy}", "warn")
                return

            option_type = "put" if strategy in ("cash_secured_put", "bull_put_spread", "protective_put") else "call"
            strike = contract.get("strike", 0)
            premium = self._option_mid_price(contract)
            iv = contract.get("iv", 20)

            if strike <= 0 or not chain_expiry or premium <= 0:
                self._log(f"Invalid contract for {symbol}: strike={strike} prem={premium} exp={chain_expiry}", "warn")
                return

            pos = pf.enter_position(
                symbol=symbol,
                strategy=strategy,
                strike=strike,
                expiry=chain_expiry,
                option_type=option_type,
                contracts=1,
                entry_premium=premium,
                underlying_price=underlying_price,
                iv=iv,
                prediction_id=prediction_id,
            )
            if pos:
                pos["auto_entered"] = True
                # Link prediction to position
                if prediction_id:
                    get_tracker().link_position(prediction_id, str(pos["id"]))
                    get_tracker().update_entry(prediction_id, premium, {"symbol": symbol, "strike": strike, "expiry": chain_expiry})
                pf.save()
                self._log(
                    f"ENTER {symbol} {strategy} ${premium:.2f} @{strike:.0f} exp{chain_expiry}",
                    "trade",
                )
        except Exception as e:
            self._log(f"Auto-enter {symbol} error: {e}", "error")

    def _option_mid_price(self, contract: dict) -> float:
        bid = contract.get("bid", 0)
        ask = contract.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return contract.get("last_price", 0)

    def _select_option_for_strategy(
        self, chain: dict, strategy: str, price: float
    ) -> Optional[dict]:
        """Select the appropriate option contract for a given strategy."""
        calls = chain.get("calls", [])
        puts = chain.get("puts", [])
        targets = {
            "covered_call": (calls, price * 1.02),
            "cash_secured_put": (puts, price * 0.98),
            "bull_put_spread": (puts, price * 0.95),
            "protective_put": (puts, price * 0.97),
        }
        options, target_strike = targets.get(strategy, ([], 0))
        if not options:
            return None
        # Find option closest to target strike
        best = None
        best_diff = float("inf")
        for opt in options:
            diff = abs(opt.get("strike", 0) - target_strike)
            if diff < best_diff:
                best_diff = diff
                best = opt
        return best

    def _auto_exit_options(self, pf, summary: list[dict]) -> None:
        """Check all open positions for exit conditions."""
        for pos in pf.positions:
            if pos["status"] != "open":
                continue
            symbol = pos["symbol"]
            strategy = pos["strategy"]
            entry_premium = pos.get("entry_premium", 0)
            current_premium = pos.get("current_premium", 0)
            direction = pos.get("direction", 1)

            if current_premium <= 0 or entry_premium <= 0:
                continue

            # 1. Committee exit signal
            vote = next((v for v in summary if v.get("symbol") == symbol), None)
            if vote:
                member_votes = vote.get("member_votes", [])
                exit_count = sum(1 for m in member_votes if m.get("action") == "exit" or m.get("strategy") == "exit")
                if exit_count >= 2:
                    self._close_option(pf, pos, current_premium, "Committee exit signal")
                    continue

            # 2. Expiration
            try:
                expiry = pos.get("expiry", "")
                if expiry:
                    dte = (date.fromisoformat(expiry) - date.today()).days
                    if dte <= 3:
                        self._close_option(pf, pos, current_premium, f"Expiring DTE={dte}")
                        continue
            except Exception:
                pass

            # 3. Short position TP/SL
            if direction == -1:
                collected = (entry_premium - current_premium) / entry_premium
                if collected >= 0.80:
                    self._close_option(pf, pos, current_premium, "TP 80% collected")
                    continue
                if current_premium / entry_premium >= 2.0:
                    self._close_option(pf, pos, current_premium, "SL premium doubled")
                    continue

            # 4. Long position TP/SL
            if direction == 1:
                ret = (current_premium - entry_premium) / entry_premium
                if ret >= 0.50:
                    self._close_option(pf, pos, current_premium, "TP +50%")
                    continue
                if ret <= -0.50:
                    self._close_option(pf, pos, current_premium, "SL -50%")
                    continue

    def _close_option(self, pf, pos: dict, current_premium: float, reason: str) -> None:
        """Close an options position and log it."""
        try:
            underlying_price = pos.get("entry_price", 0)
            result = pf.close_position(pos["id"], current_premium, underlying_price)
            if result:
                self._log(f"CLOSE {pos['symbol']} {pos['strategy']} ${result.get('pnl', 0):+.2f} ({reason})", "trade")
        except Exception as e:
            self._log(f"Close error {pos['symbol']}: {e}", "error")

    # ---- Dashboard ----

    def _regenerate_dashboard(self) -> None:
        """Regenerate dashboard.html."""
        with self._state_lock:
            self._current_phase = "dashboard"

        try:
            from generate_dashboard import generate as gen_dash
            html = gen_dash()
            Path("dashboard.html").write_text(html, encoding="utf-8")
        except Exception as e:
            self._log(f"Dashboard regeneration error: {e}", "warn")

    # ---- Logging ----

    def _log(self, message: str, level: str = "info") -> None:
        """Add entry to activity log (thread-safe)."""
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "message": message}
        with self._state_lock:
            self._log_entries.append(entry)
            if len(self._log_entries) > MAX_LOG_ENTRIES:
                self._log_entries = self._log_entries[-MAX_LOG_ENTRIES:]
        getattr(logger, level, logger.info)(f"  [auto] {message}")
