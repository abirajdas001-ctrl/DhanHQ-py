# Automated Options Trading Bot for Dhan HQ API (Short Strangle with OTM Hedges)
This code has been specifically structured for seamless deployment directly inside Google Colab, Jupyter Notebooks, or standard Python scripts.

## How to use this Bot and Backtester in Google Colab:
1. Open a new notebook on Google Colab (https://colab.research.google.com).
2. Create three code cells and paste each section from below.
3. Configure your API keys (Client ID and Access Token) in the parameters cell, select `live_trading` as `True` or `False` (Simulation Mode), and run!

---

### CELL 1: Install Dependencies
```python
!pip install dhanhq pandas numpy nest_asyncio
```

### CELL 2: Live Trading Bot & Backtester Source Code
```python
import os
import sys
import asyncio
import logging
import datetime
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Setup logging for Notebook context
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ColabDhanOptionsBot")

# Try importing dhanhq
try:
    from dhanhq import DhanContext, dhanhq
except ImportError:
    logger.error("dhanhq library is not installed. Please install it in the cell above!")


class DhanOptionsStrangleBot:
    """
    Automated Options Trading Bot for Dhan HQ API.
    Implements a Net Credit Short Strangle with dynamic further OTM hedges.
    """
    def __init__(
        self,
        client_id: str,
        access_token: str,
        live_trading: bool = False,
        lot_size: Optional[int] = None,
        quantity_lots: int = 2,
        entry_time_str: str = "09:45",
        exit_time_str: str = "14:00",
        poll_interval_seconds: int = 5,
    ):
        self.client_id = client_id
        self.access_token = access_token
        self.live_trading = live_trading
        self.quantity_lots = quantity_lots
        self.entry_time_str = entry_time_str
        self.exit_time_str = exit_time_str
        self.poll_interval_seconds = poll_interval_seconds

        # Dhan context and client
        self.ctx = None
        self.dhan = None
        if self.live_trading:
            if not self.client_id or not self.access_token:
                raise ValueError("Client ID and Access Token are required for live trading!")
            self.ctx = DhanContext(self.client_id, self.access_token)
            self.dhan = dhanhq(self.ctx)
            logger.info("DhanHQ client initialized for LIVE trading.")
        else:
            logger.info("DhanHQ client initialized in SIMULATION / PAPER trading mode.")

        # Real-time lot size caching (Defaults to 65 as requested)
        self.lot_size = lot_size or self.fetch_real_time_nifty_lot_size()
        self.quantity_contracts = self.quantity_lots * self.lot_size
        logger.info(f"Nifty Lot Size configured as: {self.lot_size}. Total contracts to trade: {self.quantity_contracts}")

        # Position tracking
        self.active_trade = False
        self.expiry_date: Optional[str] = None
        self.legs: Dict[str, Dict[str, Any]] = {}
        self.total_net_premium_collected = 0.0
        self.max_sl_points = 0.0
        self.daily_tp_points = 0.0

    def fetch_real_time_nifty_lot_size(self) -> int:
        """
        Fetches the real-time NIFTY lot size by downloading the compact scrip master CSV from Dhan.
        Falls back to 65 if download fails or is unavailable.
        """
        logger.info("Fetching Nifty real-time lot size from Dhan scrip master...")
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            nifty_df = df[
                (df['SEM_TRADING_SYMBOL'].str.startswith('NIFTY', na=False)) &
                (df['SEM_SEGMENT'] == 'D')
            ]
            if not nifty_df.empty:
                lot_size = int(nifty_df['SEM_LOT_UNITS'].iloc[0])
                logger.info(f"Successfully fetched real-time Nifty lot size from scrip master: {lot_size}")
                return lot_size
        except Exception as e:
            logger.warning(f"Could not fetch lot size from CSV: {e}. Falling back to default Nifty lot size of 65.")
        return 65

    def get_time_from_str(self, time_str: str) -> datetime.time:
        parts = list(map(int, time_str.split(":")))
        return datetime.time(parts[0], parts[1])

    def is_expiry_day_to_skip(self, today_date: Optional[datetime.date] = None) -> bool:
        today = today_date or datetime.date.today()
        if today.weekday() == 1: # Tuesday is 1
            return True
        return False

    def select_options_legs(self, option_chain_data: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        last_price = option_chain_data.get("last_price")
        if not last_price:
            inner_data = option_chain_data.get("data", {})
            last_price = inner_data.get("last_price")
            oc_map = inner_data.get("oc", {})
        else:
            oc_map = option_chain_data.get("oc", {})

        if not oc_map:
            logger.error("Option chain data map 'oc' is empty or invalid.")
            return None

        calls: List[Dict[str, Any]] = []
        puts: List[Dict[str, Any]] = []

        for strike_str, strike_data in oc_map.items():
            strike_price = float(strike_str)
            ce = strike_data.get("ce")
            pe = strike_data.get("pe")

            if ce:
                delta_val = ce.get("greeks", {}).get("delta") or 0.0
                calls.append({
                    "strike": strike_price,
                    "security_id": ce.get("security_id"),
                    "last_price": ce.get("last_price", ce.get("average_price", 0.0)),
                    "delta": float(delta_val),
                    "type": "CE"
                })
            if pe:
                delta_val = pe.get("greeks", {}).get("delta") or 0.0
                puts.append({
                    "strike": strike_price,
                    "security_id": pe.get("security_id"),
                    "last_price": pe.get("last_price", pe.get("average_price", 0.0)),
                    "delta": float(delta_val),
                    "type": "PE"
                })

        if not calls or not puts:
            logger.error("Could not retrieve call or put legs from option chain.")
            return None

        # Both deltas 20
        sold_call = min(calls, key=lambda x: abs(abs(x["delta"]) - 0.20))
        sold_put = min(puts, key=lambda x: abs(abs(x["delta"]) - 0.20))

        further_otm_calls = [c for c in calls if c["strike"] > sold_call["strike"] and c["last_price"] > 0]
        if not further_otm_calls:
            further_otm_calls = [c for c in calls if c["strike"] >= sold_call["strike"]]
        target_call_premium = sold_call["last_price"] * 0.5
        bought_call = min(further_otm_calls, key=lambda x: abs(x["last_price"] - target_call_premium))

        further_otm_puts = [p for p in puts if p["strike"] < sold_put["strike"] and p["last_price"] > 0]
        if not further_otm_puts:
            further_otm_puts = [p for p in puts if p["strike"] <= sold_put["strike"]]
        target_put_premium = sold_put["last_price"] * 0.5
        bought_put = min(further_otm_puts, key=lambda x: abs(x["last_price"] - target_put_premium))

        return {
            "sold_call": sold_call,
            "sold_put": sold_put,
            "bought_call": bought_call,
            "bought_put": bought_put
        }

    def execute_order(self, security_id: str, action: str, qty: int) -> Optional[str]:
        if not self.live_trading:
            logger.info(f"[SIMULATION] Executed Order - Action: {action}, Security ID: {security_id}, Qty: {qty}")
            return f"SIM_ORDER_{security_id}_{action}"

        try:
            order_id = self.dhan.place_order(
                security_id=str(security_id),
                exchange_segment="NSE_FNO",
                transaction_type=action,
                quantity=qty,
                order_type="MARKET",
                product_type="INTRA"
            )
            logger.info(f"Live Order placed: ID {order_id} ({action} {security_id} x{qty})")
            return order_id
        except Exception as e:
            logger.error(f"Failed to execute order for Security ID {security_id}: {e}")
            return None

    def execute_strangle_positions(self, legs: Dict[str, Dict[str, Any]]) -> bool:
        logger.info("Executing trade entry positions (BUY Hedges FIRST)...")

        bc_id = self.execute_order(legs["bought_call"]["security_id"], "BUY", self.quantity_contracts)
        if not bc_id:
            logger.error("Hedge Call Order failed. Aborting entry sequence!")
            return False

        bp_id = self.execute_order(legs["bought_put"]["security_id"], "BUY", self.quantity_contracts)
        if not bp_id:
            logger.error("Hedge Put Order failed. Aborting entry sequence!")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            return False

        sc_id = self.execute_order(legs["sold_call"]["security_id"], "SELL", self.quantity_contracts)
        if not sc_id:
            logger.error("Short Call Order failed! Neutralizing hedges.")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)
            return False

        sp_id = self.execute_order(legs["sold_put"]["security_id"], "SELL", self.quantity_contracts)
        if not sp_id:
            logger.error("Short Put Order failed! Neutralizing positions.")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["sold_call"]["security_id"], "BUY", self.quantity_contracts)
            return False

        self.legs = {
            "sold_call": {**legs["sold_call"], "entry_price": legs["sold_call"]["last_price"], "order_id": sc_id},
            "sold_put": {**legs["sold_put"], "entry_price": legs["sold_put"]["last_price"], "order_id": sp_id},
            "bought_call": {**legs["bought_call"], "entry_price": legs["bought_call"]["last_price"], "order_id": bc_id},
            "bought_put": {**legs["bought_put"], "entry_price": legs["bought_put"]["last_price"], "order_id": bp_id},
        }

        self.total_net_premium_collected = (
            self.legs["sold_call"]["entry_price"] + self.legs["sold_put"]["entry_price"]
        ) - (
            self.legs["bought_call"]["entry_price"] + self.legs["bought_put"]["entry_price"]
        )

        self.max_sl_points = self.total_net_premium_collected / 2.0
        self.daily_tp_points = self.max_sl_points * 0.5

        logger.info("=== TRADE ENTRY COMPLETED ===")
        logger.info(f"Sold Call: Strike {self.legs['sold_call']['strike']} @ {self.legs['sold_call']['entry_price']}")
        logger.info(f"Sold Put: Strike {self.legs['sold_put']['strike']} @ {self.legs['sold_put']['entry_price']}")
        logger.info(f"Bought Call Hedge: Strike {self.legs['bought_call']['strike']} @ {self.legs['bought_call']['entry_price']}")
        logger.info(f"Bought Put Hedge: Strike {self.legs['bought_put']['strike']} @ {self.legs['bought_put']['entry_price']}")
        logger.info(f"Net Premium Collected: {self.total_net_premium_collected:.2f} points")
        logger.info(f"Stop-Loss Threshold: {self.total_net_premium_collected + self.max_sl_points:.2f} (SL: {self.max_sl_points:.2f} points)")
        logger.info(f"Take-Profit Threshold: {self.total_net_premium_collected - self.daily_tp_points:.2f} (TP: {self.daily_tp_points:.2f} points)")
        logger.info("=============================")

        self.active_trade = True
        return True

    def square_off_all_positions(self, reason: str = "Exit Condition Met"):
        logger.info(f"=== SQUARING OFF ALL POSITIONS: {reason} ===")
        if not self.active_trade:
            return

        self.execute_order(self.legs["sold_call"]["security_id"], "BUY", self.quantity_contracts)
        self.execute_order(self.legs["sold_put"]["security_id"], "BUY", self.quantity_contracts)
        self.execute_order(self.legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
        self.execute_order(self.legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)

        logger.info("All positions closed.")
        self.active_trade = False

    async def get_current_greeks_and_prices(self) -> Optional[Dict[str, float]]:
        if not self.live_trading:
            return {
                "sold_call_ltp": self.legs["sold_call"]["entry_price"],
                "sold_put_ltp": self.legs["sold_put"]["entry_price"],
                "bought_call_ltp": self.legs["bought_call"]["entry_price"],
                "bought_put_ltp": self.legs["bought_put"]["entry_price"],
            }

        try:
            oc_data = self.dhan.option_chain(
                under_security_id=13,
                under_exchange_segment="IDX_I",
                expiry=self.expiry_date
            )
            inner_data = oc_data.get("data", {})
            oc_map = inner_data.get("oc", {})

            sc_strike = str(float(self.legs["sold_call"]["strike"]))
            sp_strike = str(float(self.legs["sold_put"]["strike"]))
            bc_strike = str(float(self.legs["bought_call"]["strike"]))
            bp_strike = str(float(self.legs["bought_put"]["strike"]))

            prices = {
                "sold_call_ltp": oc_map.get(sc_strike, {}).get("ce", {}).get("last_price", self.legs["sold_call"]["entry_price"]),
                "sold_put_ltp": oc_map.get(sp_strike, {}).get("pe", {}).get("last_price", self.legs["sold_put"]["entry_price"]),
                "bought_call_ltp": oc_map.get(bc_strike, {}).get("ce", {}).get("last_price", self.legs["bought_call"]["entry_price"]),
                "bought_put_ltp": oc_map.get(bp_strike, {}).get("pe", {}).get("last_price", self.legs["bought_put"]["entry_price"]),
            }
            return prices
        except Exception as e:
            logger.error(f"Error fetching live prices: {e}")
            return None

    async def run(self):
        logger.info("Starting options strangle trading cycle...")

        if self.is_expiry_day_to_skip():
            logger.info("Skipping trade deployment: Today is Tuesday (Nifty Expiry Day).")
            return

        entry_time = self.get_time_from_str(self.entry_time_str)
        exit_time = self.get_time_from_str(self.exit_time_str)

        while True:
            now = datetime.datetime.now().time()
            if now >= entry_time:
                break
            logger.info(f"Waiting for entry time {self.entry_time_str}. Current: {now.strftime('%H:%M:%S')}")
            await asyncio.sleep(30)

        logger.info("Entry time reached! Querying option chain...")
        if self.live_trading:
            try:
                exp_list_resp = self.dhan.expiry_list(under_security_id=13, under_exchange_segment="IDX_I")
                dates = exp_list_resp.get("data", [])
                if not dates:
                    logger.error("Could not fetch active expiry dates!")
                    return
                dates.sort()
                self.expiry_date = dates[0]
                chain_resp = self.dhan.option_chain(
                    under_security_id=13,
                    under_exchange_segment="IDX_I",
                    expiry=self.expiry_date
                )
            except Exception as e:
                logger.error(f"Dhan API query failed: {e}")
                return
        else:
            self.expiry_date = (datetime.date.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            chain_resp = self.generate_mock_option_chain()

        selected_legs = self.select_options_legs(chain_resp)
        if not selected_legs:
            logger.error("Failed to choose viable options contracts.")
            return

        success = self.execute_strangle_positions(selected_legs)
        if not success:
            return

        while self.active_trade:
            now = datetime.datetime.now().time()
            if now >= exit_time:
                self.square_off_all_positions("Time Exit (2:00 PM Square-Off)")
                break

            current_prices = await self.get_current_greeks_and_prices()
            if current_prices:
                sc_ltp = current_prices["sold_call_ltp"]
                sp_ltp = current_prices["sold_put_ltp"]
                bc_ltp = current_prices["bought_call_ltp"]
                bp_ltp = current_prices["bought_put_ltp"]

                curr_net_premium = (sc_ltp + sp_ltp) - (bc_ltp + bp_ltp)
                pl_points = self.total_net_premium_collected - curr_net_premium

                logger.info(
                    f"Monitoring - Current Net Premium: {curr_net_premium:.2f} | "
                    f"PnL: {pl_points:+.2f} pts | "
                    f"SC: {sc_ltp:.1f}, SP: {sp_ltp:.1f}, BC: {bc_ltp:.1f}, BP: {bp_ltp:.1f}"
                )

                if curr_net_premium >= (self.total_net_premium_collected + self.max_sl_points):
                    self.square_off_all_positions(f"Stop-Loss (SL) hit of {self.max_sl_points:.2f} points.")
                    break

                if curr_net_premium <= (self.total_net_premium_collected - self.daily_tp_points):
                    self.square_off_all_positions(f"Take-Profit (TP) hit of {self.daily_tp_points:.2f} points.")
                    break

            await asyncio.sleep(self.poll_interval_seconds)

        logger.info("Session cycle complete.")

    def generate_mock_option_chain(self) -> Dict[str, Any]:
        oc = {}
        for strike in range(23500, 24500, 50):
            dist = strike - 24000
            c_delta = max(0.01, 0.5 - dist * 0.001) if dist >= 0 else 0.5 + abs(dist) * 0.001
            p_delta = min(-0.01, -0.5 + dist * 0.001) if dist <= 0 else -0.5 - dist * 0.001
            c_price = max(1.0, 150 - dist * 0.6) if dist >= 0 else 150 + abs(dist)
            p_price = max(1.0, 150 - abs(dist) * 0.6) if dist <= 0 else 150 + dist

            oc[f"{strike}.000000"] = {
                "ce": {"security_id": f"CE_{strike}", "average_price": c_price, "last_price": c_price, "greeks": {"delta": c_delta}},
                "pe": {"security_id": f"PE_{strike}", "average_price": p_price, "last_price": p_price, "greeks": {"delta": p_delta}}
            }
        return {"data": {"last_price": 24000.0, "oc": oc}}


class DhanOptionsStrangleBacktester:
    """
    Backtesting Engine for the Dhan HQ Options Strangle Strategy.
    Simulates the strategy over a historical dataset.
    """
    def __init__(self, lot_size: int = 65, quantity_lots: int = 2, entry_time_str: str = "09:45", exit_time_str: str = "14:00"):
        self.lot_size = lot_size
        self.quantity_lots = quantity_lots
        self.qty_contracts = self.lot_size * self.quantity_lots
        self.entry_time = datetime.datetime.strptime(entry_time_str, "%H:%M").time()
        self.exit_time = datetime.datetime.strptime(exit_time_str, "%H:%M").time()

    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        if historical_data.empty:
            return {"error": "Historical dataset is empty."}

        historical_data['date'] = pd.to_datetime(historical_data['date']).dt.date
        unique_dates = sorted(historical_data['date'].unique())

        trades = []
        for date in unique_dates:
            if date.weekday() == 1: # Skip Tuesdays (Nifty Expiry)
                continue

            day_data = historical_data[historical_data['date'] == date]
            entry_rows = day_data[day_data['time'].astype(str).str.startswith("09:45")]
            if entry_rows.empty:
                entry_rows = day_data

            calls = entry_rows[entry_rows['option_type'] == 'CE']
            puts = entry_rows[entry_rows['option_type'] == 'PE']
            if calls.empty or puts.empty:
                continue

            sold_call = calls.iloc[(calls['delta'].abs() - 0.20).abs().argsort()[:1]].iloc[0]
            sold_put = puts.iloc[(puts['delta'].abs() - 0.20).abs().argsort()[:1]].iloc[0]

            further_calls = calls[calls['strike'] > sold_call['strike']]
            if further_calls.empty:
                further_calls = calls
            bought_call = further_calls.iloc[(further_calls['premium'] - sold_call['premium'] * 0.5).abs().argsort()[:1]].iloc[0]

            further_puts = puts[puts['strike'] < sold_put['strike']]
            if further_puts.empty:
                further_puts = puts
            bought_put = further_puts.iloc[(further_puts['premium'] - sold_put['premium'] * 0.5).abs().argsort()[:1]].iloc[0]

            net_premium_collected = (sold_call['premium'] + sold_put['premium']) - (bought_call['premium'] + bought_put['premium'])
            max_sl_points = net_premium_collected / 2.0
            daily_tp_points = max_sl_points * 0.5

            sc_id, sp_id, bc_id, bp_id = sold_call['strike'], sold_put['strike'], bought_call['strike'], bought_put['strike']
            after_entry = day_data[day_data['time'].astype(str) >= "09:45"]
            unique_times = sorted(after_entry['time'].unique())

            exit_reason = "Time Exit (2:00 PM Square-Off)"
            exit_time = "14:00"
            exit_net_premium = net_premium_collected

            for t_val in unique_times:
                time_str = str(t_val)
                if time_str >= "14:00":
                    break

                min_rows = after_entry[after_entry['time'] == t_val]
                sc_curr = min_rows[(min_rows['strike'] == sc_id) & (min_rows['option_type'] == 'CE')]
                sp_curr = min_rows[(min_rows['strike'] == sp_id) & (min_rows['option_type'] == 'PE')]
                bc_curr = min_rows[(min_rows['strike'] == bc_id) & (min_rows['option_type'] == 'CE')]
                bp_curr = min_rows[(min_rows['strike'] == bp_id) & (min_rows['option_type'] == 'PE')]

                if sc_curr.empty or sp_curr.empty or bc_curr.empty or bp_curr.empty:
                    continue

                sc_p, sp_p = sc_curr['premium'].iloc[0], sp_curr['premium'].iloc[0]
                bc_p, bp_p = bc_curr['premium'].iloc[0], bp_curr['premium'].iloc[0]
                curr_net_premium = (sc_p + sp_p) - (bc_p + bp_p)

                if curr_net_premium >= (net_premium_collected + max_sl_points):
                    exit_reason = "Stop-Loss (SL) Hit"
                    exit_time = time_str
                    exit_net_premium = curr_net_premium
                    break

                if curr_net_premium <= (net_premium_collected - daily_tp_points):
                    exit_reason = "Take-Profit (TP) Hit"
                    exit_time = time_str
                    exit_net_premium = curr_net_premium
                    break

                exit_net_premium = curr_net_premium

            p_and_l_points = net_premium_collected - exit_net_premium
            p_and_l_amount = p_and_l_points * self.qty_contracts

            trades.append({
                "date": date, "sold_call_strike": sc_id, "sold_put_strike": sp_id, "bought_call_strike": bc_id, "bought_put_strike": bp_id,
                "net_premium_collected": net_premium_collected, "exit_net_premium": exit_net_premium, "p_and_l_points": p_and_l_points,
                "p_and_l_amount": p_and_l_amount, "exit_time": exit_time, "exit_reason": exit_reason
            })

        if not trades:
            return {"error": "No executed trades."}

        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['p_and_l_amount'].sum()
        win_rate = (len(trades_df[trades_df['p_and_l_amount'] > 0]) / len(trades_df)) * 100

        cumulative_pnl = trades_df['p_and_l_amount'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()

        return {
            "metrics": {
                "total_trades": len(trades_df),
                "win_rate_percent": round(win_rate, 2),
                "total_net_pnl_amount": round(total_pnl, 2),
                "max_drawdown_amount": round(max_drawdown, 2),
            },
            "trades": trades_df
        }

    @staticmethod
    def generate_dummy_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
        date_range = pd.date_range(start_date, end_date)
        records = []
        for d in date_range:
            day_str = d.strftime("%Y-%m-%d")
            for strike in range(23600, 24400, 50):
                dist = strike - 24000
                c_delta = max(0.01, 0.5 - dist * 0.001)
                p_delta = min(-0.01, -0.5 + dist * 0.001)

                for time_str in ["09:45", "11:30", "14:00"]:
                    market_move = 0 if time_str == "09:45" else (50 if time_str == "11:30" else 80)
                    dist_now = strike - (24000 + market_move)

                    c_prem = max(2.0, 150 - dist_now * 0.5 + random.uniform(-10, 10))
                    p_prem = max(2.0, 150 + dist_now * 0.5 + random.uniform(-10, 10))

                    records.append({"date": day_str, "time": time_str, "underlying_price": 24000.0 + market_move, "strike": strike, "option_type": "CE", "premium": c_prem, "delta": c_delta})
                    records.append({"date": day_str, "time": time_str, "underlying_price": 24000.0 + market_move, "strike": strike, "option_type": "PE", "premium": p_prem, "delta": p_delta})

        return pd.DataFrame(records)
```

### CELL 3: Execution Settings and Initialization
```python
# ==========================================
#      USER SETTING CONFIGURATIONS
# ==========================================
# Paste your Dhan HQ API keys below if running LIVE or PAPER/SIMULATION.
CLIENT_ID = "YOUR_CLIENT_ID"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# Toggle LIVE_TRADING to True to connect to live market. If False, runs in simulated/paper mode.
LIVE_TRADING = False

# ==========================================
#         RUN LIVE/SIMULATION BOT
# ==========================================
print("--- Launching Dhan Options Automated Bot ---")
bot = DhanOptionsStrangleBot(
    client_id=CLIENT_ID,
    access_token=ACCESS_TOKEN,
    live_trading=LIVE_TRADING,
    quantity_lots=2,
    poll_interval_seconds=3
)

# Launch Bot Task in the notebook environment
import nest_asyncio
nest_asyncio.apply() # Allows running asyncio in Jupyter
asyncio.run(bot.run())

# ==========================================
#            RUN SYSTEM BACKTEST
# ==========================================
print("\n" + "="*40)
print("            RUNNING SYSTEM BACKTEST")
print("="*40)
print("Generating sample historical Nifty options dataset...")
dummy_historical_data = DhanOptionsStrangleBacktester.generate_dummy_historical_data("2026-07-01", "2026-07-28")

print("Executing historical backtest on generated options data...")
backtest_engine = DhanOptionsStrangleBacktester(lot_size=65, quantity_lots=2)
results = backtest_engine.run_backtest(dummy_historical_data)

if "error" in results:
    print(f"Error occurred during backtest: {results['error']}")
else:
    print("\n" + "*"*45)
    print("        BACKTEST PERFORMANCE RESULTS")
    print("*"*45)
    for k, v in results['metrics'].items():
        print(f"{k.replace('_', ' ').title():<25}: {v}")
    print("*"*45)
```
