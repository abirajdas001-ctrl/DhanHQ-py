"""
========================================================================================
             DHAN HQ API AUTOMATED OPTIONS TRADING BOT & BACKTESTING SYSTEM
========================================================================================
Designed and optimized for seamless execution in Google Colab, Jupyter Notebooks, or locally.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Open a new Google Colab notebook (https://colab.research.google.com).
2. Create a code cell and paste this entire code.
3. Configure your API credentials and flags in the 'USER CONFIGURATION' section below.
4. Run the cell!

FEATURES:
- Automated Real-time Options Short Strangle (CE and PE near 20 Delta).
- Hedge Leg buying: Further OTM legs with ~half the premium of the sold legs (Hedges are bought first for portfolio margin benefits).
- Exit Rules: Daily Max Stop-Loss (SL) = 50% of collected net premium; Take-Profit (TP) = 50% of Max SL; or Time Square-Off at 2:00 PM.
- Expiry Protection: Automatically skips execution on Tuesdays (Nifty Weekly Expiry Day as of Sept 2025).
- Real-time Lot size retrieval with custom default of 65.
- Complete Backtesting system included in a single unified script.
========================================================================================
"""

# ======================================================================================
#                            1. USER CONFIGURATION SECTION
# ======================================================================================
# Enter your Dhan Client ID and API Access Token here:
CLIENT_ID = "YOUR_CLIENT_ID_HERE"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"

# Execution parameters:
LIVE_TRADING = False     # Set to True to trade live on Dhan. Set to False for simulation/paper trading.
RUN_BACKTEST = True      # Set to True to run the historical backtest simulation immediately on dummy data.
QUANTITY_LOTS = 2        # Number of lots (sell 2 lots Call + 2 lots Put, buy 2 lots Call hedge + 2 lots Put hedge)
DEFAULT_LOT_SIZE = 65    # Lot size default for Nifty 50 (e.g. 65)

# Entry & exit times (IST):
ENTRY_TIME_STR = "09:45"
EXIT_TIME_STR = "14:00"
POLL_INTERVAL_SECONDS = 5  # Time interval to check SL/TP/Time conditions
# ======================================================================================


import os
import sys
import asyncio
import logging
import datetime
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Configure clean logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DhanStrangleSystem")

# Install missing libraries dynamically in Google Colab if they aren't present
try:
    import nest_asyncio
except ImportError:
    logger.info("Installing nest_asyncio package dynamically for Jupyter support...")
    os.system("pip install nest_asyncio")
    import nest_asyncio

try:
    from dhanhq import DhanContext, dhanhq
except ImportError:
    logger.info("Installing dhanhq package dynamically...")
    os.system("pip install dhanhq")
    from dhanhq import DhanContext, dhanhq

# Apply nest_asyncio to support running the asyncio loop directly within Colab/Jupyter Notebooks
nest_asyncio.apply()


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
        lot_size: int = 65,
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
            if not self.client_id or not self.access_token or "YOUR_" in self.client_id:
                raise ValueError("Valid Client ID and Access Token are required for LIVE trading! Please edit the USER CONFIGURATION section.")
            self.ctx = DhanContext(self.client_id, self.access_token)
            self.dhan = dhanhq(self.ctx)
            logger.info("DhanHQ client initialized successfully for LIVE trading.")
        else:
            logger.info("DhanHQ client initialized in SIMULATION / PAPER trading mode.")

        # Real-time lot size caching
        self.lot_size = lot_size
        if self.live_trading:
            self.lot_size = self.fetch_real_time_nifty_lot_size()

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
        Falls back to default if download fails or is unavailable.
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
            logger.warning(f"Could not fetch lot size from CSV: {e}. Falling back to default Nifty lot size of {self.lot_size}.")
        return self.lot_size

    def get_time_from_str(self, time_str: str) -> datetime.time:
        """Helper to convert HH:MM string to datetime.time"""
        parts = list(map(int, time_str.split(":")))
        return datetime.time(parts[0], parts[1])

    def is_expiry_day_to_skip(self, today_date: Optional[datetime.date] = None) -> bool:
        """
        Nifty weekly expiry is every Tuesday as of Sept 2025.
        Returns True if today is Tuesday, meaning we skip trading.
        """
        today = today_date or datetime.date.today()
        # Monday is 0, Tuesday is 1, ..., Sunday is 6
        if today.weekday() == 1:
            return True
        return False

    def select_options_legs(self, option_chain_data: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Selects the 4 options legs based on delta and premium rules from the option chain:
        1. Sell Call near 20 delta
        2. Sell Put near 20 delta
        3. Buy Call further OTM with premium ~50% of Sold Call premium
        4. Buy Put further OTM with premium ~50% of Sold Put premium
        """
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
                delta_val = ce.get("greeks", {}).get("delta")
                if delta_val is None:
                    delta_val = 0.0
                calls.append({
                    "strike": strike_price,
                    "security_id": ce.get("security_id"),
                    "last_price": ce.get("last_price", ce.get("average_price", 0.0)),
                    "delta": float(delta_val),
                    "type": "CE"
                })
            if pe:
                delta_val = pe.get("greeks", {}).get("delta")
                if delta_val is None:
                    delta_val = 0.0
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

        # 1. Sell leg Call: Near 20 delta (Call delta is usually positive, e.g. 0.20)
        sold_call = min(calls, key=lambda x: abs(abs(x["delta"]) - 0.20))

        # 2. Sell leg Put: Near 20 delta (Put delta is usually negative, e.g. -0.20)
        sold_put = min(puts, key=lambda x: abs(abs(x["delta"]) - 0.20))

        # 3. Hedge leg Call: Buy further OTM Call with premium ~50% of Sold Call premium
        # Further OTM for Calls means strike is strictly greater than sold Call strike
        further_otm_calls = [c for c in calls if c["strike"] > sold_call["strike"] and c["last_price"] > 0]
        if not further_otm_calls:
            further_otm_calls = [c for c in calls if c["strike"] >= sold_call["strike"]]

        target_call_premium = sold_call["last_price"] * 0.5
        bought_call = min(further_otm_calls, key=lambda x: abs(x["last_price"] - target_call_premium))

        # 4. Hedge leg Put: Buy further OTM Put with premium ~50% of Sold Put premium
        # Further OTM for Puts means strike is strictly less than sold Put strike
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
        """
        Executes an order on Dhan HQ API.
        action: 'BUY' or 'SELL'
        """
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
            logger.info(f"Live Order placed successfully. ID: {order_id} - Action: {action}, Security ID: {security_id}, Qty: {qty}")
            return order_id
        except Exception as e:
            logger.error(f"Failed to execute order for Security ID {security_id} ({action}): {e}")
            return None

    def execute_strangle_positions(self, legs: Dict[str, Dict[str, Any]]) -> bool:
        """
        Executes order entries for all four options legs.
        Sequence: Places BUY (hedge) orders first to obtain portfolio margin benefits,
        followed by SELL (strangle) orders.
        """
        logger.info("Executing trade entry positions (BUY Hedges FIRST)...")

        # 1. Buy Call Hedge
        bc_id = self.execute_order(legs["bought_call"]["security_id"], "BUY", self.quantity_contracts)
        if not bc_id:
            logger.error("Hedge Call Order failed. Aborting entry sequence!")
            return False

        # 2. Buy Put Hedge
        bp_id = self.execute_order(legs["bought_put"]["security_id"], "BUY", self.quantity_contracts)
        if not bp_id:
            logger.error("Hedge Put Order failed. Aborting entry sequence!")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            return False

        # 3. Sell Call Leg
        sc_id = self.execute_order(legs["sold_call"]["security_id"], "SELL", self.quantity_contracts)
        if not sc_id:
            logger.error("Short Call Order failed! Neutralizing hedges.")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)
            return False

        # 4. Sell Put Leg
        sp_id = self.execute_order(legs["sold_put"]["security_id"], "SELL", self.quantity_contracts)
        if not sp_id:
            logger.error("Short Put Order failed! Neutralizing positions.")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["sold_call"]["security_id"], "BUY", self.quantity_contracts)
            return False

        # Record entered prices and details
        self.legs = {
            "sold_call": {**legs["sold_call"], "entry_price": legs["sold_call"]["last_price"], "order_id": sc_id},
            "sold_put": {**legs["sold_put"], "entry_price": legs["sold_put"]["last_price"], "order_id": sp_id},
            "bought_call": {**legs["bought_call"], "entry_price": legs["bought_call"]["last_price"], "order_id": bc_id},
            "bought_put": {**legs["bought_put"], "entry_price": legs["bought_put"]["last_price"], "order_id": bp_id},
        }

        # Strangle pricing logic
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
        logger.info(f"Daily Stop-Loss (SL): {self.max_sl_points:.2f} points (Trigger when Net Premium >= {self.total_net_premium_collected + self.max_sl_points:.2f})")
        logger.info(f"Daily Take-Profit (TP): {self.daily_tp_points:.2f} points (Trigger when Net Premium <= {self.total_net_premium_collected - self.daily_tp_points:.2f})")
        logger.info("=============================")

        self.active_trade = True
        return True

    def square_off_all_positions(self, reason: str = "Exit Condition Met"):
        """
        Squares off all four options legs.
        Opposite transactions to close open risk.
        """
        logger.info(f"=== SQUARING OFF ALL POSITIONS: {reason} ===")
        if not self.active_trade:
            logger.warning("No active trade to square off.")
            return

        # Sell short legs to cover: BUY
        self.execute_order(self.legs["sold_call"]["security_id"], "BUY", self.quantity_contracts)
        self.execute_order(self.legs["sold_put"]["security_id"], "BUY", self.quantity_contracts)

        # Close hedge positions: SELL
        self.execute_order(self.legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
        self.execute_order(self.legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)

        logger.info("All positions squared off successfully.")
        self.active_trade = False

    async def get_current_greeks_and_prices(self) -> Optional[Dict[str, float]]:
        """
        Fetches current LTP for all 4 option legs.
        """
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
            logger.error(f"Error fetching real-time market prices: {e}")
            return None

    async def run(self):
        """
        Main runner of the options trading bot.
        """
        logger.info("Initializing automated options strangle bot...")

        if self.is_expiry_day_to_skip():
            logger.info("Today is Tuesday (Nifty Weekly Expiry Day). Skipping trade deployment according to rules.")
            return

        entry_time = self.get_time_from_str(self.entry_time_str)
        exit_time = self.get_time_from_str(self.exit_time_str)

        # Wait until Entry Time (9:45 AM)
        while True:
            now = datetime.datetime.now().time()
            if now >= entry_time:
                logger.info(f"Entry time {self.entry_time_str} reached.")
                break
            logger.info(f"Waiting for entry time ({self.entry_time_str}). Current time: {now.strftime('%H:%M:%S')}")
            await asyncio.sleep(30)

        # Fetch active expiry and option chain to deploy strangle
        logger.info("Fetching NIFTY options chain details...")
        if self.live_trading:
            try:
                exp_list_resp = self.dhan.expiry_list(under_security_id=13, under_exchange_segment="IDX_I")
                dates = exp_list_resp.get("data", [])
                if not dates:
                    logger.error("Could not fetch active expiry list. Exiting.")
                    return
                dates.sort()
                self.expiry_date = dates[0]
                logger.info(f"Targeting weekly options expiry: {self.expiry_date}")

                chain_resp = self.dhan.option_chain(
                    under_security_id=13,
                    under_exchange_segment="IDX_I",
                    expiry=self.expiry_date
                )
            except Exception as e:
                logger.error(f"Failed to query Dhan API: {e}")
                return
        else:
            self.expiry_date = (datetime.date.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            logger.info(f"[SIMULATION] Selecting active expiry: {self.expiry_date}")
            chain_resp = self.generate_mock_option_chain()

        selected_legs = self.select_options_legs(chain_resp)
        if not selected_legs:
            logger.error("Failed to select viable strangle and hedge legs. Exiting strategy.")
            return

        # Deploy trade positions
        success = self.execute_strangle_positions(selected_legs)
        if not success:
            logger.error("Failed to enter strangle positions.")
            return

        # Async monitoring loop
        logger.info("Starting real-time position monitoring...")
        while self.active_trade:
            now = datetime.datetime.now().time()
            if now >= exit_time:
                self.square_off_all_positions("Time Exit (2:00 PM Square-Off) reached.")
                break

            current_prices = await self.get_current_greeks_and_prices()
            if current_prices:
                sc_ltp = current_prices["sold_call_ltp"]
                sp_ltp = current_prices["sold_put_ltp"]
                bc_ltp = current_prices["bought_call_ltp"]
                bp_ltp = current_prices["bought_put_ltp"]

                current_net_premium = (sc_ltp + sp_ltp) - (bc_ltp + bp_ltp)
                pl_points = self.total_net_premium_collected - current_net_premium

                logger.info(
                    f"Monitoring - Current Net Premium: {current_net_premium:.2f} | "
                    f"P&L: {pl_points:+.2f} pts | "
                    f"SC: {sc_ltp:.1f}, SP: {sp_ltp:.1f}, BC: {bc_ltp:.1f}, BP: {bp_ltp:.1f}"
                )

                if current_net_premium >= (self.total_net_premium_collected + self.max_sl_points):
                    self.square_off_all_positions(f"Stop-Loss (SL) of {self.max_sl_points:.2f} points hit.")
                    break

                if current_net_premium <= (self.total_net_premium_collected - self.daily_tp_points):
                    self.square_off_all_positions(f"Take-Profit (TP) of {self.daily_tp_points:.2f} points hit.")
                    break

            await asyncio.sleep(self.poll_interval_seconds)

        logger.info("Options strategy session completed.")

    def generate_mock_option_chain(self) -> Dict[str, Any]:
        """
        Construct a mock option chain for simulation and unit testing.
        Centered around Nifty at 24000.
        """
        oc = {}
        for strike in range(23500, 24500, 50):
            dist = strike - 24000
            if dist < 0:
                c_delta = min(0.99, 0.5 + abs(dist) / 1000.0)
                c_price = 150.0 + abs(dist)
            else:
                c_delta = max(0.01, 0.5 - dist / 1000.0)
                c_price = max(1.0, 150.0 - dist * 0.6)

            if dist > 0:
                p_delta = max(-0.99, -0.5 - dist / 1000.0)
                p_price = 150.0 + dist
            else:
                p_delta = min(-0.01, -0.5 + abs(dist) / 1000.0)
                p_price = max(1.0, 150.0 - abs(dist) * 0.6)

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

            # Select both CE and PE nearest to 20 delta
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


def main():
    print("="*65)
    print("        LAUNCHING DHAN AUTOMATED STRANGLE STRATEGY SYSTEM")
    print("="*65)

    # 1. Run Live/Simulation Trading Bot
    print("\n--- Phase 1: Real-time Trading Bot Mode ---")
    bot = DhanOptionsStrangleBot(
        client_id=CLIENT_ID,
        access_token=ACCESS_TOKEN,
        live_trading=LIVE_TRADING,
        lot_size=DEFAULT_LOT_SIZE,
        quantity_lots=QUANTITY_LOTS,
        entry_time_str=ENTRY_TIME_STR,
        exit_time_str=EXIT_TIME_STR,
        poll_interval_seconds=POLL_INTERVAL_SECONDS
    )

    # Run the bot in simulation context first to ensure everything works
    try:
        # In Jupyter, this starts the async event loop immediately
        asyncio.run(bot.run())
    except Exception as ex:
        logger.error(f"Error during real-time trading loop execution: {ex}")

    # 2. Run Historical Backtest
    if RUN_BACKTEST:
        print("\n--- Phase 2: Historical Backtesting Engine ---")
        print("Generating mock options data for historical backtest (24000 ATM base)...")
        dummy_df = DhanOptionsStrangleBacktester.generate_dummy_historical_data("2026-07-01", "2026-07-28")

        backtester = DhanOptionsStrangleBacktester(
            lot_size=DEFAULT_LOT_SIZE,
            quantity_lots=QUANTITY_LOTS,
            entry_time_str=ENTRY_TIME_STR,
            exit_time_str=EXIT_TIME_STR
        )
        print("Running historical backtest simulation...")
        results = backtester.run_backtest(dummy_df)

        if "error" in results:
            print(f"Backtest failed: {results['error']}")
        else:
            print("\n" + "*"*50)
            print("            BACKTEST PERFORMANCE RESULTS")
            print("*"*50)
            for metric, val in results['metrics'].items():
                print(f"{metric.replace('_', ' ').title():<28}: {val}")
            print("*"*50)
            print("\nBacktest executed successfully. All metrics computed.")


if __name__ == "__main__":
    main()
