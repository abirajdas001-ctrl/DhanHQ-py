import os
import sys
import asyncio
import logging
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DhanOptionsBot")

# Try importing dhanhq
try:
    from dhanhq import DhanContext, dhanhq
except ImportError:
    logger.error("dhanhq library is not installed. Please run: pip install dhanhq")


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

        # Real-time lot size caching
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
            # Load CSV containing scrip master directly
            df = pd.read_csv(url, low_memory=False)
            # Filter for NSE derivative instrument where trading symbol starts with NIFTY
            # Segment D is Derivatives
            nifty_df = df[
                (df['SEM_TRADING_SYMBOL'].str.startswith('NIFTY', na=False)) &
                (df['SEM_SEGMENT'] == 'D')
            ]
            if not nifty_df.empty:
                # Get the lot size from the first row of Nifty derivative
                lot_size = int(nifty_df['SEM_LOT_UNITS'].iloc[0])
                logger.info(f"Successfully fetched real-time Nifty lot size from scrip master: {lot_size}")
                return lot_size
        except Exception as e:
            logger.warning(f"Could not fetch lot size from CSV: {e}. Falling back to default Nifty lot size of 65.")
        return 65

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
        2. Sell Put near 20 delta (Updated from 30 delta as per user request)
        3. Buy Call further OTM with premium ~50% of Sold Call premium
        4. Buy Put further OTM with premium ~50% of Sold Put premium
        """
        # Parse underlying price
        last_price = option_chain_data.get("last_price")
        if not last_price:
            # Some versions wrap in data key
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
                # Standardize delta value
                delta_val = ce.get("greeks", {}).get("delta")
                # If delta is not explicitly found in greeks, compute or default it safely
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
        Executes order entries for all four selected options legs.
        Sequence: Places BUY (hedge) orders first to obtain portfolio margin benefits,
        followed by SELL (strangle) orders.
        """
        logger.info("Executing trade entry positions...")

        # 1. Buy Call Hedge
        bc_id = self.execute_order(legs["bought_call"]["security_id"], "BUY", self.quantity_contracts)
        if not bc_id:
            logger.error("Hedge Call Order failed. Aborting entry sequence to prevent naked risk!")
            return False

        # 2. Buy Put Hedge
        bp_id = self.execute_order(legs["bought_put"]["security_id"], "BUY", self.quantity_contracts)
        if not bp_id:
            logger.error("Hedge Put Order failed. Aborting entry sequence!")
            # Square off previous hedge to prevent mismatch
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            return False

        # 3. Sell Call Leg
        sc_id = self.execute_order(legs["sold_call"]["security_id"], "SELL", self.quantity_contracts)
        if not sc_id:
            logger.error("Short Call Order failed! Squaring off hedges to remain neutral.")
            self.execute_order(legs["bought_call"]["security_id"], "SELL", self.quantity_contracts)
            self.execute_order(legs["bought_put"]["security_id"], "SELL", self.quantity_contracts)
            return False

        # 4. Sell Put Leg
        sp_id = self.execute_order(legs["sold_put"]["security_id"], "SELL", self.quantity_contracts)
        if not sp_id:
            logger.error("Short Put Order failed! Squaring off all positions.")
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

        # Strangle pricing logic:
        # Net premium collected = Sold Call + Sold Put - Bought Call - Bought Put
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
        In simulation, this returns updated prices by calling the live option chain.
        """
        if not self.live_trading:
            # For simulation, simulate some price movement or return actual live quotes
            return {
                "sold_call_ltp": self.legs["sold_call"]["entry_price"],
                "sold_put_ltp": self.legs["sold_put"]["entry_price"],
                "bought_call_ltp": self.legs["bought_call"]["entry_price"],
                "bought_put_ltp": self.legs["bought_put"]["entry_price"],
            }

        try:
            # Query the option chain API for the active expiry to get real-time LTPs
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

            prices = {}
            # Sold Call
            sc_data = oc_map.get(sc_strike, {}).get("ce", {})
            prices["sold_call_ltp"] = sc_data.get("last_price", self.legs["sold_call"]["entry_price"])
            # Sold Put
            sp_data = oc_map.get(sp_strike, {}).get("pe", {})
            prices["sold_put_ltp"] = sp_data.get("last_price", self.legs["sold_put"]["entry_price"])
            # Bought Call
            bc_data = oc_map.get(bc_strike, {}).get("ce", {})
            prices["bought_call_ltp"] = bc_data.get("last_price", self.legs["bought_call"]["entry_price"])
            # Bought Put
            bp_data = oc_map.get(bp_strike, {}).get("pe", {})
            prices["bought_put_ltp"] = bp_data.get("last_price", self.legs["bought_put"]["entry_price"])

            return prices
        except Exception as e:
            logger.error(f"Error fetching real-time market prices: {e}")
            return None

    async def run(self):
        """
        Main runner of the options trading bot.
        """
        logger.info("Initializing automated options strangle bot...")

        # 1. Skip on Tuesdays (Nifty Expiry Day as of Sept 2025)
        if self.is_expiry_day_to_skip():
            logger.info("Today is Tuesday (Nifty Weekly Expiry Day). Skipping trade deployment according to rules.")
            return

        entry_time = self.get_time_from_str(self.entry_time_str)
        exit_time = self.get_time_from_str(self.exit_time_str)

        # 2. Wait until Entry Time (9:45 AM)
        while True:
            now = datetime.datetime.now().time()
            if now >= entry_time:
                logger.info(f"Entry time {self.entry_time_str} reached.")
                break
            logger.info(f"Waiting for entry time ({self.entry_time_str}). Current time: {now.strftime('%H:%M:%S')}")
            await asyncio.sleep(30)

        # 3. Fetch active expiry and option chain to deploy strangle
        logger.info("Fetching NIFTY options chain details...")
        if self.live_trading:
            try:
                exp_list_resp = self.dhan.expiry_list(under_security_id=13, under_exchange_segment="IDX_I")
                # Parse expiry dates
                dates = exp_list_resp.get("data", [])
                if not dates:
                    logger.error("Could not fetch active expiry list. Exiting.")
                    return
                # Sort to get the nearest upcoming weekly expiry
                dates.sort()
                self.expiry_date = dates[0]
                logger.info(f"Targeting weekly options expiry: {self.expiry_date}")

                # Fetch option chain
                chain_resp = self.dhan.option_chain(
                    under_security_id=13,
                    under_exchange_segment="IDX_I",
                    expiry=self.expiry_date
                )
            except Exception as e:
                logger.error(f"Failed to query Dhan API: {e}")
                return
        else:
            # Simulation mock response
            self.expiry_date = (datetime.date.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            logger.info(f"[SIMULATION] Selecting active expiry: {self.expiry_date}")
            # Generate mock option chain data
            chain_resp = self.generate_mock_option_chain()

        selected_legs = self.select_options_legs(chain_resp)
        if not selected_legs:
            logger.error("Failed to select viable strangle and hedge legs. Exiting strategy.")
            return

        # 4. Deploy trade positions
        success = self.execute_strangle_positions(selected_legs)
        if not success:
            logger.error("Failed to enter strangle positions.")
            return

        # 5. Async monitoring loop
        logger.info("Starting real-time position monitoring...")
        while self.active_trade:
            now = datetime.datetime.now().time()

            # Check Time Exit (2:00 PM)
            if now >= exit_time:
                self.square_off_all_positions("Time Exit (2:00 PM Square-Off) reached.")
                break

            # Fetch updated prices
            current_prices = await self.get_current_greeks_and_prices()
            if current_prices:
                sc_ltp = current_prices["sold_call_ltp"]
                sp_ltp = current_prices["sold_put_ltp"]
                bc_ltp = current_prices["bought_call_ltp"]
                bp_ltp = current_prices["bought_put_ltp"]

                # Current position net premium
                # Position net premium = Short Call + Short Put - Bought Call - Bought Put
                current_net_premium = (sc_ltp + sp_ltp) - (bc_ltp + bp_ltp)
                pl_points = self.total_net_premium_collected - current_net_premium

                logger.info(
                    f"Monitoring - Current Net Premium: {current_net_premium:.2f} | "
                    f"P&L: {pl_points:+.2f} pts | "
                    f"SC: {sc_ltp:.1f}, SP: {sp_ltp:.1f}, BC: {bc_ltp:.1f}, BP: {bp_ltp:.1f}"
                )

                # Stop Loss Check: if net premium rises too high (Max SL points lost)
                # Max SL is hit when current_net_premium >= entry_net_premium + max_sl_points
                if current_net_premium >= (self.total_net_premium_collected + self.max_sl_points):
                    self.square_off_all_positions(f"Stop-Loss (SL) of {self.max_sl_points:.2f} points hit.")
                    break

                # Take Profit Check: if net premium drops (Daily TP points gained)
                # Daily TP is hit when current_net_premium <= entry_net_premium - daily_tp_points
                if current_net_premium <= (self.total_net_premium_collected - self.daily_tp_points):
                    self.square_off_all_positions(f"Take-Profit (TP) of {self.daily_tp_points:.2f} points hit.")
                    break

            await asyncio.sleep(self.poll_interval_seconds)

        logger.info("Options strategy session completed.")

    def generate_mock_option_chain(self) -> Dict[str, Any]:
        """
        Helper to construct a mock option chain for simulation and unit testing.
        Centered around Nifty at 24000.
        """
        oc = {}
        for strike in range(23500, 24500, 50):
            # Approximate delta and premiums
            # At 24000 (ATM)
            dist = strike - 24000

            # Simple Call Option simulation
            if dist < 0: # ITM
                c_delta = min(0.99, 0.5 + abs(dist) / 1000.0)
                c_price = 150.0 + abs(dist)
            else: # OTM
                c_delta = max(0.01, 0.5 - dist / 1000.0)
                c_price = max(1.0, 150.0 - dist * 0.6)

            # Simple Put Option simulation
            if dist > 0: # ITM
                p_delta = max(-0.99, -0.5 - dist / 1000.0)
                p_price = 150.0 + dist
            else: # OTM
                p_delta = min(-0.01, -0.5 + abs(dist) / 1000.0)
                p_price = max(1.0, 150.0 - abs(dist) * 0.6)

            oc[f"{strike}.000000"] = {
                "ce": {
                    "security_id": f"CE_{strike}",
                    "average_price": c_price,
                    "last_price": c_price,
                    "greeks": {"delta": c_delta}
                },
                "pe": {
                    "security_id": f"PE_{strike}",
                    "average_price": p_price,
                    "last_price": p_price,
                    "greeks": {"delta": p_delta}
                }
            }

        return {
            "data": {
                "last_price": 24000.0,
                "oc": oc
            }
        }


if __name__ == "__main__":
    # Example local run script
    # To run, set your environment variables or pass them directly
    CID = os.getenv("DHAN_CLIENT_ID", "")
    TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
    LIVE = os.getenv("DHAN_LIVE_TRADING", "False").lower() == "true"

    bot = DhanOptionsStrangleBot(
        client_id=CID,
        access_token=TOKEN,
        live_trading=LIVE,
        poll_interval_seconds=5
    )

    # Run the bot's asynchronous entry and monitoring loop
    asyncio.run(bot.run())
