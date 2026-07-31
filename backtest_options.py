import datetime
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class DhanOptionsStrangleBacktester:
    """
    Backtesting Engine for the Dhan HQ Options Strangle Strategy.
    Simulates the strategy over a historical dataset.
    """
    def __init__(
        self,
        lot_size: int = 65, # Default lot size updated to 65 as requested
        quantity_lots: int = 2,
        entry_time_str: str = "09:45",
        exit_time_str: str = "14:00",
    ):
        self.lot_size = lot_size
        self.quantity_lots = quantity_lots
        self.qty_contracts = self.lot_size * self.quantity_lots
        self.entry_time = datetime.datetime.strptime(entry_time_str, "%H:%M").time()
        self.exit_time = datetime.datetime.strptime(exit_time_str, "%H:%M").time()

    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the backtest over the provided DataFrame.
        """
        if historical_data.empty:
            return {"error": "Historical dataset is empty."}

        # Normalize date column
        historical_data['date'] = pd.to_datetime(historical_data['date']).dt.date
        unique_dates = sorted(historical_data['date'].unique())

        trades = []
        daily_returns = []

        for date in unique_dates:
            # Skip Tuesday (Nifty Expiry Day as of Sept 2025)
            if date.weekday() == 1:
                continue

            day_data = historical_data[historical_data['date'] == date]
            entry_rows = day_data[day_data['time'].astype(str).str.startswith("09:45")]
            if entry_rows.empty:
                entry_rows = day_data

            # Split into calls and puts
            calls = entry_rows[entry_rows['option_type'] == 'CE']
            puts = entry_rows[entry_rows['option_type'] == 'PE']

            if calls.empty or puts.empty:
                continue

            # Selection Logic matching live bot (both 20 delta)
            # 1. Sell Call near 20 delta
            sold_call = calls.iloc[(calls['delta'].abs() - 0.20).abs().argsort()[:1]].iloc[0]
            # 2. Sell Put near 20 delta (updated from 30)
            sold_put = puts.iloc[(puts['delta'].abs() - 0.20).abs().argsort()[:1]].iloc[0]

            # 3. Hedge Call: Buy further OTM Call with premium ~50% of Sold Call premium
            further_calls = calls[calls['strike'] > sold_call['strike']]
            if further_calls.empty:
                further_calls = calls
            target_call_prem = sold_call['premium'] * 0.5
            bought_call = further_calls.iloc[(further_calls['premium'] - target_call_prem).abs().argsort()[:1]].iloc[0]

            # 4. Hedge Put: Buy further OTM Put with premium ~50% of Sold Put premium
            further_puts = puts[puts['strike'] < sold_put['strike']]
            if further_puts.empty:
                further_puts = puts
            target_put_prem = sold_put['premium'] * 0.5
            bought_put = further_puts.iloc[(further_puts['premium'] - target_put_prem).abs().argsort()[:1]].iloc[0]

            # Calculate total net premium collected
            net_premium_collected = (sold_call['premium'] + sold_put['premium']) - (bought_call['premium'] + bought_put['premium'])
            max_sl_points = net_premium_collected / 2.0
            daily_tp_points = max_sl_points * 0.5

            sc_id = sold_call['strike']
            sp_id = sold_put['strike']
            bc_id = bought_call['strike']
            bp_id = bought_put['strike']

            after_entry = day_data[day_data['time'].astype(str) >= "09:45"]
            unique_times = sorted(after_entry['time'].unique())

            exit_reason = "Time Exit (2:00 PM Square-Off)"
            exit_time = "14:00"
            exit_net_premium = net_premium_collected

            sc_exit_price = sold_call['premium']
            sp_exit_price = sold_put['premium']
            bc_exit_price = bought_call['premium']
            bp_exit_price = bought_put['premium']

            for t_val in unique_times:
                time_str = str(t_val)
                if time_str >= "14:00":
                    break

                # Get prices at this minute
                min_rows = after_entry[after_entry['time'] == t_val]
                sc_curr = min_rows[(min_rows['strike'] == sc_id) & (min_rows['option_type'] == 'CE')]
                sp_curr = min_rows[(min_rows['strike'] == sp_id) & (min_rows['option_type'] == 'PE')]
                bc_curr = min_rows[(min_rows['strike'] == bc_id) & (min_rows['option_type'] == 'CE')]
                bp_curr = min_rows[(min_rows['strike'] == bp_id) & (min_rows['option_type'] == 'PE')]

                if sc_curr.empty or sp_curr.empty or bc_curr.empty or bp_curr.empty:
                    continue

                sc_p = sc_curr['premium'].iloc[0]
                sp_p = sp_curr['premium'].iloc[0]
                bc_p = bc_curr['premium'].iloc[0]
                bp_p = bp_curr['premium'].iloc[0]

                curr_net_premium = (sc_p + sp_p) - (bc_p + bp_p)

                # Check SL
                if curr_net_premium >= (net_premium_collected + max_sl_points):
                    exit_reason = "Stop-Loss (SL) Hit"
                    exit_time = time_str
                    exit_net_premium = curr_net_premium
                    sc_exit_price, sp_exit_price, bc_exit_price, bp_exit_price = sc_p, sp_p, bc_p, bp_p
                    break

                # Check TP
                if curr_net_premium <= (net_premium_collected - daily_tp_points):
                    exit_reason = "Take-Profit (TP) Hit"
                    exit_time = time_str
                    exit_net_premium = curr_net_premium
                    sc_exit_price, sp_exit_price, bc_exit_price, bp_exit_price = sc_p, sp_p, bc_p, bp_p
                    break

                # Update prices for Time exit fallback
                sc_exit_price, sp_exit_price, bc_exit_price, bp_exit_price = sc_p, sp_p, bc_p, bp_p
                exit_net_premium = curr_net_premium

            p_and_l_points = net_premium_collected - exit_net_premium
            p_and_l_amount = p_and_l_points * self.qty_contracts

            trades.append({
                "date": date,
                "sold_call_strike": sc_id,
                "sold_put_strike": sp_id,
                "bought_call_strike": bc_id,
                "bought_put_strike": bp_id,
                "net_premium_collected": net_premium_collected,
                "exit_net_premium": exit_net_premium,
                "p_and_l_points": p_and_l_points,
                "p_and_l_amount": p_and_l_amount,
                "exit_time": exit_time,
                "exit_reason": exit_reason
            })
            daily_returns.append(p_and_l_amount)

        if not trades:
            return {"error": "No viable trades were executed in the backtest period."}

        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['p_and_l_amount'].sum()
        win_trades = trades_df[trades_df['p_and_l_amount'] > 0]
        loss_trades = trades_df[trades_df['p_and_l_amount'] <= 0]
        win_rate = (len(win_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        # Calculate max drawdown
        cumulative_pnl = trades_df['p_and_l_amount'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdowns = running_max - cumulative_pnl
        max_drawdown = drawdowns.max()

        # Calculate Sharpe Ratio
        std_pnl = trades_df['p_and_l_amount'].std()
        mean_pnl = trades_df['p_and_l_amount'].mean()
        sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0

        return {
            "metrics": {
                "total_trades": len(trades_df),
                "win_trades": len(win_trades),
                "loss_trades": len(loss_trades),
                "win_rate_percent": round(win_rate, 2),
                "total_net_pnl_amount": round(total_pnl, 2),
                "max_drawdown_amount": round(max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "average_trade_pnl": round(mean_pnl, 2) if len(trades_df) > 0 else 0.0
            },
            "trades": trades_df
        }

    @staticmethod
    def generate_dummy_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
        """
        Helper method to generate mock options intraday data for Nifty 50.
        """
        date_range = pd.date_range(start_date, end_date)
        records = []

        for d in date_range:
            day_str = d.strftime("%Y-%m-%d")
            for strike in range(23600, 24400, 50):
                dist = strike - 24000
                c_delta = max(0.01, 0.5 - dist * 0.001)
                p_delta = min(-0.01, -0.5 + dist * 0.001)

                c_prem_945 = max(5.0, 150 - dist * 0.5 + random.uniform(-5, 5))
                p_prem_945 = max(5.0, 150 + dist * 0.5 + random.uniform(-5, 5))

                records.append({
                    "date": day_str, "time": "09:45", "underlying_price": 24000.0, "strike": strike, "option_type": "CE", "premium": c_prem_945, "delta": c_delta
                })
                records.append({
                    "date": day_str, "time": "09:45", "underlying_price": 24000.0, "strike": strike, "option_type": "PE", "premium": p_prem_945, "delta": p_delta
                })

                market_move = random.choice([-100, -50, 0, 50, 100])
                dist_1130 = strike - (24000 + market_move)
                c_prem_1130 = max(2.0, 150 - dist_1130 * 0.5 + random.uniform(-10, 10))
                p_prem_1130 = max(2.0, 150 + dist_1130 * 0.5 + random.uniform(-10, 10))

                records.append({
                    "date": day_str, "time": "11:30", "underlying_price": 24000.0 + market_move, "strike": strike, "option_type": "CE", "premium": c_prem_1130, "delta": c_delta
                })
                records.append({
                    "date": day_str, "time": "11:30", "underlying_price": 24000.0 + market_move, "strike": strike, "option_type": "PE", "premium": p_prem_1130, "delta": p_delta
                })

                market_move_final = market_move + random.choice([-50, 0, 50])
                dist_1400 = strike - (24000 + market_move_final)
                c_prem_1400 = max(1.0, 150 - dist_1400 * 0.5 + random.uniform(-5, 5))
                p_prem_1400 = max(1.0, 150 + dist_1400 * 0.5 + random.uniform(-5, 5))

                records.append({
                    "date": day_str, "time": "14:00", "underlying_price": 24000.0 + market_move_final, "strike": strike, "option_type": "CE", "premium": c_prem_1400, "delta": c_delta
                })
                records.append({
                    "date": day_str, "time": "14:00", "underlying_price": 24000.0 + market_move_final, "strike": strike, "option_type": "PE", "premium": p_prem_1400, "delta": p_delta
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating mock historical options data for backtesting...")
    dummy_df = DhanOptionsStrangleBacktester.generate_dummy_historical_data("2026-07-01", "2026-07-28")

    backtester = DhanOptionsStrangleBacktester(lot_size=65, quantity_lots=2)
    print("Running Options Strangle Strategy backtest...")
    results = backtester.run_backtest(dummy_df)

    if "error" in results:
        print(f"Backtest Failed: {results['error']}")
    else:
        print("\n" + "="*40)
        print("          BACKTEST PERFORMANCE METRICS")
        print("="*40)
        for metric, val in results['metrics'].items():
            print(f"{metric.replace('_', ' ').title():<30}: {val}")
        print("="*40)
        print("\nFirst 5 Executed Trades:")
        print(results['trades'].head())
