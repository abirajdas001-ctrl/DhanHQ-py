import unittest
import datetime
import pandas as pd
from dhan_options_bot import DhanOptionsStrangleBot
from backtest_options import DhanOptionsStrangleBacktester

class TestDhanOptionsBot(unittest.TestCase):
    def setUp(self):
        # Initialize bot in simulation mode so no API keys are required
        self.bot = DhanOptionsStrangleBot(
            client_id="MOCK_ID",
            access_token="MOCK_TOKEN",
            live_trading=False,
            lot_size=65, # Updated lot size to 65
            quantity_lots=2
        )

    def test_tuesday_expiry_skip_rule(self):
        """
        Verify that Tuesday is recognized as an expiry day and skipped.
        """
        # Test a Tuesday (e.g. Sept 2, 2025 is Tuesday)
        tuesday_date = datetime.date(2025, 9, 2)
        self.assertTrue(self.bot.is_expiry_day_to_skip(tuesday_date))

        # Test a Monday (e.g. Sept 1, 2025 is Monday)
        monday_date = datetime.date(2025, 9, 1)
        self.assertFalse(self.bot.is_expiry_day_to_skip(monday_date))

    def test_leg_selection_logic(self):
        """
        Test that options leg selection picks exact target deltas and premiums.
        Both CE and PE must be selected around 20 delta.
        """
        mock_chain = self.bot.generate_mock_option_chain()
        selected = self.bot.select_options_legs(mock_chain)

        self.assertIsNotNone(selected)

        # Sell Call near 20 delta
        # Sell Put near 20 delta
        self.assertAlmostEqual(selected["sold_call"]["delta"], 0.20, delta=0.08)
        self.assertAlmostEqual(abs(selected["sold_put"]["delta"]), 0.20, delta=0.08)

        # Bought Call premium ~50% of sold Call premium
        self.assertAlmostEqual(selected["bought_call"]["last_price"], selected["sold_call"]["last_price"] * 0.5, delta=15.0)
        # Bought Put premium ~50% of sold Put premium
        self.assertAlmostEqual(selected["bought_put"]["last_price"], selected["sold_put"]["last_price"] * 0.5, delta=15.0)

    def test_sl_tp_calculations(self):
        """
        Check that Daily Stop-Loss (SL) and Daily Take-Profit (TP) are calculated correctly.
        """
        mock_chain = self.bot.generate_mock_option_chain()
        selected = self.bot.select_options_legs(mock_chain)

        self.assertTrue(self.bot.execute_strangle_positions(selected))

        # Verify total net premium calculation
        net_prem = (selected["sold_call"]["last_price"] + selected["sold_put"]["last_price"]) - (selected["bought_call"]["last_price"] + selected["bought_put"]["last_price"])
        self.assertAlmostEqual(self.bot.total_net_premium_collected, net_prem)

        # Verify SL = total net premium collected / 2
        self.assertAlmostEqual(self.bot.max_sl_points, net_prem / 2.0)
        # Verify TP = SL * 0.5
        self.assertAlmostEqual(self.bot.daily_tp_points, (net_prem / 2.0) * 0.5)


class TestDhanStrangleBacktester(unittest.TestCase):
    def test_backtest_runner_and_metrics(self):
        """
        Test the backtester's ability to run on historical data and produce correct metrics.
        """
        backtester = DhanOptionsStrangleBacktester(lot_size=65, quantity_lots=2)

        # Create a self-contained mock dataframe of option chains to prevent network requests during unit tests
        records = []
        for strike in [23600, 23800, 24000, 24200, 24400]:
            records.append({
                "date": "2024-01-03", "time": "09:45", "underlying_price": 24000.0, "strike": strike, "option_type": "CE",
                "premium": 150.0 - (strike - 24000) * 0.5, "delta": max(0.01, 0.5 - (strike - 24000) / 1000.0)
            })
            records.append({
                "date": "2024-01-03", "time": "09:45", "underlying_price": 24000.0, "strike": strike, "option_type": "PE",
                "premium": 150.0 + (strike - 24000) * 0.5, "delta": min(-0.01, -0.5 + (strike - 24000) / 1000.0)
            })
            records.append({
                "date": "2024-01-03", "time": "14:00", "underlying_price": 24000.0, "strike": strike, "option_type": "CE",
                "premium": 150.0 - (strike - 24000) * 0.5, "delta": max(0.01, 0.5 - (strike - 24000) / 1000.0)
            })
            records.append({
                "date": "2024-01-03", "time": "14:00", "underlying_price": 24000.0, "strike": strike, "option_type": "PE",
                "premium": 150.0 + (strike - 24000) * 0.5, "delta": min(-0.01, -0.5 + (strike - 24000) / 1000.0)
            })

        dummy_data = pd.DataFrame(records)
        results = backtester.run_backtest(dummy_data)

        self.assertNotIn("error", results)
        self.assertIn("metrics", results)
        self.assertIn("trades", results)

        metrics = results["metrics"]
        self.assertGreaterEqual(metrics["total_trades"], 1)
        self.assertIn("win_rate_percent", metrics)
        self.assertIn("total_net_pnl_amount", metrics)


if __name__ == "__main__":
    unittest.main()
