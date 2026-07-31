import unittest
from dhan_client import DhanClient

class TestDhanClient(unittest.TestCase):
    def test_simulation_mode_initialization(self):
        """
        Verify that DhanClient initializes properly in simulation mode
        even without valid keys, and does not instantiate the SDK context or client.
        """
        client = DhanClient(client_id="MOCK_ID", access_token="MOCK_TOKEN", live_trading=False)
        self.assertFalse(client.live_trading)
        self.assertIsNone(client.get_client())
        self.assertIsNone(client.ctx)

    def test_live_mode_missing_keys(self):
        """
        Verify that DhanClient raises ValueError when initializing in live mode
        with missing or placeholder keys.
        """
        with self.assertRaises(ValueError):
            DhanClient(client_id="", access_token="", live_trading=True)

        with self.assertRaises(ValueError):
            DhanClient(client_id="YOUR_CLIENT_ID", access_token="MOCK_TOKEN", live_trading=True)

    def test_live_mode_initialization(self):
        """
        Verify that DhanClient initializes DhanContext and dhanhq when proper keys are supplied in live mode.
        """
        # Since initializing with dummy credentials might hit the SDK's internal context creation,
        # we can verify that the objects are instantiated.
        client = DhanClient(client_id="VALID_ID_123", access_token="VALID_TOKEN_123", live_trading=True)
        self.assertTrue(client.live_trading)
        self.assertIsNotNone(client.ctx)
        self.assertIsNotNone(client.get_client())

if __name__ == "__main__":
    unittest.main()
