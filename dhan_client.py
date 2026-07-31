import logging
from typing import Optional
from dhanhq import DhanContext, dhanhq

logger = logging.getLogger("DhanClient")

class DhanClient:
    """
    A basic client structure for initializing and interacting with the Dhan HQ API.
    Handles authentication and provides direct access to the underlying dhanhq SDK client.
    """
    def __init__(
        self,
        client_id: str,
        access_token: str,
        live_trading: bool = False
    ):
        self.client_id = client_id
        self.access_token = access_token
        self.live_trading = live_trading

        self.ctx: Optional[DhanContext] = None
        self.dhan: Optional[dhanhq] = None

        if self.live_trading:
            if not self.client_id or not self.access_token or "YOUR_" in self.client_id:
                raise ValueError("Valid Client ID and Access Token are required for LIVE trading!")
            self.ctx = DhanContext(self.client_id, self.access_token)
            self.dhan = dhanhq(self.ctx)
            logger.info("DhanHQ client initialized successfully for LIVE trading.")
        else:
            logger.info("DhanHQ client initialized in SIMULATION / PAPER trading mode.")

    def get_client(self) -> Optional[dhanhq]:
        """
        Returns the initialized dhanhq instance, or None if in simulation mode.
        """
        return self.dhan
