"""
Funding actor following NautilusTrader patterns.

Handles funding rate events and payments through the MessageBus system.
"""

from nautilus_trader.common.actor import Actor
from rich.console import Console

console = Console()


class FundingActor(Actor):
    """
    Native NautilusTrader Actor for funding rate tracking.
    
    Handles funding rate events and payments through the MessageBus system.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._funding_events_received = 0
        self._total_funding_cost = 0.0
        
        console.print("[green]✅ Native FundingActor initialized[/green]")

    def on_start(self) -> None:
        """Called when the actor starts."""
        # NTPA: Use NT-native logging
        self.log.info("FundingActor started")
        console.print("[blue]🚀 FundingActor started and ready for funding events[/blue]")

    def on_stop(self) -> None:
        """Called when the actor stops."""
        self.log.info(f"FundingActor stopped - {self._funding_events_received} events processed")
        console.print(f"[yellow]⏹️ FundingActor stopped - {self._funding_events_received} events processed[/yellow]")

    def on_reset(self) -> None:
        """Called when the actor resets."""
        self._funding_events_received = 0
        self._total_funding_cost = 0.0
        self.log.info("FundingActor reset")
        console.print("[blue]🔄 FundingActor reset[/blue]")

    def on_data(self, data) -> None:
        """Handle funding rate events."""
        try:
            from nautilus_test.funding.data import FundingPaymentEvent
            if isinstance(data, FundingPaymentEvent):
                self._funding_events_received += 1
                self._total_funding_cost += float(data.payment_amount)
                
                console.print(
                    f"[cyan]💰 Funding: ${float(data.payment_amount):.4f} "
                    f"({'payment' if data.is_payment else 'receipt'})[/cyan]"
                )
        except ImportError:
            pass  # Funding system not available

    @property
    def total_funding_cost(self) -> float:
        """Get total funding cost accumulated."""
        return self._total_funding_cost

    @property
    def events_received(self) -> int:
        """Get number of funding events received."""
        return self._funding_events_received
