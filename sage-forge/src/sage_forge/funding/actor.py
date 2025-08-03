"""
SAGE-Forge enhanced funding actor for NautilusTrader integration.

Provides bulletproof funding rate event handling with SAGE-Forge configuration,
professional analytics, and seamless NautilusTrader MessageBus integration.
"""

from nautilus_trader.common.actor import Actor
from rich.console import Console

from sage_forge.core.config import get_config
from sage_forge.funding.data import FundingPaymentEvent

console = Console()


class FundingActor(Actor):
    """
    SAGE-Forge enhanced NautilusTrader Actor for funding rate tracking and analytics.
    
    Features:
    - NautilusTrader-native MessageBus integration
    - SAGE-Forge configuration system integration
    - Professional funding cost tracking and analytics
    - Real-time P&L impact monitoring
    - Enhanced logging and performance reporting
    """

    def __init__(self, config=None):
        super().__init__(config)
        
        # SAGE-Forge configuration integration
        self.sage_config = get_config()
        funding_config = self.sage_config.get_funding_config()
        
        # Enhanced tracking with SAGE-Forge analytics
        self._funding_events_received = 0
        self._total_funding_cost = 0.0
        self._total_funding_received = 0.0
        self._funding_payments_count = 0
        self._funding_receipts_count = 0
        
        # SAGE-Forge performance tracking
        self._largest_payment = 0.0
        self._largest_receipt = 0.0
        self._funding_rate_history = []
        
        # Alert thresholds from SAGE-Forge config
        self._alert_threshold = funding_config.get("alert_threshold", 50.0)
        self._daily_limit = funding_config.get("daily_limit", 500.0)
        
        console.print("[green]✅ SAGE-Forge FundingActor initialized with enhanced analytics[/green]")

    def on_start(self) -> None:
        """Start SAGE-Forge funding actor with configuration logging."""
        self.log.info("SAGE-Forge FundingActor started")
        
        # Log SAGE-Forge configuration
        funding_config = self.sage_config.get_funding_config()
        self.log.info(f"Funding config: alert_threshold=${self._alert_threshold}, daily_limit=${self._daily_limit}")
        
        console.print("[blue]🚀 SAGE-Forge FundingActor ready for funding events and analytics[/blue]")

    def on_stop(self) -> None:
        """Stop with comprehensive SAGE-Forge performance reporting."""
        self.log.info(f"SAGE-Forge FundingActor stopped - {self._funding_events_received} events processed")
        
        # Enhanced performance summary
        net_funding_cost = self._total_funding_cost - self._total_funding_received
        console.print(f"[yellow]⏹️ SAGE-Forge FundingActor stopped[/yellow]")
        console.print(f"[cyan]📊 Total events: {self._funding_events_received}[/cyan]")
        console.print(f"[cyan]💰 Payments: {self._funding_payments_count} (${self._total_funding_cost:.4f})[/cyan]")
        console.print(f"[cyan]💰 Receipts: {self._funding_receipts_count} (${self._total_funding_received:.4f})[/cyan]")
        console.print(f"[cyan]📈 Net cost: ${net_funding_cost:.4f}[/cyan]")
        
        if self._funding_rate_history:
            avg_rate = sum(self._funding_rate_history) / len(self._funding_rate_history)
            console.print(f"[dim]📊 Average funding rate: {avg_rate*100:.4f}%[/dim]")

    def on_reset(self) -> None:
        """Reset with SAGE-Forge state cleanup."""
        self._funding_events_received = 0
        self._total_funding_cost = 0.0
        self._total_funding_received = 0.0
        self._funding_payments_count = 0
        self._funding_receipts_count = 0
        self._largest_payment = 0.0
        self._largest_receipt = 0.0
        self._funding_rate_history.clear()
        
        self.log.info("SAGE-Forge FundingActor reset")
        console.print("[blue]🔄 SAGE-Forge FundingActor reset with analytics cleared[/blue]")

    def on_data(self, data) -> None:
        """Handle funding rate events with SAGE-Forge enhanced analytics."""
        try:
            # Handle SAGE-Forge FundingPaymentEvent
            if isinstance(data, FundingPaymentEvent):
                self._process_funding_payment(data)
            else:
                # Try to handle legacy funding events for backward compatibility
                self._try_handle_legacy_funding(data)
                
        except Exception as e:
            self.log.error(f"SAGE-Forge FundingActor error processing data: {e}")
            console.print(f"[red]❌ Error processing funding data: {e}[/red]")

    def _process_funding_payment(self, event: FundingPaymentEvent) -> None:
        """Process SAGE-Forge funding payment event with comprehensive analytics."""
        self._funding_events_received += 1
        payment_amount = float(event.payment_amount)
        
        # Track funding rate history for analytics
        self._funding_rate_history.append(event.funding_rate)
        
        if event.is_payment:
            # Funding payment (cost)
            self._funding_payments_count += 1
            self._total_funding_cost += abs(payment_amount)
            self._largest_payment = max(self._largest_payment, abs(payment_amount))
            
            # SAGE-Forge alert system
            if abs(payment_amount) > self._alert_threshold:
                console.print(f"[red]🚨 Large funding payment: ${abs(payment_amount):.4f} (threshold: ${self._alert_threshold})[/red]")
                self.log.warning(f"Large funding payment: ${abs(payment_amount):.4f} for {event.instrument_id}")
            
            console.print(
                f"[red]💸 Funding payment: ${abs(payment_amount):.4f} "
                f"@ {event.funding_rate*100:.4f}% on {event.instrument_id}[/red]"
            )
        else:
            # Funding receipt (income)
            self._funding_receipts_count += 1
            self._total_funding_received += abs(payment_amount)
            self._largest_receipt = max(self._largest_receipt, abs(payment_amount))
            
            console.print(
                f"[green]💰 Funding receipt: ${abs(payment_amount):.4f} "
                f"@ {event.funding_rate*100:.4f}% on {event.instrument_id}[/green]"
            )
        
        # Daily limit monitoring
        net_cost = self._total_funding_cost - self._total_funding_received
        if net_cost > self._daily_limit:
            console.print(f"[red]🚨 Daily funding limit exceeded: ${net_cost:.2f} (limit: ${self._daily_limit})[/red]")
            self.log.warning(f"Daily funding limit exceeded: ${net_cost:.2f}")

    def _try_handle_legacy_funding(self, data) -> None:
        """Handle legacy funding events for backward compatibility."""
        try:
            # Try to import legacy funding event (nautilus_test compatibility)
            from nautilus_test.funding.data import FundingPaymentEvent as LegacyFundingPaymentEvent
            
            if isinstance(data, LegacyFundingPaymentEvent):
                self._funding_events_received += 1
                payment_amount = float(data.payment_amount)
                self._total_funding_cost += abs(payment_amount)
                
                console.print(
                    f"[cyan]💰 Legacy funding: ${abs(payment_amount):.4f} "
                    f"({'payment' if data.is_payment else 'receipt'})[/cyan]"
                )
        except ImportError:
            # Legacy funding system not available
            pass

    # SAGE-Forge enhanced properties and analytics methods
    
    @property
    def total_funding_cost(self) -> float:
        """Get total funding cost accumulated (payments only)."""
        return self._total_funding_cost
    
    @property
    def total_funding_received(self) -> float:
        """Get total funding received (receipts only)."""
        return self._total_funding_received

    @property
    def net_funding_cost(self) -> float:
        """Get net funding cost (payments - receipts)."""
        return self._total_funding_cost - self._total_funding_received

    @property
    def events_received(self) -> int:
        """Get number of funding events received."""
        return self._funding_events_received
    
    @property
    def funding_payments_count(self) -> int:
        """Get number of funding payments made."""
        return self._funding_payments_count
    
    @property
    def funding_receipts_count(self) -> int:
        """Get number of funding receipts received."""
        return self._funding_receipts_count

    def get_funding_analytics(self) -> dict:
        """Get comprehensive SAGE-Forge funding analytics report."""
        net_cost = self.net_funding_cost
        avg_payment = self._total_funding_cost / max(self._funding_payments_count, 1)
        avg_receipt = self._total_funding_received / max(self._funding_receipts_count, 1)
        avg_funding_rate = sum(self._funding_rate_history) / max(len(self._funding_rate_history), 1)
        
        return {
            "sage_forge_version": self.sage_config.version,
            "total_events": self._funding_events_received,
            "funding_payments": {
                "count": self._funding_payments_count,
                "total": self._total_funding_cost,
                "average": avg_payment,
                "largest": self._largest_payment,
            },
            "funding_receipts": {
                "count": self._funding_receipts_count,
                "total": self._total_funding_received,
                "average": avg_receipt,
                "largest": self._largest_receipt,
            },
            "net_funding_cost": net_cost,
            "average_funding_rate": avg_funding_rate,
            "average_funding_rate_bps": avg_funding_rate * 10000,
            "config": {
                "alert_threshold": self._alert_threshold,
                "daily_limit": self._daily_limit,
            },
            "risk_metrics": {
                "daily_limit_utilization": (net_cost / self._daily_limit * 100) if self._daily_limit > 0 else 0,
                "largest_single_cost": max(self._largest_payment, self._largest_receipt),
            }
        }

    def reset_analytics(self) -> None:
        """Reset analytics counters (useful for daily/session resets)."""
        self._funding_events_received = 0
        self._total_funding_cost = 0.0
        self._total_funding_received = 0.0
        self._funding_payments_count = 0
        self._funding_receipts_count = 0
        self._largest_payment = 0.0
        self._largest_receipt = 0.0
        self._funding_rate_history.clear()
        
        console.print("[blue]🔄 SAGE-Forge funding analytics reset[/blue]")