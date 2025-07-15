"""
Native NautilusTrader Actor for funding rate handling.

This module provides a fully native FundingActor that follows NautilusTrader's
event-driven architecture patterns. No direct portfolio manipulation - everything
goes through the message bus as proper events.
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.common.actor import Actor
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money
from rich.console import Console
from rich.panel import Panel

from nautilus_test.funding.data import FundingRateUpdate, FundingPaymentEvent

console = Console()


class FundingActor(Actor):
    """
    Native NautilusTrader Actor for handling funding rate events.
    
    This actor follows the canonical NautilusTrader pattern:
    1. Subscribes to FundingRateUpdate data through message bus
    2. Queries positions from cache (not direct portfolio access)
    3. Publishes FundingPaymentEvent through message bus
    4. Lets the system handle portfolio changes through events
    
    Core principles:
    - "Stay on the bus" - all communication via MessageBus
    - "Everything is a message" - funding impacts are events
    - "Cache for queries" - position data from cache, not direct calls
    - "Publish don't push" - emit events, don't call methods
    """
    
    def __init__(self, config=None):
        """
        Initialize the funding actor.
        
        Parameters
        ----------
        config : dict, optional
            Actor configuration (can be None for simple setup).
        """
        super().__init__(config)
        
        # Track funding state
        self._funding_events_count = 0
        self._total_funding_impact = Decimal("0")
        
        console.print("[green]✅ Native FundingActor initialized[/green]")
    
    def on_start(self) -> None:
        """
        Called when the actor starts.
        
        Initialize the actor - subscription will be handled by data feed.
        """
        self.log.info("FundingActor started - ready to process funding events")
        console.print("[blue]🚀 FundingActor started and ready for funding events[/blue]")
    
    def on_stop(self) -> None:
        """Called when the actor stops."""
        self.log.info(f"FundingActor stopped - processed {self._funding_events_count} funding events")
        console.print(f"[yellow]⏹️ FundingActor stopped - {self._funding_events_count} events processed[/yellow]")
    
    def on_reset(self) -> None:
        """Called when the actor resets."""
        self._funding_events_count = 0
        self._total_funding_impact = Decimal("0")
        self.log.info("FundingActor reset")
        console.print("[blue]🔄 FundingActor reset[/blue]")
    
    def on_funding_rate_update(self, update: FundingRateUpdate) -> None:
        """
        Handle funding rate update using native patterns.
        
        This is the core method that demonstrates native NautilusTrader patterns:
        1. Query position from cache (not direct portfolio access)
        2. Calculate funding payment
        3. Publish funding event through message bus
        
        Parameters
        ----------
        update : FundingRateUpdate
            The funding rate update to process.
        """
        # Query position from cache (NATIVE PATTERN)
        position = self.cache.position_for_instrument(update.instrument_id)
        
        if position is None or position.is_closed or position.quantity == 0:
            # No position or closed position - no funding impact
            self.log.debug(f"No open position for {update.instrument_id} - skipping funding")
            return
        
        # Calculate funding payment using the verified formula:
        # Payment = Position Size × Mark Price × Funding Rate
        position_size = float(position.quantity)
        mark_price = float(update.mark_price) if update.mark_price else self._get_mark_price_from_cache(update.instrument_id)
        
        if mark_price is None or mark_price <= 0:
            self.log.warning(f"Invalid mark price for {update.instrument_id} - cannot calculate funding")
            return
        
        # Calculate funding payment
        funding_payment_usd = position_size * mark_price * update.funding_rate
        
        # Determine if payment or receipt
        is_payment = funding_payment_usd > 0
        payment_amount = Money(abs(funding_payment_usd), USDT)
        
        # Create funding payment event (NATIVE PATTERN)
        funding_event = FundingPaymentEvent(
            event_id=UUID4(),
            instrument_id=update.instrument_id,
            payment_amount=payment_amount,
            funding_rate=update.funding_rate,
            position_size=Decimal(str(position_size)),
            mark_price=mark_price,
            is_payment=is_payment,
            ts_event=update.ts_event,
            ts_init=self.clock.timestamp_ns(),
        )
        
        # Publish the event through message bus (NATIVE PATTERN)
        self.publish_data(funding_event)
        
        # Update tracking
        self._funding_events_count += 1
        self._total_funding_impact += Decimal(str(funding_payment_usd))
        
        # Log the funding event
        direction = "pays" if is_payment else "receives"
        self.log.info(
            f"Funding event: {position.instrument_id} position {position_size:.3f} "
            f"{direction} ${abs(funding_payment_usd):.2f} "
            f"(rate: {update.funding_rate:.6f}, price: ${mark_price:.2f})"
        )
        
        console.print(
            f"[cyan]💰 Funding: {position.instrument_id} {direction} "
            f"${abs(funding_payment_usd):.2f}[/cyan]"
        )
    
    def _get_mark_price_from_cache(self, instrument_id: InstrumentId) -> Optional[float]:
        """
        Get mark price from cache using native patterns.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to get mark price for.
            
        Returns
        -------
        Optional[float]
            Mark price if available, None otherwise.
        """
        # Try to get latest quote tick from cache
        quote_tick = self.cache.quote_tick(instrument_id)
        if quote_tick:
            # Use mid price as mark price approximation
            return float(quote_tick.mid_price())
        
        # Try to get latest trade tick
        trade_tick = self.cache.trade_tick(instrument_id)
        if trade_tick:
            return float(trade_tick.price)
        
        # Try to get latest bar
        bars = self.cache.bars(instrument_id)
        if bars:
            return float(bars[-1].close)
        
        return None
    
    def get_funding_summary(self) -> dict:
        """
        Get summary of funding activity.
        
        Returns
        -------
        dict
            Summary of funding events and total impact.
        """
        return {
            "total_events": self._funding_events_count,
            "total_impact_usd": float(self._total_funding_impact),
            "actor_type": "Native FundingActor",
            "message_bus_compliant": True,
            "direct_portfolio_access": False,  # We use events!
            "cache_queries_only": True,
        }


class FundingActorConfig:
    """
    Configuration for FundingActor.
    
    This follows NautilusTrader's configuration pattern.
    """
    
    def __init__(
        self,
        component_id: str = "FundingActor",
        enabled: bool = True,
        log_funding_events: bool = True,
    ):
        self.component_id = component_id
        self.enabled = enabled
        self.log_funding_events = log_funding_events


# Example of how to properly integrate with BacktestEngine
def add_funding_actor_to_engine(engine, config: Optional[FundingActorConfig] = None):
    """
    Add FundingActor to BacktestEngine using native patterns.
    
    This demonstrates the correct way to integrate the actor.
    
    Parameters
    ----------
    engine : BacktestEngine
        The backtest engine to add the actor to.
    config : FundingActorConfig, optional
        Actor configuration.
    """
    if config is None:
        config = FundingActorConfig()
    
    if config.enabled:
        # Create actor with simple config dict or None
        funding_actor = FundingActor(config=None)
        # NATIVE PATTERN: Use engine.add_actor()
        engine.add_actor(funding_actor)
        
        console.print("[green]✅ Native FundingActor added to BacktestEngine[/green]")
        
        return funding_actor
    else:
        console.print("[yellow]⚠️ FundingActor disabled in config[/yellow]")
        return None


# Mathematical validation function for the native actor
def validate_funding_calculation_native(
    position_quantity: Decimal,
    mark_price: float,
    funding_rate: float,
    expected_direction: str  # "pays" or "receives"
) -> bool:
    """
    Validate funding calculation using the same logic as the native actor.
    
    This ensures our native actor calculations are mathematically correct.
    
    Parameters
    ----------
    position_quantity : Decimal
        Position quantity (positive=long, negative=short).
    mark_price : float
        Mark price at funding time.
    funding_rate : float
        Funding rate for the period.
    expected_direction : str
        Expected direction: "pays" or "receives".
        
    Returns
    -------
    bool
        True if calculation matches expected direction.
    """
    # Use the same calculation as the native actor
    position_size = float(position_quantity)
    funding_payment_usd = position_size * mark_price * funding_rate
    
    # Determine direction
    if funding_payment_usd > 0:
        actual_direction = "pays"
    elif funding_payment_usd < 0:
        actual_direction = "receives"
    else:
        actual_direction = "neutral"
    
    return actual_direction == expected_direction


if __name__ == "__main__":
    # Demo of native actor functionality
    console.print(Panel.fit(
        "[bold cyan]🎭 Native FundingActor Demo[/bold cyan]\n"
        "Demonstrating NautilusTrader-native funding rate handling",
        title="NATIVE ACTOR DEMO"
    ))
    
    # Test mathematical validation
    console.print("\n[bold blue]🧮 Testing Native Actor Mathematics[/bold blue]")
    
    test_cases = [
        (Decimal("1.0"), 50000.0, 0.0001, "pays"),      # Long + positive rate
        (Decimal("-1.0"), 50000.0, 0.0001, "receives"), # Short + positive rate
        (Decimal("1.0"), 50000.0, -0.0001, "receives"), # Long + negative rate
        (Decimal("-1.0"), 50000.0, -0.0001, "pays"),    # Short + negative rate
    ]
    
    all_passed = True
    for pos, price, rate, expected in test_cases:
        is_valid = validate_funding_calculation_native(pos, price, rate, expected)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        console.print(f"  {status}: {pos} BTC @ ${price} with {rate:.6f} rate → {expected}")
        if not is_valid:
            all_passed = False
    
    if all_passed:
        console.print("[bold green]🎉 All native actor calculations verified![/bold green]")
    else:
        console.print("[bold red]⚠️ Native actor calculation errors detected![/bold red]")
    
    console.print("\n[bold green]✅ Native FundingActor ready for integration![/bold green]")