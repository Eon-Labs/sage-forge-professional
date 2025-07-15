"""
Funding rate manager for NautilusTrader integration.

Provides high-level orchestration of funding rate data and payment processing.
Integrates with NautilusTrader's timer system and event architecture.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Callable
from pathlib import Path

from nautilus_trader.common.component import Component, TimeEvent
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.position import Position
from rich.console import Console

from nautilus_test.funding.data import FundingRateUpdate, FundingPaymentEvent
from nautilus_test.funding.provider import FundingRateProvider
from nautilus_test.funding.calculator import FundingPaymentCalculator

console = Console()


class FundingRateManager(Component):
    """
    Manages funding rates and payments for perpetual futures.
    
    Integrates with NautilusTrader's Component architecture for:
    - Timer-based funding cycle monitoring
    - Event handling and routing
    - Logging and error handling
    - Configuration management
    """
    
    def __init__(
        self,
        funding_provider: Optional[FundingRateProvider] = None,
        funding_calculator: Optional[FundingPaymentCalculator] = None,
        cache_dir: Optional[Path] = None,
        enabled_instruments: Optional[Set[InstrumentId]] = None,
        funding_frequency_hours: int = 8,
        auto_start_timers: bool = True,
    ) -> None:
        """
        Initialize the funding rate manager.
        
        Parameters
        ----------
        funding_provider : FundingRateProvider, optional
            The funding rate data provider.
        funding_calculator : FundingPaymentCalculator, optional
            The funding payment calculator.
        cache_dir : Path, optional
            Directory for caching funding data.
        enabled_instruments : Set[InstrumentId], optional
            Instruments to monitor for funding (all if None).
        funding_frequency_hours : int
            Hours between funding payments (default 8).
        auto_start_timers : bool
            Whether to automatically start funding timers.
        """
        super().__init__(clock=None)
        
        # Initialize components
        self._funding_provider = funding_provider or FundingRateProvider(cache_dir=cache_dir)
        self._funding_calculator = funding_calculator or FundingPaymentCalculator()
        
        # Configuration
        self._enabled_instruments = enabled_instruments or set()
        self._funding_frequency = timedelta(hours=funding_frequency_hours)
        self._auto_start_timers = auto_start_timers
        
        # State tracking
        self._current_funding_rates: Dict[InstrumentId, FundingRateUpdate] = {}
        self._funding_timers: Dict[InstrumentId, str] = {}  # instrument -> timer_name
        self._funding_history: List[FundingPaymentEvent] = []
        
        # Callbacks
        self._funding_payment_handlers: List[Callable[[FundingPaymentEvent], None]] = []
        self._funding_rate_handlers: List[Callable[[FundingRateUpdate], None]] = []
        
        console.print("[green]✅ FundingRateManager initialized[/green]")
    
    def add_instrument(self, instrument_id: InstrumentId) -> None:
        """
        Add an instrument for funding rate monitoring.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to monitor.
        """
        self._enabled_instruments.add(instrument_id)
        
        if self._auto_start_timers and not self.is_disposed:
            self._start_funding_timer(instrument_id)
        
        console.print(f"[blue]📊 Added funding monitoring for {instrument_id}[/blue]")
    
    def remove_instrument(self, instrument_id: InstrumentId) -> None:
        """
        Remove an instrument from funding rate monitoring.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to stop monitoring.
        """
        self._enabled_instruments.discard(instrument_id)
        
        # Stop associated timer
        timer_name = self._funding_timers.get(instrument_id)
        if timer_name:
            self.stop_timer(timer_name)
            del self._funding_timers[instrument_id]
        
        # Remove current funding rate
        self._current_funding_rates.pop(instrument_id, None)
        
        console.print(f"[blue]📊 Removed funding monitoring for {instrument_id}[/blue]")
    
    def add_funding_payment_handler(
        self, 
        handler: Callable[[FundingPaymentEvent], None],
    ) -> None:
        """Add a handler for funding payment events."""
        self._funding_payment_handlers.append(handler)
    
    def add_funding_rate_handler(
        self, 
        handler: Callable[[FundingRateUpdate], None],
    ) -> None:
        """Add a handler for funding rate update events."""
        self._funding_rate_handlers.append(handler)
    
    async def load_historical_funding_rates(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        max_records: Optional[int] = None,
    ) -> List[FundingRateUpdate]:
        """
        Load historical funding rates for an instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to load funding rates for.
        start_time : datetime
            Start time for historical data.
        end_time : datetime, optional
            End time for historical data (defaults to now).
        max_records : int, optional
            Maximum number of records to load.
            
        Returns
        -------
        List[FundingRateUpdate]
            Historical funding rate updates.
        """
        end_time = end_time or datetime.now(timezone.utc)
        
        console.print(f"[cyan]📚 Loading historical funding rates for {instrument_id}[/cyan]")
        console.print(f"   Time range: {start_time} to {end_time}")
        
        funding_rates = await self._funding_provider.get_historical_funding_rates(
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time,
            max_records=max_records,
        )
        
        # Store most recent rate
        if funding_rates:
            self._current_funding_rates[instrument_id] = funding_rates[-1]
            console.print(f"[green]✅ Loaded {len(funding_rates)} historical funding rates[/green]")
        
        return funding_rates
    
    def calculate_funding_payments(
        self,
        positions: List[Position],
        mark_prices: Dict[InstrumentId, Price],
        funding_time: Optional[datetime] = None,
    ) -> List[FundingPaymentEvent]:
        """
        Calculate funding payments for current positions.
        
        Parameters
        ----------
        positions : List[Position]
            Positions to calculate funding for.
        mark_prices : Dict[InstrumentId, Price]
            Current mark prices.
        funding_time : datetime, optional
            Funding time (defaults to now).
            
        Returns
        -------
        List[FundingPaymentEvent]
            Calculated funding payment events.
        """
        funding_time = funding_time or datetime.now(timezone.utc)
        funding_events = []
        
        for position in positions:
            if position.instrument_id not in self._enabled_instruments:
                continue
            
            # Get current funding rate
            funding_rate = self._current_funding_rates.get(position.instrument_id)
            if not funding_rate:
                console.print(f"[yellow]⚠️ No funding rate available for {position.instrument_id}[/yellow]")
                continue
            
            # Get mark price
            mark_price = mark_prices.get(position.instrument_id)
            if not mark_price:
                console.print(f"[yellow]⚠️ No mark price available for {position.instrument_id}[/yellow]")
                continue
            
            # Calculate funding payment
            funding_event = self._funding_calculator.calculate_funding_payment(
                position=position,
                funding_rate_update=funding_rate,
                mark_price=mark_price,
            )
            
            if funding_event:
                funding_events.append(funding_event)
                self._funding_history.append(funding_event)
                
                # Notify handlers
                for handler in self._funding_payment_handlers:
                    try:
                        handler(funding_event)
                    except Exception as e:
                        self._log.error(f"Error in funding payment handler: {e}")
        
        if funding_events:
            console.print(f"[green]💰 Calculated {len(funding_events)} funding payments[/green]")
        
        return funding_events
    
    def get_funding_summary(self) -> Dict[str, any]:
        """Get summary of all funding activity."""
        return self._funding_calculator.get_funding_summary(self._funding_history)
    
    def _start_funding_timer(self, instrument_id: InstrumentId) -> None:
        """Start funding timer for an instrument."""
        timer_name = f"funding_{instrument_id}"
        
        # Calculate next funding time (align to 8-hour cycles)
        now = datetime.now(timezone.utc)
        next_funding = self._calculate_next_funding_time(now)
        
        # Set timer for next funding cycle
        time_to_next = next_funding - now
        
        self.set_timer(
            name=timer_name,
            callback=self._on_funding_timer,
            time_ns=dt_to_unix_nanos(next_funding),
            stop_time_ns=None,  # Recurring timer
        )
        
        self._funding_timers[instrument_id] = timer_name
        
        console.print(f"[blue]⏰ Funding timer set for {instrument_id}: next funding in {time_to_next}[/blue]")
    
    def _calculate_next_funding_time(self, current_time: datetime) -> datetime:
        """Calculate next funding time aligned to 8-hour cycles (00:00, 08:00, 16:00 UTC)."""
        # Funding occurs at 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        current_hour = current_time.hour
        
        # Find next funding hour
        next_hour = None
        for hour in funding_hours:
            if hour > current_hour:
                next_hour = hour
                break
        
        if next_hour is None:
            # Next funding is tomorrow at 00:00
            next_funding = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        else:
            # Next funding is today
            next_funding = current_time.replace(
                hour=next_hour, minute=0, second=0, microsecond=0
            )
        
        return next_funding
    
    def _on_funding_timer(self, event: TimeEvent) -> None:
        """Handle funding timer events."""
        # This would be called every 8 hours for funding calculations
        # In a real implementation, this would:
        # 1. Fetch current positions
        # 2. Get latest funding rates
        # 3. Calculate and apply funding payments
        # 4. Schedule next funding timer
        
        console.print(f"[yellow]⏰ Funding timer triggered: {event.name}[/yellow]")
        
        # Extract instrument_id from timer name
        instrument_id_str = event.name.replace("funding_", "")
        try:
            instrument_id = InstrumentId.from_str(instrument_id_str)
            
            # Reschedule for next funding cycle
            self._start_funding_timer(instrument_id)
            
        except Exception as e:
            self._log.error(f"Error handling funding timer {event.name}: {e}")
    
    def start(self) -> None:
        """Start the funding rate manager."""
        if self._auto_start_timers:
            for instrument_id in self._enabled_instruments:
                self._start_funding_timer(instrument_id)
        
        console.print("[green]🚀 FundingRateManager started[/green]")
    
    def stop(self) -> None:
        """Stop the funding rate manager."""
        # Stop all timers
        for timer_name in self._funding_timers.values():
            self.stop_timer(timer_name)
        
        self._funding_timers.clear()
        console.print("[yellow]⏹️ FundingRateManager stopped[/yellow]")
    
    def reset(self) -> None:
        """Reset the funding rate manager state."""
        self.stop()
        self._current_funding_rates.clear()
        self._funding_history.clear()
        console.print("[blue]🔄 FundingRateManager reset[/blue]")
    
    def dispose(self) -> None:
        """Dispose of the funding rate manager."""
        self.reset()
        super().dispose()
        console.print("[red]🗑️ FundingRateManager disposed[/red]")