"""
Funding payment calculator for perpetual futures positions.

Calculates funding payments based on position size, funding rates, and market prices.
Integrates with NautilusTrader's position and money handling.
"""

from decimal import Decimal
from typing import Dict, List, Optional

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money, Price
from nautilus_trader.model.position import Position
from rich.console import Console

from nautilus_test.funding.data import FundingRateUpdate, FundingPaymentEvent

console = Console()


class FundingPaymentCalculator:
    """
    Calculates funding payments for perpetual futures positions.
    
    Follows established crypto market conventions:
    - Positive funding rate: Long positions pay short positions
    - Negative funding rate: Short positions pay long positions
    - Payment = Position Size × Mark Price × Funding Rate
    """
    
    def __init__(self) -> None:
        """Initialize the funding payment calculator."""
        pass
    
    def calculate_funding_payment(
        self,
        position: Position,
        funding_rate_update: FundingRateUpdate,
        mark_price: Price,
    ) -> Optional[FundingPaymentEvent]:
        """
        Calculate funding payment for a position.
        
        Parameters
        ----------
        position : Position
            The position to calculate funding for.
        funding_rate_update : FundingRateUpdate
            The funding rate data.
        mark_price : Price
            The mark price for payment calculation.
            
        Returns
        -------
        FundingPaymentEvent | None
            The funding payment event, or None if no position.
        """
        # Skip if no position or position is flat
        if position.is_closed or position.size == 0:
            return None
        
        # Verify instrument matches
        if position.instrument_id != funding_rate_update.instrument_id:
            console.print(f"[yellow]⚠️ Instrument mismatch: {position.instrument_id} vs {funding_rate_update.instrument_id}[/yellow]")
            return None
        
        # Calculate funding payment
        # Formula: payment = position_size × mark_price × funding_rate
        position_size_signed = position.signed_qty  # Positive for long, negative for short
        notional_value = abs(float(position_size_signed)) * float(mark_price)
        
        # Funding payment logic:
        # payment = position_size × mark_price × funding_rate
        # Positive payment = outgoing (you pay)
        # Negative payment = incoming (you receive)
        payment_amount_signed = float(position_size_signed) * float(mark_price) * funding_rate_update.funding_rate
        
        # Create Money object (positive = outgoing payment, negative = incoming payment)
        settlement_currency = position.instrument.settlement_currency
        payment_amount = Money(abs(payment_amount_signed), settlement_currency)
        is_payment = payment_amount_signed > 0
        
        # Create funding payment event
        funding_event = FundingPaymentEvent(
            event_id=UUID4(),
            instrument_id=position.instrument_id,
            payment_amount=payment_amount,
            funding_rate=funding_rate_update.funding_rate,
            position_size=position_size_signed,
            mark_price=float(mark_price),
            is_payment=is_payment,
            ts_event=funding_rate_update.ts_event,
            ts_init=funding_rate_update.ts_init,
        )
        
        return funding_event
    
    def calculate_multiple_funding_payments(
        self,
        positions: List[Position],
        funding_rate_update: FundingRateUpdate,
        mark_prices: Dict[InstrumentId, Price],
    ) -> List[FundingPaymentEvent]:
        """
        Calculate funding payments for multiple positions.
        
        Parameters
        ----------
        positions : List[Position]
            List of positions to calculate funding for.
        funding_rate_update : FundingRateUpdate
            The funding rate data.
        mark_prices : Dict[InstrumentId, Price]
            Mark prices for each instrument.
            
        Returns
        -------
        List[FundingPaymentEvent]
            List of funding payment events.
        """
        funding_events = []
        
        for position in positions:
            if position.instrument_id == funding_rate_update.instrument_id:
                mark_price = mark_prices.get(position.instrument_id)
                if mark_price is None:
                    console.print(f"[yellow]⚠️ No mark price available for {position.instrument_id}[/yellow]")
                    continue
                
                funding_event = self.calculate_funding_payment(
                    position=position,
                    funding_rate_update=funding_rate_update,
                    mark_price=mark_price,
                )
                
                if funding_event:
                    funding_events.append(funding_event)
        
        return funding_events
    
    def get_funding_summary(
        self,
        funding_events: List[FundingPaymentEvent],
    ) -> Dict[str, any]:
        """
        Generate a summary of funding payments.
        
        Parameters
        ----------
        funding_events : List[FundingPaymentEvent]
            List of funding payment events.
            
        Returns
        -------
        Dict[str, any]
            Summary statistics of funding payments.
        """
        if not funding_events:
            return {
                "total_events": 0,
                "total_paid": Money(0, "USD"),
                "total_received": Money(0, "USD"),
                "net_funding": Money(0, "USD"),
                "instruments": [],
            }
        
        # Group by currency for proper aggregation
        payments_by_currency = {}
        receipts_by_currency = {}
        instruments = set()
        
        for event in funding_events:
            currency = event.payment_amount.currency
            amount = event.payment_amount.as_double()
            
            instruments.add(str(event.instrument_id))
            
            if event.is_payment:
                payments_by_currency[currency] = payments_by_currency.get(currency, 0) + amount
            else:
                receipts_by_currency[currency] = receipts_by_currency.get(currency, 0) + amount
        
        # Calculate totals (assuming single currency for simplicity)
        main_currency = funding_events[0].payment_amount.currency
        total_paid = payments_by_currency.get(main_currency, 0)
        total_received = receipts_by_currency.get(main_currency, 0)
        net_funding = total_received - total_paid
        
        return {
            "total_events": len(funding_events),
            "total_paid": Money(total_paid, main_currency),
            "total_received": Money(total_received, main_currency),
            "net_funding": Money(net_funding, main_currency),
            "instruments": sorted(list(instruments)),
            "payments_by_currency": payments_by_currency,
            "receipts_by_currency": receipts_by_currency,
        }
    
    def validate_funding_calculation(
        self,
        position_size: Decimal,
        mark_price: float,
        funding_rate: float,
        expected_payment: float,
        tolerance: float = 1e-8,
    ) -> bool:
        """
        Validate a funding calculation against expected result.
        
        Parameters
        ----------
        position_size : Decimal
            The position size (signed).
        mark_price : float
            The mark price.
        funding_rate : float
            The funding rate.
        expected_payment : float
            The expected payment amount.
        tolerance : float
            Tolerance for floating point comparison.
            
        Returns
        -------
        bool
            True if calculation matches expected result.
        """
        # Calculate payment using same logic as calculate_funding_payment
        notional_value = abs(float(position_size)) * mark_price
        
        # Apply simplified logic: payment = position_size × mark_price × funding_rate
        calculated_payment = float(position_size) * mark_price * funding_rate
        
        # Check if within tolerance
        diff = abs(calculated_payment - expected_payment)
        is_valid = diff <= tolerance
        
        if not is_valid:
            console.print(f"[red]❌ Funding calculation validation failed:[/red]")
            console.print(f"   Position: {position_size}, Mark Price: {mark_price}")
            console.print(f"   Funding Rate: {funding_rate}")
            console.print(f"   Expected: {expected_payment}, Calculated: {calculated_payment}")
            console.print(f"   Difference: {diff} (tolerance: {tolerance})")
        
        return is_valid