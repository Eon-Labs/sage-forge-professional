"""
Funding payment calculator for perpetual futures positions.

Calculates funding payments based on position size, funding rates, and market prices.
Integrates with NautilusTrader's position and money handling.
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money, Price
from nautilus_trader.model.position import Position
from rich.console import Console

from nautilus_test.funding.data import FundingPaymentEvent, FundingRateUpdate

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

        # 🔍 CRITICAL FIX #2: Enhanced funding calculation with mathematical validation
        console.print("[yellow]🔍 DEBUG: Starting funding payment calculation...[/yellow]")

        # Calculate funding payment with detailed logging
        # Formula: payment = position_size × mark_price × funding_rate
        position_size_signed = position.signed_qty  # Positive for long, negative for short
        position_size_float = float(position_size_signed)
        mark_price_float = float(mark_price)
        funding_rate_float = funding_rate_update.funding_rate

        console.print(f"[blue]📊 DEBUG: Position size: {position_size_float:.6f} BTC[/blue]")
        console.print(f"[blue]💰 DEBUG: Mark price: ${mark_price_float:.2f}[/blue]")
        console.print(f"[blue]📈 DEBUG: Funding rate: {funding_rate_float:.6f} ({funding_rate_float*100:.4f}%)[/blue]")

        # Calculate notional value
        notional_value = abs(position_size_float) * mark_price_float
        console.print(f"[cyan]💼 DEBUG: Notional value: ${notional_value:.2f}[/cyan]")

        # Funding payment calculation with mathematical validation
        # payment = position_size × mark_price × funding_rate
        payment_amount_signed = position_size_float * mark_price_float * funding_rate_float
        payment_amount_abs = abs(payment_amount_signed)

        console.print(f"[cyan]🧮 DEBUG: Calculation: {position_size_float:.6f} × ${mark_price_float:.2f} × {funding_rate_float:.6f} = ${payment_amount_signed:.6f}[/cyan]")
        console.print(f"[cyan]💸 DEBUG: Absolute payment amount: ${payment_amount_abs:.6f}[/cyan]")

        # 🚨 MATHEMATICAL VALIDATION: Verify calculation makes sense
        # For a typical funding rate (0.01% = 0.0001), minimum expected payment:
        # 0.002 BTC × $117,000 × 0.0001 = $0.0234 per interval
        # For 4 intervals over 2 days = minimum $0.0936

        expected_minimum_per_interval = 0.002 * 117000 * 0.0001  # $0.0234
        if abs(position_size_float) >= 0.002 and mark_price_float >= 100000 and abs(funding_rate_float) >= 0.0001:
            expected_minimum = expected_minimum_per_interval
            if payment_amount_abs < expected_minimum * 0.1:  # Allow for 90% variance
                console.print(f"[red]🚨 WARNING: Payment amount ${payment_amount_abs:.6f} seems too low![/red]")
                console.print(f"[red]📊 Expected minimum: ${expected_minimum:.6f} per interval[/red]")
                console.print("[red]🔍 Verify funding rate and position size are correct[/red]")

        # Funding payment logic:
        # Positive payment = outgoing (you pay)
        # Negative payment = incoming (you receive)
        console.print(f"[cyan]💰 DEBUG: Payment direction: {'OUTGOING (pay)' if payment_amount_signed > 0 else 'INCOMING (receive)'}[/cyan]")

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
        positions: list[Position],
        funding_rate_update: FundingRateUpdate,
        mark_prices: dict[InstrumentId, Price],
    ) -> list[FundingPaymentEvent]:
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
        funding_events: list[FundingPaymentEvent],
    ) -> dict[str, any]:
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
            console.print("[red]❌ Funding calculation validation failed:[/red]")
            console.print(f"   Position: {position_size}, Mark Price: {mark_price}")
            console.print(f"   Funding Rate: {funding_rate}")
            console.print(f"   Expected: {expected_payment}, Calculated: {calculated_payment}")
            console.print(f"   Difference: {diff} (tolerance: {tolerance})")

        return is_valid
