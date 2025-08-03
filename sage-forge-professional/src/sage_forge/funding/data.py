"""
SAGE-Forge funding rate data structures.

Provides NautilusTrader-native data structures for funding rate integration
with enhanced SAGE-Forge configuration and validation.
"""

from decimal import Decimal
from typing import Any

from nautilus_trader.core.data import Data
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money
from rich.console import Console

from sage_forge.core.config import get_config

console = Console()


class FundingRateUpdate(Data):
    """
    SAGE-Forge enhanced funding rate update for perpetual futures contracts.
    
    Features:
    - NautilusTrader-native data model integration
    - SAGE-Forge configuration system integration
    - Enhanced validation and quality checks
    - Professional logging and analytics
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        funding_rate: float,
        funding_time: int,  # Unix timestamp in nanoseconds
        mark_price: float | None = None,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        """
        Initialize SAGE-Forge funding rate update with validation.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument identifier.
        funding_rate : float
            The funding rate (e.g., 0.0001 = 0.01%).
        funding_time : int
            The funding time as Unix timestamp in nanoseconds.
        mark_price : float, optional
            The mark price at funding time.
        ts_event : int
            The event timestamp in nanoseconds.
        ts_init : int
            The initialization timestamp in nanoseconds.
        """
        # SAGE-Forge configuration integration
        self.sage_config = get_config()
        
        # Validate funding rate bounds
        max_funding_rate = self.sage_config.get_funding_config().get("max_funding_rate", 0.01)  # 1%
        if abs(funding_rate) > max_funding_rate:
            console.print(f"[yellow]⚠️ High funding rate detected: {funding_rate*100:.4f}% (max: {max_funding_rate*100:.2f}%)[/yellow]")
        
        # Set attributes following NautilusTrader data pattern
        self.instrument_id = instrument_id
        self.funding_rate = funding_rate
        self.funding_time = funding_time
        self.mark_price = mark_price

        # Set timestamps with underscores (following NautilusTrader pattern)
        self._ts_event = ts_event
        self._ts_init = ts_init

    @property
    def ts_event(self) -> int:
        """The event timestamp in nanoseconds."""
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """The initialization timestamp in nanoseconds."""
        return self._ts_init
    
    @property
    def funding_rate_bps(self) -> float:
        """Get funding rate in basis points."""
        return self.funding_rate * 10000
    
    @property
    def funding_rate_percentage(self) -> float:
        """Get funding rate as percentage."""
        return self.funding_rate * 100

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"instrument_id={self.instrument_id}, "
            f"funding_rate={self.funding_rate:.6f} ({self.funding_rate_percentage:.4f}%), "
            f"funding_time={self.funding_time}, "
            f"mark_price={self.mark_price}, "
            f"ts_event={self.ts_event}, "
            f"ts_init={self.ts_init})"
        )

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "FundingRateUpdate":
        """Create from dictionary representation with SAGE-Forge validation."""
        return FundingRateUpdate(
            instrument_id=InstrumentId.from_str(values["instrument_id"]),
            funding_rate=values["funding_rate"],
            funding_time=values["funding_time"],
            mark_price=values.get("mark_price"),
            ts_event=values["ts_event"],
            ts_init=values["ts_init"],
        )

    @staticmethod
    def to_dict(obj: "FundingRateUpdate") -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": type(obj).__name__,
            "instrument_id": str(obj.instrument_id),
            "funding_rate": obj.funding_rate,
            "funding_time": obj.funding_time,
            "mark_price": obj.mark_price,
            "ts_event": obj.ts_event,
            "ts_init": obj.ts_init,
        }


class FundingPaymentEvent(Data):
    """
    SAGE-Forge enhanced funding payment event for account balance integration.
    
    Features:
    - NautilusTrader event system integration
    - SAGE-Forge configuration and risk management
    - Enhanced P&L tracking and analytics  
    - Professional logging and reporting
    """

    def __init__(
        self,
        event_id: UUID4,
        instrument_id: InstrumentId,
        payment_amount: Money,
        funding_rate: float,
        position_size: Decimal,
        mark_price: float,
        is_payment: bool,  # True if paying funding, False if receiving
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        """
        Initialize SAGE-Forge funding payment event with risk validation.
        
        Parameters
        ----------
        event_id : UUID4
            The unique event identifier.
        instrument_id : InstrumentId
            The instrument identifier.
        payment_amount : Money
            The funding payment amount (positive = paid, negative = received).
        funding_rate : float
            The funding rate applied.
        position_size : Decimal
            The position size at funding time.
        mark_price : float
            The mark price used for calculation.
        is_payment : bool
            Whether this is a payment (True) or receipt (False).
        ts_event : int
            The event timestamp in nanoseconds.
        ts_init : int
            The initialization timestamp in nanoseconds.
        """
        # SAGE-Forge configuration integration
        self.sage_config = get_config()
        
        # Risk management validation
        max_payment = self.sage_config.get_funding_config().get("max_single_payment", 1000.0)
        if float(payment_amount) > max_payment:
            console.print(f"[red]⚠️ Large funding payment: ${float(payment_amount):.2f} (max: ${max_payment:.2f})[/red]")
        
        self.event_id = event_id
        self.instrument_id = instrument_id
        self.payment_amount = payment_amount
        self.funding_rate = funding_rate
        self.position_size = position_size
        self.mark_price = mark_price
        self.is_payment = is_payment

        super().__init__(ts_event, ts_init)
        
        # SAGE-Forge analytics logging
        direction = "payment" if is_payment else "receipt"
        console.print(f"[cyan]💰 Funding {direction}: ${float(payment_amount):.4f} @ {funding_rate*100:.4f}%[/cyan]")

    @property
    def impact_on_pnl(self) -> float:
        """Calculate impact on P&L (negative for payments, positive for receipts)."""
        return -float(self.payment_amount) if self.is_payment else float(self.payment_amount)

    def __repr__(self) -> str:
        direction = "payment" if self.is_payment else "receipt"
        return (
            f"{type(self).__name__}("
            f"event_id={self.event_id}, "
            f"instrument_id={self.instrument_id}, "
            f"payment_amount={self.payment_amount}, "
            f"funding_rate={self.funding_rate:.6f} ({self.funding_rate*100:.4f}%), "
            f"position_size={self.position_size}, "
            f"direction={direction}, "
            f"pnl_impact={self.impact_on_pnl:.4f}, "
            f"ts_event={self.ts_event})"
        )

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "FundingPaymentEvent":
        """Create from dictionary representation."""
        return FundingPaymentEvent(
            event_id=UUID4.from_str(values["event_id"]),
            instrument_id=InstrumentId.from_str(values["instrument_id"]),
            payment_amount=Money.from_str(values["payment_amount"]),
            funding_rate=values["funding_rate"],
            position_size=Decimal(values["position_size"]),
            mark_price=values["mark_price"],
            is_payment=values["is_payment"],
            ts_event=values["ts_event"],
            ts_init=values["ts_init"],
        )

    @staticmethod
    def to_dict(obj: "FundingPaymentEvent") -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": type(obj).__name__,
            "event_id": str(obj.event_id),
            "instrument_id": str(obj.instrument_id),
            "payment_amount": str(obj.payment_amount),
            "funding_rate": obj.funding_rate,
            "position_size": str(obj.position_size),
            "mark_price": obj.mark_price,
            "is_payment": obj.is_payment,
            "ts_event": obj.ts_event,
            "ts_init": obj.ts_init,
        }