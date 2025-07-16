"""
Funding rate data structures for NautilusTrader integration.

Following NautilusTrader's data model patterns for consistency with existing
adapter and event architecture.
"""

from decimal import Decimal
from typing import Any

from nautilus_trader.core.data import Data
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money


class FundingRateUpdate(Data):
    """
    Represents a funding rate update for a perpetual futures contract.
    
    Follows NautilusTrader's data model pattern similar to BinanceFuturesMarkPriceUpdate.
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
        Initialize a funding rate update.
        
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
        # Set attributes following NautilusTrader data pattern
        self.instrument_id = instrument_id
        self.funding_rate = funding_rate
        self.funding_time = funding_time
        self.mark_price = mark_price

        # Set timestamps with underscores (following BinanceFuturesMarkPriceUpdate pattern)
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"instrument_id={self.instrument_id}, "
            f"funding_rate={self.funding_rate}, "
            f"funding_time={self.funding_time}, "
            f"mark_price={self.mark_price}, "
            f"ts_event={self.ts_event}, "
            f"ts_init={self.ts_init})"
        )

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "FundingRateUpdate":
        """Create from dictionary representation."""
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
    Represents a funding payment/charge event for an account.
    
    Integrates with NautilusTrader's event system for account balance updates.
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
        Initialize a funding payment event.
        
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
        self.event_id = event_id
        self.instrument_id = instrument_id
        self.payment_amount = payment_amount
        self.funding_rate = funding_rate
        self.position_size = position_size
        self.mark_price = mark_price
        self.is_payment = is_payment

        super().__init__(ts_event, ts_init)

    def __repr__(self) -> str:
        direction = "payment" if self.is_payment else "receipt"
        return (
            f"{type(self).__name__}("
            f"event_id={self.event_id}, "
            f"instrument_id={self.instrument_id}, "
            f"payment_amount={self.payment_amount}, "
            f"funding_rate={self.funding_rate}, "
            f"position_size={self.position_size}, "
            f"direction={direction}, "
            f"ts_event={self.ts_event})"
        )

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "FundingPaymentEvent":
        """Create from dictionary representation."""
        return FundingPaymentEvent(
            event_id=UUID4(values["event_id"]),
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
